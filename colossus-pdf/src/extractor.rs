//! PDF text extraction with page-level granularity.
//!
//! # Rust Learning: Newtype pattern
//!
//! `PdfTextExtractor` wraps `pdf_oxide::PdfDocument` to provide a
//! focused API for text extraction. This is the "newtype" pattern —
//! wrapping an external type to control the public interface and
//! add domain-specific methods without exposing the underlying library.

use crate::classifier::{
    ContentType, PageClassification, PdfClassification, TEXT_CHAR_THRESHOLD,
};
use crate::error::PdfError;
use pdf_oxide::PdfDocument;

/// Extracted text for a single page.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PageText {
    /// 0-indexed page number.
    pub page_index: u32,
    /// 1-indexed page number (what humans see).
    pub page_number: u32,
    /// The extracted text content.
    pub text: String,
    /// Number of characters extracted.
    pub char_count: usize,
}

/// Opens a PDF and provides page-level text extraction.
///
/// This is the primary entry point for colossus-pdf. All text
/// extraction flows through this type.
///
/// # Rust Learning: Interior caching
///
/// `page_cache` is a `Vec<Option<String>>` pre-sized to the page count.
/// Each slot starts as `None` and gets filled on first extraction.
/// This avoids re-extracting pages that have already been read while
/// keeping the API simple — callers don't need to manage a separate cache.
pub struct PdfTextExtractor {
    doc: PdfDocument,
    page_count: u32,
    /// Cache of extracted text per page (lazy — populated on demand).
    page_cache: Vec<Option<String>>,
    /// Holds the temp file alive when opened via `from_bytes`.
    /// `PdfDocument` may read lazily from disk, so we must keep
    /// the file around for the extractor's entire lifetime.
    _temp_file: Option<tempfile::NamedTempFile>,
}

impl PdfTextExtractor {
    /// Open a PDF from a file path.
    pub fn open(path: &str) -> Result<Self, PdfError> {
        let mut doc = PdfDocument::open(path)
            .map_err(|e| PdfError::OpenError(format!("{e}")))?;

        let page_count = doc
            .page_count()
            .map_err(|e| PdfError::OpenError(format!("Failed to read page count: {e}")))?
            as u32;

        tracing::debug!("Opened PDF: {path} ({page_count} pages)");

        let page_cache = vec![None; page_count as usize];

        Ok(Self {
            doc,
            page_count,
            page_cache,
            _temp_file: None,
        })
    }

    /// Open a PDF from bytes in memory.
    ///
    /// # Rust Learning: Temp files for library compatibility
    ///
    /// `pdf_oxide` 0.3.x only supports opening PDFs from file paths.
    /// We write the bytes to a temporary file and open that. The temp
    /// file is kept alive for the lifetime of the extractor via `_temp_file`,
    /// because `PdfDocument` may read lazily from disk.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, PdfError> {
        use std::io::Write;

        // Write bytes to a named temp file that persists while we hold the handle
        let mut temp_file = tempfile::NamedTempFile::new()?;
        temp_file.write_all(&bytes)?;
        temp_file.flush()?;

        let path = temp_file.path().to_string_lossy().to_string();

        let mut doc = PdfDocument::open(&path)
            .map_err(|e| PdfError::OpenError(format!("{e}")))?;

        let page_count = doc
            .page_count()
            .map_err(|e| PdfError::OpenError(format!("Failed to read page count: {e}")))?
            as u32;

        tracing::debug!("Opened PDF from bytes ({page_count} pages)");

        let page_cache = vec![None; page_count as usize];

        Ok(Self {
            doc,
            page_count,
            page_cache,
            _temp_file: Some(temp_file),
        })
    }

    /// Total number of pages in the PDF.
    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    /// Extract text from a specific page (1-indexed, matching human page numbers).
    ///
    /// Results are cached — calling this twice for the same page
    /// returns the cached result without re-extracting.
    ///
    /// # Rust Learning: Returning `&str` from cached data
    ///
    /// We return `&str` (a borrowed string slice) rather than cloning the
    /// `String`. The caller gets a read-only view into our cache, which is
    /// zero-cost. The lifetime of the returned `&str` is tied to `&mut self`,
    /// so Rust ensures the cache isn't invalidated while the reference exists.
    pub fn extract_page(&mut self, page_number: u32) -> Result<&str, PdfError> {
        if page_number == 0 || page_number > self.page_count {
            return Err(PdfError::PageOutOfRange(page_number, self.page_count));
        }

        let index = (page_number - 1) as usize;

        // If not cached, extract and store
        if self.page_cache[index].is_none() {
            let text = self
                .doc
                .extract_text(index)
                .map_err(|e| PdfError::ExtractionError {
                    page: page_number,
                    message: format!("{e}"),
                })?;

            tracing::debug!(
                "Extracted page {page_number}: {} chars",
                text.len()
            );

            self.page_cache[index] = Some(text);
        }

        // Safe: we just ensured the slot is Some
        Ok(self.page_cache[index].as_deref().unwrap_or_default())
    }

    /// Extract text from ALL pages. Returns a Vec of PageText.
    ///
    /// This populates the internal cache for all pages.
    pub fn extract_all_pages(&mut self) -> Result<Vec<PageText>, PdfError> {
        let mut pages = Vec::with_capacity(self.page_count as usize);

        for page_num in 1..=self.page_count {
            let text = self.extract_page(page_num)?.to_owned();
            let char_count = text.len();

            pages.push(PageText {
                page_index: page_num - 1,
                page_number: page_num,
                text,
                char_count,
            });
        }

        Ok(pages)
    }

    /// Check if a specific page has extractable text (text layer present).
    ///
    /// Returns false for scanned pages with no text layer.
    pub fn page_has_text(&mut self, page_number: u32) -> Result<bool, PdfError> {
        let text = self.extract_page(page_number)?;
        Ok(!text.trim().is_empty())
    }

    /// Check if the document has ANY extractable text on any page.
    pub fn has_text_layer(&mut self) -> Result<bool, PdfError> {
        for page_num in 1..=self.page_count {
            if self.page_has_text(page_num)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Classify the PDF's content type by extracting text from each page.
    ///
    /// Pages with fewer than `TEXT_CHAR_THRESHOLD` (50) characters are
    /// classified as scanned (needing OCR). The extracted text is cached
    /// internally — subsequent calls to `extract_page` return cached
    /// results. Classification + extraction costs no more than
    /// extraction alone.
    pub fn classify(&mut self) -> Result<PdfClassification, PdfError> {
        let mut page_details = Vec::with_capacity(self.page_count as usize);
        let mut text_pages = 0usize;
        let mut scanned_pages = 0usize;
        let mut pages_needing_ocr = Vec::new();
        let mut total_chars = 0usize;

        for page_num in 1..=self.page_count {
            let text = self.extract_page(page_num)?;
            let char_count = text.trim().len();
            let has_text = char_count >= TEXT_CHAR_THRESHOLD;
            let needs_ocr = !has_text;

            total_chars += char_count;

            if has_text {
                text_pages += 1;
            } else {
                scanned_pages += 1;
                pages_needing_ocr.push(page_num);
            }

            page_details.push(PageClassification {
                page_number: page_num,
                char_count,
                has_text,
                needs_ocr,
            });
        }

        let content_type = if scanned_pages == 0 {
            ContentType::TextBased
        } else if text_pages == 0 {
            ContentType::Scanned
        } else {
            ContentType::Mixed
        };

        Ok(PdfClassification {
            content_type,
            page_count: self.page_count as usize,
            text_pages,
            scanned_pages,
            pages_needing_ocr,
            total_chars,
            page_details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small 3-page test PDF using pdf_oxide's DocumentBuilder.
    ///
    /// Page 1: "Test Document\nThis is page one with some sample text."
    /// Page 2: "Chapter Two\nThe quick brown fox jumps over the lazy dog."
    /// Page 3: "Final Page\nConclusion and summary of findings."
    fn create_test_pdf_bytes() -> Vec<u8> {
        use pdf_oxide::writer::DocumentBuilder;

        let mut builder = DocumentBuilder::new();

        builder
            .letter_page()
            .heading(1, "Test Document")
            .paragraph("This is page one with some sample text.")
            .done();

        builder
            .letter_page()
            .heading(1, "Chapter Two")
            .paragraph("The quick brown fox jumps over the lazy dog.")
            .done();

        builder
            .letter_page()
            .heading(1, "Final Page")
            .paragraph("Conclusion and summary of findings.")
            .done();

        builder.build().expect("Failed to build test PDF")
    }

    #[test]
    fn test_open_from_bytes_and_page_count() {
        let bytes = create_test_pdf_bytes();
        let extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");
        assert_eq!(extractor.page_count(), 3);
    }

    #[test]
    fn test_extract_page_contains_expected_text() {
        let bytes = create_test_pdf_bytes();
        let mut extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");

        let page1 = extractor.extract_page(1).expect("Failed to extract page 1");
        assert!(
            page1.contains("Test Document"),
            "Page 1 should contain 'Test Document', got: {page1}"
        );

        let page2 = extractor.extract_page(2).expect("Failed to extract page 2");
        assert!(
            page2.contains("quick brown fox"),
            "Page 2 should contain 'quick brown fox', got: {page2}"
        );
    }

    #[test]
    fn test_page_has_text() {
        let bytes = create_test_pdf_bytes();
        let mut extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");

        assert!(extractor.page_has_text(1).expect("page_has_text failed"));
        assert!(extractor.page_has_text(3).expect("page_has_text failed"));
    }

    #[test]
    fn test_page_out_of_range() {
        let bytes = create_test_pdf_bytes();
        let mut extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");

        // Page 0 is invalid (1-indexed)
        assert!(matches!(
            extractor.extract_page(0),
            Err(PdfError::PageOutOfRange(0, 3))
        ));

        // Page 4 is beyond the 3-page document
        assert!(matches!(
            extractor.extract_page(4),
            Err(PdfError::PageOutOfRange(4, 3))
        ));
    }

    #[test]
    fn test_caching_returns_same_result() {
        let bytes = create_test_pdf_bytes();
        let mut extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");

        let first_call = extractor.extract_page(1).expect("extract failed").to_owned();
        let second_call = extractor.extract_page(1).expect("extract failed").to_owned();
        assert_eq!(first_call, second_call, "Cached result should match");
    }

    #[test]
    fn test_extract_all_pages() {
        let bytes = create_test_pdf_bytes();
        let mut extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");

        let pages = extractor.extract_all_pages().expect("extract_all failed");
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[2].page_number, 3);
        assert!(pages[0].char_count > 0);
    }
}
