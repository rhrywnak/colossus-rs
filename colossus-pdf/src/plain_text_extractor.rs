//! Plain text file extractor.
//!
//! Reads a .txt file and splits it into pages using form-feed
//! characters (\x0C) as page boundaries. If no form-feeds are
//! present, the entire file content becomes page 1.
//!
//! ## Rust Learning: The simplest extractor
//!
//! This is the baseline — no parsing, no binary format decoding,
//! just file I/O. It shows the minimum a `DocumentExtractor` must do:
//! read bytes, produce `ExtractedPage` values, handle errors.

use std::path::Path;

use crate::document_extractor::{DocumentExtractor, ExtractedPage};
use crate::PdfError;

/// Human-readable extractor name returned by `DocumentExtractor::name`.
const EXTRACTOR_NAME: &str = "plain_text";

/// ASCII form-feed character used as a page boundary in plain text files.
///
/// Many text editors and print utilities insert `\x0C` between pages.
/// We split on this character so that printable text dumps preserve
/// their original page structure.
const FORM_FEED: char = '\x0C';

/// Extracts text from plain `.txt` files.
///
/// Splits on form-feed characters (`\x0C`) for page boundaries.
/// If no form-feeds are present, returns the entire content as page 1.
///
/// Stateless and reusable — a single instance can extract any number
/// of files concurrently because all state lives on the stack of
/// `extract()`.
pub struct PlainTextExtractor;

impl DocumentExtractor for PlainTextExtractor {
    fn extract(&self, file_path: &Path) -> Result<Vec<ExtractedPage>, PdfError> {
        // ## Rust Learning: surface I/O errors with context
        //
        // `read_to_string` returns `io::Error`, which `PdfError` can
        // accept via its `Io(#[from] io::Error)` variant. But that
        // discards the file path. We map to `OpenError` so the error
        // string includes the path that failed to open — much easier
        // to diagnose in production logs.
        let content = std::fs::read_to_string(file_path).map_err(|e| {
            PdfError::OpenError(format!(
                "Failed to read text file '{}': {}",
                file_path.display(),
                e
            ))
        })?;

        // ## Rust Learning: Form-feed as page break
        //
        // The ASCII form-feed character (\x0C) is the traditional
        // page break marker in plain text files. Many text editors
        // and print utilities insert it between pages. If present,
        // each section between form-feeds becomes a page.
        let pages: Vec<&str> = content.split(FORM_FEED).collect();

        let extracted: Vec<ExtractedPage> = pages
            .iter()
            .enumerate()
            .map(|(i, text)| ExtractedPage {
                page_number: (i + 1) as i32,
                text_content: (*text).to_string(),
                is_ocr: false,
            })
            .collect();

        if extracted.is_empty() {
            // Defensive: `str::split` on an empty string yields a
            // single empty slice, so `extracted` is normally non-empty.
            // Keep this branch so the contract "always at least one
            // page" survives any future refactor of the split logic.
            Ok(vec![ExtractedPage {
                page_number: 1,
                text_content: String::new(),
                is_ocr: false,
            }])
        } else {
            Ok(extracted)
        }
    }

    fn name(&self) -> &str {
        EXTRACTOR_NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn plain_text_extractor_single_page() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "Hello world\nLine two").unwrap();

        let extractor = PlainTextExtractor;
        let pages = extractor.extract(f.path()).unwrap();

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[0].text_content, "Hello world\nLine two");
        assert!(!pages[0].is_ocr);
    }

    #[test]
    fn plain_text_extractor_form_feed_pages() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "Page one\x0CPage two\x0CPage three").unwrap();

        let extractor = PlainTextExtractor;
        let pages = extractor.extract(f.path()).unwrap();

        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[0].text_content, "Page one");
        assert_eq!(pages[1].page_number, 2);
        assert_eq!(pages[1].text_content, "Page two");
        assert_eq!(pages[2].page_number, 3);
        assert_eq!(pages[2].text_content, "Page three");
    }

    #[test]
    fn plain_text_extractor_name_is_stable() {
        let extractor = PlainTextExtractor;
        assert_eq!(extractor.name(), "plain_text");
    }

    #[test]
    fn plain_text_extractor_missing_file_returns_open_error() {
        let extractor = PlainTextExtractor;
        let result = extractor.extract(Path::new("/nonexistent/path/does/not/exist.txt"));
        assert!(matches!(result, Err(PdfError::OpenError(_))));
    }
}
