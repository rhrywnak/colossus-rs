//! classifier.rs
//!
//! PDF content classification — distinguishes text-based pages from scanned pages.
//! Exists so the extraction pipeline can route only scanned pages through OCR,
//! avoiding the cost and accuracy loss of OCR'ing pages that already have a
//! clean text layer.
//!
//! ## Rust Learning: Plain-data types with serde
//!
//! `PdfClassification` and `PageClassification` are plain data records with
//! no methods beyond what `derive` provides. `Serialize`/`Deserialize` come
//! from `serde`, and `Debug`/`Clone` are standard. Keeping the shape simple
//! lets the type round-trip through JSON (e.g. stored in a database JSONB
//! column or returned from an API) with zero extra code.

use serde::{Deserialize, Serialize};

/// Minimum characters on a page to consider it text-based.
/// Pages below this threshold are treated as scanned/needing OCR.
/// Matches the OCR char_threshold in the colossus-legal ExtractText step.
pub const TEXT_CHAR_THRESHOLD: usize = 50;

/// Result of classifying a PDF's content type.
///
/// Produced by `PdfTextExtractor::classify()`. Tells the caller
/// which pages are text-based (directly extractable) and which
/// are scanned (need OCR). This enables smart routing in the
/// extraction pipeline — only OCR the pages that need it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfClassification {
    /// Overall content type of the document.
    pub content_type: ContentType,
    /// Total number of pages.
    pub page_count: usize,
    /// Number of pages with extractable text (>= threshold chars).
    pub text_pages: usize,
    /// Number of pages classified as scanned (< threshold chars).
    pub scanned_pages: usize,
    /// Page numbers (1-indexed) that need OCR.
    pub pages_needing_ocr: Vec<u32>,
    /// Total characters extracted across all pages.
    pub total_chars: usize,
    /// Per-page classification details.
    pub page_details: Vec<PageClassification>,
}

/// Content type of a PDF document.
///
/// Represents the aggregate classification across all pages. Downstream
/// consumers use this to decide whether to skip OCR entirely (`TextBased`),
/// OCR the whole document (`Scanned`), or process pages selectively (`Mixed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    /// All pages contain extractable text.
    TextBased,
    /// All pages are scanned images with no extractable text.
    Scanned,
    /// Some pages are text-based, some are scanned.
    Mixed,
}

impl ContentType {
    /// String representation for database storage.
    ///
    /// Kept as an explicit method rather than relying on serde so that
    /// SQL binding code can use `ContentType::Mixed.as_str()` directly
    /// without going through a `serde_json` round-trip. Matches the
    /// enum SQL binding pattern used across colossus-rs.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextBased => "text_based",
            Self::Scanned => "scanned",
            Self::Mixed => "mixed",
        }
    }
}

/// Classification of a single page.
///
/// One of these is produced per page during classification. `has_text` and
/// `needs_ocr` are the inverse of each other — both are stored explicitly
/// so downstream code reads naturally (`if page.needs_ocr { ... }`) without
/// forcing readers to negate a boolean at each use site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageClassification {
    /// Page number (1-indexed).
    pub page_number: u32,
    /// Number of characters extracted from this page.
    pub char_count: usize,
    /// Whether this page has enough text to be considered text-based.
    pub has_text: bool,
    /// Whether this page needs OCR to extract text.
    pub needs_ocr: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PdfTextExtractor;

    /// Helper: create a small 3-page text-based PDF using pdf_oxide's
    /// DocumentBuilder. Paragraphs are deliberately longer than
    /// `TEXT_CHAR_THRESHOLD` (50 chars) per page so the fixture exercises
    /// a genuine text-based document end-to-end.
    fn create_test_pdf_bytes() -> Vec<u8> {
        use pdf_oxide::writer::DocumentBuilder;

        let mut builder = DocumentBuilder::new();

        builder
            .letter_page()
            .heading(1, "Test Document")
            .paragraph(
                "This is page one with some sample text used to exercise \
                 classification of a text-based PDF document end to end.",
            )
            .done();

        builder
            .letter_page()
            .heading(1, "Chapter Two")
            .paragraph(
                "The quick brown fox jumps over the lazy dog repeatedly, \
                 providing plenty of characters for the classifier to see.",
            )
            .done();

        builder
            .letter_page()
            .heading(1, "Final Page")
            .paragraph(
                "Conclusion, summary of findings, and closing remarks for \
                 the document to ensure this page exceeds the threshold.",
            )
            .done();

        builder.build().expect("Failed to build test PDF")
    }

    #[test]
    fn text_based_pdf_classification() {
        let bytes = create_test_pdf_bytes();
        let mut extractor = PdfTextExtractor::from_bytes(bytes)
            .expect("Failed to open test PDF");

        let result = extractor.classify().expect("classify failed");

        assert_eq!(result.content_type, ContentType::TextBased);
        assert_eq!(result.page_count, 3);
        assert_eq!(result.text_pages, 3);
        assert_eq!(result.scanned_pages, 0);
        assert!(
            result.pages_needing_ocr.is_empty(),
            "no pages should need OCR, got: {:?}",
            result.pages_needing_ocr
        );
        assert!(result.total_chars > 0, "should have extracted characters");
        assert_eq!(result.page_details.len(), 3);
        assert_eq!(result.page_details[0].page_number, 1);
        assert_eq!(result.page_details[2].page_number, 3);
        for page in &result.page_details {
            assert!(page.has_text, "page {} should have text", page.page_number);
            assert!(!page.needs_ocr, "page {} should not need OCR", page.page_number);
        }
    }

    #[test]
    fn content_type_as_str() {
        assert_eq!(ContentType::TextBased.as_str(), "text_based");
        assert_eq!(ContentType::Scanned.as_str(), "scanned");
        assert_eq!(ContentType::Mixed.as_str(), "mixed");
    }

    #[test]
    fn classification_serializes_to_json() {
        let classification = PdfClassification {
            content_type: ContentType::Mixed,
            page_count: 2,
            text_pages: 1,
            scanned_pages: 1,
            pages_needing_ocr: vec![2],
            total_chars: 123,
            page_details: vec![
                PageClassification {
                    page_number: 1,
                    char_count: 123,
                    has_text: true,
                    needs_ocr: false,
                },
                PageClassification {
                    page_number: 2,
                    char_count: 0,
                    has_text: false,
                    needs_ocr: true,
                },
            ],
        };

        let json = serde_json::to_value(&classification).expect("serialize failed");

        assert_eq!(json["content_type"], "mixed");
        assert_eq!(json["page_count"], 2);
        assert_eq!(json["text_pages"], 1);
        assert_eq!(json["scanned_pages"], 1);
        assert_eq!(json["pages_needing_ocr"], serde_json::json!([2]));
        assert_eq!(json["total_chars"], 123);
        assert_eq!(json["page_details"].as_array().unwrap().len(), 2);
        assert_eq!(json["page_details"][1]["needs_ocr"], true);
    }

    #[test]
    fn page_classification_fields() {
        let with_text = PageClassification {
            page_number: 1,
            char_count: 200,
            has_text: true,
            needs_ocr: false,
        };
        assert!(with_text.has_text);
        assert!(!with_text.needs_ocr);
        assert_eq!(with_text.page_number, 1);
        assert_eq!(with_text.char_count, 200);

        let scanned = PageClassification {
            page_number: 7,
            char_count: 3,
            has_text: false,
            needs_ocr: true,
        };
        assert!(!scanned.has_text);
        assert!(scanned.needs_ocr);
        assert_eq!(scanned.page_number, 7);
        assert_eq!(scanned.char_count, 3);
    }
}
