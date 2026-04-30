//! Adapter that wraps `PdfTextExtractor` to implement `DocumentExtractor`.
//!
//! This is the bridge between the existing PDF extraction code and
//! the new trait-based interface. The adapter delegates all work to
//! `PdfTextExtractor` — it adds no new logic, just the trait boundary.
//!
//! ## Rust Learning: Adapter pattern in Rust
//!
//! We can't implement `DocumentExtractor` directly on `PdfTextExtractor`
//! because `PdfTextExtractor` has a different API shape (open + extract_page
//! per page vs. extract all pages at once). The adapter translates
//! between the two interfaces. This is the classic Adapter pattern —
//! same purpose as in OOP, but implemented as a wrapper struct
//! rather than inheritance.

use std::path::Path;

use crate::document_extractor::{DocumentExtractor, ExtractedPage};
use crate::extractor::PdfTextExtractor;
use crate::PdfError;

/// Human-readable extractor name returned by `DocumentExtractor::name`.
const EXTRACTOR_NAME: &str = "pdf_oxide";

/// Adapter that makes `PdfTextExtractor` implement `DocumentExtractor`.
///
/// Constructed with no arguments — all configuration comes from
/// the file being extracted. Stateless and reusable across files.
///
/// Note: `PdfTextExtractor::open` accepts `&str`, so this adapter
/// converts the trait's `&Path` argument via `to_string_lossy`.
/// On non-UTF-8 paths the conversion is lossy, but the underlying
/// `pdf_oxide` open call would fail anyway on such paths.
pub struct PdfOxideAdapter;

impl DocumentExtractor for PdfOxideAdapter {
    fn extract(&self, file_path: &Path) -> Result<Vec<ExtractedPage>, PdfError> {
        // ## Rust Learning: Path → &str via Cow
        //
        // `to_string_lossy` returns `Cow<'_, str>`. Borrowing it with
        // `&` produces `&Cow<str>`, which auto-derefs to `&str` when
        // passed to a function expecting `&str`. No allocation on
        // already-UTF-8 paths.
        let path_str = file_path.to_string_lossy();
        let mut extractor = PdfTextExtractor::open(&path_str)?;

        let page_count = extractor.page_count();
        let mut pages = Vec::with_capacity(page_count as usize);

        for page_num in 1..=page_count {
            // `extract_page` returns `&str` borrowed from the cache;
            // we own a copy so the result outlives the extractor.
            let text = extractor.extract_page(page_num)?.to_owned();
            pages.push(ExtractedPage {
                page_number: page_num as i32,
                text_content: text,
                is_ocr: false,
            });
        }

        Ok(pages)
    }

    fn name(&self) -> &str {
        EXTRACTOR_NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_oxide_adapter_name_is_stable() {
        let adapter = PdfOxideAdapter;
        assert_eq!(adapter.name(), "pdf_oxide");
    }

    #[test]
    fn pdf_oxide_adapter_missing_file_returns_open_error() {
        let adapter = PdfOxideAdapter;
        let result = adapter.extract(Path::new("/nonexistent/path/does/not/exist.pdf"));
        assert!(matches!(result, Err(PdfError::OpenError(_))));
    }
}
