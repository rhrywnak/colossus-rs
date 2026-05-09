//! DOCX (Word OOXML) text extractor.
//!
//! Uses the `docx-rust` crate to parse the OOXML structure and pull
//! plain text out of every paragraph in the document body.
//!
//! ## Page detection (limitation)
//!
//! OOXML page breaks live inside Run elements as
//! `<w:br w:type="page"/>`. Splitting on them requires walking
//! `Run.content` and matching on `Break` variants — a follow-up
//! task once we have a real DOCX corpus to test against.
//!
//! Until then, all body content is returned as a single page.
//! Every extraction emits a `tracing::info!` so this limitation
//! is visible in production logs (never silent).
//!
//! ## Tables (limitation)
//!
//! `BodyContent::Table` cells contain paragraphs of their own.
//! We currently skip tables; supporting them well needs a recursive
//! walk that respects cell ordering. Will revisit when a real
//! document requires it.
//!
//! ## Rust Learning: Two-step open-then-parse
//!
//! `DocxFile` reads and owns the file bytes. `Docx<'_>` is a
//! parsed view that **borrows** from those bytes. Both values must
//! live in the same scope — if you wrote
//! `let docx = DocxFile::from_file(p)?.parse()?;` the temporary
//! `DocxFile` would drop at the end of the statement and `docx`
//! would dangle. Binding the file to a named local first, then
//! parsing, makes the borrow legal.

use std::path::Path;

use docx_rust::document::BodyContent;
use docx_rust::DocxFile;

use crate::document_extractor::{DocumentExtractor, ExtractedPage};
use crate::PdfError;

/// Human-readable extractor name returned by `DocumentExtractor::name`.
///
/// Mirrors `pdf_oxide` — the convention is "name of the underlying library".
const EXTRACTOR_NAME: &str = "docx_rust";

/// Single newline used to join paragraphs into a flat text block.
///
/// Per the spec we use `\n`, not `\n\n`. Empty paragraphs are
/// preserved so the join already produces the visual blank line
/// the user wrote.
const PARAGRAPH_SEPARATOR: &str = "\n";

/// Page number assigned to the single page we currently emit.
const SINGLE_PAGE_NUMBER: i32 = 1;

/// Extracts plain text from `.docx` (Word OOXML) files.
///
/// Stateless — a single instance can extract any number of files.
pub struct DocxExtractor;

impl DocumentExtractor for DocxExtractor {
    fn extract(&self, file_path: &Path) -> Result<Vec<ExtractedPage>, PdfError> {
        let docx_file = DocxFile::from_file(file_path).map_err(|e| {
            PdfError::OpenError(format!(
                "Failed to read DOCX file '{}': {:?}",
                file_path.display(),
                e
            ))
        })?;

        let docx = docx_file.parse().map_err(|e| {
            PdfError::OpenError(format!(
                "Failed to parse DOCX file '{}': {:?}",
                file_path.display(),
                e
            ))
        })?;

        // ## Rust Learning: Pattern-matching enum variants in iteration
        //
        // `body.content` holds many `BodyContent` variants
        // (Paragraph, Table, Sdt, ...). `if let` lets us pull text
        // only from the variant we currently support, ignoring the
        // rest. The compiler enforces exhaustiveness in `match` —
        // when we extend support to Tables later, the type system
        // will guide us to every call site that needs updating.
        let mut paragraphs: Vec<String> = Vec::new();
        for content in &docx.document.body.content {
            if let BodyContent::Paragraph(p) = content {
                paragraphs.push(p.text());
            }
        }

        let full_text = paragraphs.join(PARAGRAPH_SEPARATOR);

        tracing::info!(
            path = %file_path.display(),
            paragraph_count = paragraphs.len(),
            "DOCX extraction: page detection not supported, returning all content as page 1"
        );

        Ok(vec![ExtractedPage {
            page_number: SINGLE_PAGE_NUMBER,
            text_content: full_text,
            is_ocr: false,
        }])
    }

    fn name(&self) -> &str {
        EXTRACTOR_NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_returns_open_error() {
        let extractor = DocxExtractor;
        let result = extractor.extract(Path::new("/nonexistent/path/does/not/exist.docx"));
        match result {
            Err(PdfError::OpenError(msg)) => {
                assert!(
                    msg.contains("/nonexistent/path/does/not/exist.docx"),
                    "expected error to include path, got: {msg}"
                );
            }
            other => panic!("expected OpenError, got {other:?}"),
        }
    }

    /// Round-trip test: build a 3-paragraph DOCX with `docx-rust`'s
    /// writer, extract, verify all three strings appear.
    #[test]
    fn extract_simple_docx() {
        use docx_rust::document::Paragraph;
        use docx_rust::Docx;

        let mut docx = Docx::default();
        docx.document
            .push(Paragraph::default().push_text("First paragraph content"));
        docx.document
            .push(Paragraph::default().push_text("Second paragraph content"));
        docx.document
            .push(Paragraph::default().push_text("Third paragraph content"));

        let temp = tempfile::NamedTempFile::new().unwrap();
        docx.write_file(temp.path()).unwrap();

        let extractor = DocxExtractor;
        let pages = extractor.extract(temp.path()).unwrap();

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page_number, 1);
        assert!(!pages[0].is_ocr);

        let text = &pages[0].text_content;
        assert!(
            text.contains("First paragraph content"),
            "expected first paragraph in extracted text, got: {text}"
        );
        assert!(
            text.contains("Second paragraph content"),
            "expected second paragraph in extracted text, got: {text}"
        );
        assert!(
            text.contains("Third paragraph content"),
            "expected third paragraph in extracted text, got: {text}"
        );
    }
}
