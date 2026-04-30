//! File format detection and extractor factory.
//!
//! Uses the `mimetype-detector` crate to identify file formats
//! from magic bytes (file content), not file extensions. This
//! catches misnamed files — a `.pdf` extension on a `.docx` file
//! is detected correctly.
//!
//! Also hosts the `extractor_for_format` factory because the
//! factory is logically "given a format, hand back the matching
//! extractor" — it sits between `DocumentFormat` and the concrete
//! `DocumentExtractor` impls, so co-locating it with the format
//! keeps dispatch in one place.
//!
//! ## Rust Learning: Content-based vs extension-based detection
//!
//! File extensions are unreliable — users rename files, servers
//! strip extensions, downloads get mangled. Magic bytes (the
//! first few bytes of the file) are embedded in the file content
//! and can't be changed by renaming. The `mimetype-detector`
//! crate reads up to 3 KB of the file header to make its
//! determination.

use std::path::Path;

use mimetype_detector::detect_file;

use crate::docx_extractor::DocxExtractor;
use crate::pdf_oxide_adapter::PdfOxideAdapter;
use crate::plain_text_extractor::PlainTextExtractor;
use crate::DocumentExtractor;
use crate::PdfError;

/// MIME string for PDF documents (per IANA / `application/pdf`).
const MIME_PDF: &str = "application/pdf";

/// MIME string for OOXML Word documents (`.docx`).
///
/// The full IANA registration string is unwieldy but is the only
/// correct identifier — substrings like "wordprocessingml" can
/// match other OOXML payloads (templates, macro-enabled docs).
const MIME_DOCX: &str =
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

/// MIME string for plain text files.
const MIME_TEXT_PLAIN: &str = "text/plain";

/// File extension used for the `.txt` fallback path.
///
/// Matched lower-case after `Path::extension()`. Files with
/// `.TXT` or `.Txt` will not match — caller can normalize the
/// path themselves if they want case-insensitive behaviour.
const TXT_EXTENSION: &str = "txt";

/// Detected document format.
///
/// This enum lists the formats Colossus knows how to extract
/// text from. Adding a new format means: add a variant here,
/// implement `DocumentExtractor` for it, and add a match arm
/// to `extractor_for_format`. The compiler will tell you which
/// match arms are missing — that's the point of using `match`
/// without a wildcard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    /// PDF document (text-based or scanned — that distinction is
    /// made later by `PdfTextExtractor::classify`).
    Pdf,
    /// Word 2007+ document (`.docx`, OOXML).
    Docx,
    /// Plain text file.
    PlainText,
}

/// Detect the document format from file content (magic bytes).
///
/// Returns the detected format, or an error if the format is
/// unsupported or the file cannot be read. Every error message
/// includes the file path so it's actionable in production logs.
///
/// ## Fallback for `.txt`
///
/// `mimetype-detector` sometimes returns `application/octet-stream`
/// or another generic type for plain text files that lack
/// distinctive magic bytes (e.g. an empty file, or one that
/// happens to start with bytes the detector reads as binary).
/// When detection produces an unexpected MIME for a file whose
/// extension is `.txt`, we fall back to `PlainText` and emit a
/// `tracing::warn!` so the fallback is **visible** in logs —
/// never silent. Files without `.txt` extension fail with a
/// descriptive error so the caller knows what was detected.
pub fn detect_format(file_path: &Path) -> Result<DocumentFormat, PdfError> {
    let mime = detect_file(file_path).map_err(|e| {
        PdfError::OpenError(format!(
            "Failed to detect MIME type for '{}': {}",
            file_path.display(),
            e
        ))
    })?;

    // ## Rust Learning: Stringly typed dispatch with named constants
    //
    // `mime.to_string()` returns an owned `String` like
    // `"application/pdf"` or `"text/plain; charset=utf-8"` —
    // text types often carry a charset parameter. We strip
    // parameters before comparing against our `&'static str`
    // constants so the match is the *only* place these MIME
    // strings are mentioned.
    let mime_str = mime.to_string();
    let mime_main = mime_type_only(&mime_str);
    match mime_main {
        MIME_PDF => Ok(DocumentFormat::Pdf),
        MIME_DOCX => Ok(DocumentFormat::Docx),
        MIME_TEXT_PLAIN => Ok(DocumentFormat::PlainText),
        _ => fallback_or_error(file_path, &mime_str),
    }
}

/// Strip optional MIME parameters and return the bare `type/subtype`.
///
/// `mimetype-detector` reports text types as
/// `text/plain; charset=utf-8`. We compare only the canonical
/// `type/subtype` against our known constants. `split(';').next()`
/// always yields at least one element for any input, so the
/// `unwrap_or` is purely defensive.
fn mime_type_only(full_mime: &str) -> &str {
    full_mime.split(';').next().unwrap_or(full_mime).trim()
}

/// Helper for the unsupported-MIME branch of `detect_format`.
///
/// Pulled out so the main function stays under the readability
/// threshold and the fallback policy is testable on its own.
fn fallback_or_error(file_path: &Path, detected_mime: &str) -> Result<DocumentFormat, PdfError> {
    let has_txt_extension = file_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(TXT_EXTENSION))
        .unwrap_or(false);

    if has_txt_extension {
        tracing::warn!(
            path = %file_path.display(),
            detected = %detected_mime,
            "MIME detection returned non-text/plain MIME for .txt extension; falling back to PlainText"
        );
        Ok(DocumentFormat::PlainText)
    } else {
        Err(PdfError::OpenError(format!(
            "Unsupported document format '{}' for file '{}'",
            detected_mime,
            file_path.display()
        )))
    }
}

/// Create the appropriate extractor for the given format.
///
/// Returns a boxed trait object — the caller never names the
/// concrete type and works through dynamic dispatch.
///
/// ## Rust Learning: Factory function returning `Box<dyn Trait>`
///
/// This is Rust's version of the Factory pattern. We can't
/// return the concrete type because the caller doesn't know it
/// (it depends on runtime format detection). `Box<dyn DocumentExtractor>`
/// erases the concrete type and provides dynamic dispatch
/// through the vtable. The trait must be **object-safe** for
/// this to work: no generic methods, no `Self` in return
/// position. `DocumentExtractor` satisfies these constraints.
pub fn extractor_for_format(format: DocumentFormat) -> Box<dyn DocumentExtractor> {
    match format {
        DocumentFormat::Pdf => Box::new(PdfOxideAdapter),
        DocumentFormat::Docx => Box::new(DocxExtractor),
        DocumentFormat::PlainText => Box::new(PlainTextExtractor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal PDF header — `%PDF-1.4` plus enough bytes for the
    /// detector to lock on. Real PDFs have much more, but the
    /// magic-byte detection only needs the first few bytes.
    const PDF_MAGIC_FIXTURE: &[u8] =
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< >>\nendobj\n";

    #[test]
    fn detect_pdf_format() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(PDF_MAGIC_FIXTURE).unwrap();
        f.flush().unwrap();

        let format = detect_format(f.path()).unwrap();
        assert_eq!(format, DocumentFormat::Pdf);
    }

    #[test]
    fn detect_txt_format() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"This is plain ASCII text\nwith multiple lines.\n")
            .unwrap();
        f.flush().unwrap();

        let format = detect_format(f.path()).unwrap();
        assert_eq!(format, DocumentFormat::PlainText);
    }

    #[test]
    fn detect_unsupported_format_returns_open_error() {
        // Random binary content with null bytes that should not
        // match any text or known-document MIME type.
        let binary_garbage: &[u8] = &[
            0x00, 0xFF, 0x00, 0xFF, 0xDE, 0xAD, 0xBE, 0xEF, 0x13, 0x37, 0x42, 0x42,
        ];

        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(binary_garbage).unwrap();
        f.flush().unwrap();

        let result = detect_format(f.path());
        match result {
            Err(PdfError::OpenError(msg)) => {
                // Tempfile path has no .txt extension, so we go
                // to the unsupported branch (not the fallback).
                assert!(
                    msg.contains("Unsupported"),
                    "expected 'Unsupported' in error message, got: {msg}"
                );
            }
            Ok(format) => panic!("expected error for binary garbage, got format {format:?}"),
            Err(other) => panic!("expected OpenError, got {other:?}"),
        }
    }

    #[test]
    fn detect_docx_format() {
        use docx_rust::document::Paragraph;
        use docx_rust::Docx;

        // The spec listed this test as conditional on docx-rust's
        // writer being usable in-test. The DocxExtractor round-trip
        // test confirms it is, so we exercise the detector here too.
        let mut docx = Docx::default();
        docx.document
            .push(Paragraph::default().push_text("docx detection fixture"));

        let temp = tempfile::NamedTempFile::new().unwrap();
        docx.write_file(temp.path()).unwrap();

        let format = detect_format(temp.path()).unwrap();
        assert_eq!(format, DocumentFormat::Docx);
    }

    #[test]
    fn factory_returns_correct_extractor_names() {
        assert_eq!(
            extractor_for_format(DocumentFormat::Pdf).name(),
            "pdf_oxide"
        );
        assert_eq!(
            extractor_for_format(DocumentFormat::Docx).name(),
            "docx_rust"
        );
        assert_eq!(
            extractor_for_format(DocumentFormat::PlainText).name(),
            "plain_text"
        );
    }
}
