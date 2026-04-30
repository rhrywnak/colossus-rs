//! Format-agnostic document text extraction trait.
//!
//! Every file format (PDF, DOCX, TXT) has a specific extractor that
//! implements the `DocumentExtractor` trait. The pipeline calls
//! `extractor.extract(path)` and gets back `Vec<ExtractedPage>`
//! regardless of the source format.
//!
//! ## Rust Learning: Trait objects for runtime polymorphism
//!
//! The pipeline stores extractors as `Box<dyn DocumentExtractor>`.
//! This is Rust's equivalent of an interface pointer — the concrete
//! type is erased at compile time, and method calls go through a
//! vtable at runtime. The trait must be "object-safe" to allow this:
//! no generic methods, no `Self` in return position, no associated
//! types with bounds that reference `Self`.
//!
//! We pay a small vtable indirection cost per method call — negligible
//! compared to the I/O and computation inside each extractor.

use std::path::Path;

/// A page of extracted text from a document.
///
/// This is the common output type for ALL extractors. Downstream
/// consumers (chunking, LLM extraction, search) never know which
/// extractor produced the text — they see only this struct.
#[derive(Debug, Clone)]
pub struct ExtractedPage {
    /// 1-based page number.
    ///
    /// For formats without natural page boundaries (TXT, email),
    /// the entire content is page 1 unless form-feed characters
    /// (\x0C) are present, in which case each section between
    /// form-feeds is a separate page.
    pub page_number: i32,

    /// The extracted text content for this page.
    pub text_content: String,

    /// True if this page was processed by OCR (scanned PDF pages).
    /// Always false for DOCX and TXT extractors.
    pub is_ocr: bool,
}

/// Trait for document text extraction.
///
/// Every format-specific extractor implements this trait. The pipeline
/// constructs the appropriate extractor based on the detected file
/// format and calls `extract()`. The pipeline never imports or names
/// the concrete extractor type — it works through `Box<dyn DocumentExtractor>`.
///
/// ## Implementor contract
///
/// - `extract()` must return pages in document order (page 1 first).
/// - Page numbers must be 1-based and contiguous.
/// - Empty pages should still be returned (with empty `text_content`)
///   to preserve page numbering.
/// - Errors must include enough context to diagnose the failure:
///   file path, page number if applicable, and the underlying cause.
pub trait DocumentExtractor: Send + Sync {
    /// Extract text from the document at the given path.
    ///
    /// Returns pages in order. The caller owns the returned vector.
    fn extract(&self, file_path: &Path) -> Result<Vec<ExtractedPage>, crate::PdfError>;

    /// Human-readable name for logging and the audit trail.
    ///
    /// Examples: "pdf_oxide", "docx_rust", "plain_text"
    fn name(&self) -> &str;
}
