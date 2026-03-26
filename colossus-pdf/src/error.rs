//! Error types for colossus-pdf.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PdfError {
    #[error("Failed to open PDF: {0}")]
    OpenError(String),

    #[error("Failed to extract text from page {page}: {message}")]
    ExtractionError { page: u32, message: String },

    #[error("Page {0} out of range (document has {1} pages)")]
    PageOutOfRange(u32, u32),

    #[error("PDF has no text layer (scanned document)")]
    NoTextLayer,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
