//! # colossus-pdf
//!
//! PDF text extraction and page-grounding for Colossus applications.
//!
//! This crate provides tools to:
//! - Extract text from PDFs with page-level granularity
//! - Map text snippets to the pages they appear on (page grounding)
//! - Search for text across all pages of a document
//!
//! ## Rust Learning: Crate organization
//!
//! A library crate's `lib.rs` is the root module. It declares
//! sub-modules with `mod` and re-exports key types with `pub use`
//! so consumers can write `use colossus_pdf::PdfTextExtractor`
//! instead of `use colossus_pdf::extractor::PdfTextExtractor`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use colossus_pdf::{PdfTextExtractor, PageGrounder};
//!
//! let mut extractor = PdfTextExtractor::open("document.pdf").unwrap();
//! let page_text = extractor.extract_page(1).unwrap();
//!
//! let mut grounder = PageGrounder::new(&mut extractor);
//! let results = grounder.ground_snippets(&["some quote from the document"]).unwrap();
//! for result in &results {
//!     println!("{}: page {:?}", result.snippet, result.page_number);
//! }
//! ```

mod classifier;
mod document_extractor;
mod error;
mod extractor;
mod normalize;
mod page_grounder;
mod pdf_oxide_adapter;
mod plain_text_extractor;
mod text_search;

// --- Public API re-exports ---

pub use classifier::{ContentType, PageClassification, PdfClassification};
pub use document_extractor::{DocumentExtractor, ExtractedPage};
pub use error::PdfError;
pub use extractor::{PageText, PdfTextExtractor};
pub use normalize::{normalize_text, NormalizationRule};
pub use page_grounder::{GroundingResult, MatchType, PageGrounder};
pub use pdf_oxide_adapter::PdfOxideAdapter;
pub use plain_text_extractor::PlainTextExtractor;
pub use text_search::{search_text, SearchConfig, SearchHit};
