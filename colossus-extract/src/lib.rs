//! # colossus-extract
//!
//! Generic document extraction pipeline framework for Colossus applications.
//!
//! This crate provides types, traits, and utilities for building LLM-powered
//! extraction pipelines that transform unstructured documents into structured
//! knowledge. Domain-specific behavior is controlled by YAML schema files
//! and prompt templates loaded at runtime.
//!
//! ## Architecture
//!
//! The pipeline follows the neo4j-graphrag-python Component pattern, translated
//! to Rust idioms:
//!
//! - `ExtractionSchema` — defines what to extract (loaded from YAML)
//! - `PromptBuilder` — constructs LLM prompts from templates
//! - `LlmProvider` trait — model-agnostic LLM interface
//! - `EntityResolver` trait — deduplicates entities
//! - `TextSplitter` trait — chunks large documents (future use)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use colossus_extract::{ExtractionSchema, PromptBuilder, LlmProvider};
//! ```

pub mod error;
pub mod prompt;
pub mod resolver;
pub mod schema;
pub mod splitter;
pub mod traits;
pub mod types;

// --- Public API re-exports ---
pub use error::PipelineError;
pub use prompt::{PromptArtifact, PromptBuilder};
pub use schema::{
    CompletenessRule, DocumentCategory, EntityCategory, ExtractionSchema, GroundingMode,
};
pub use resolver::NormalizedEntityResolver;
pub use splitter::FixedSizeSplitter;
pub use traits::{EntityResolver, LlmProvider, TextSplitter};
pub use types::{
    ExtractedEntity, ExtractedRelationship, ExtractionResult, GroundingStatus, KnownEntity,
    PruningStats, ResolutionMethod, ResolvedEntity, TextChunk,
};
