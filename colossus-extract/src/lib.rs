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
//! - `EmbeddingProvider` trait — converts text to vectors for indexing and retrieval
//! - `EntityResolver` trait — deduplicates entities
//! - `TextSplitter` trait — chunks large documents (future use)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use colossus_extract::{ExtractionSchema, PromptBuilder, LlmProvider};
//! ```

pub mod config;
pub mod error;
pub mod prompt;
pub mod providers;
pub mod resolver;
pub mod schema;
pub mod splitter;
pub mod structure_splitter;
pub mod traits;
pub mod types;

// --- Public API re-exports ---
pub use config::ConfigAccess;
pub use error::PipelineError;
pub use prompt::{PromptArtifact, PromptBuilder};
pub use providers::embedding_provider_from_env;
pub use providers::embedding_provider_from_lookup;
pub use providers::llm_provider_from_env;
pub use providers::llm_provider_from_lookup;
pub use providers::AnthropicProvider;
pub use providers::EnvLookup;
pub use providers::FastembedProvider;
pub use providers::VllmEmbeddingProvider;
pub use providers::VllmProvider;
pub use resolver::NormalizedEntityResolver;
pub use schema::{
    CompletenessRule, DocumentCategory, EntityCategory, ExtractionSchema, GroundingMode,
};
pub use splitter::FixedSizeSplitter;
pub use structure_splitter::StructureAwareSplitter;
pub use traits::{EmbeddingProvider, EntityResolver, LlmProvider, LlmResponse, TextSplitter};
pub use types::{
    AtomicUnit, ExtractedEntity, ExtractedRelationship, ExtractionResult, GroundingStatus,
    KnownEntity, PruningStats, ResolutionMethod, ResolvedEntity, TextChunk,
};
