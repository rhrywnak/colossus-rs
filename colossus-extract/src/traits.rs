//! Pipeline traits — the extension points for domain-specific implementations.
//!
//! ## Rust Learning: async_trait
//!
//! Rust doesn't natively support async methods in traits (yet — it's stabilizing).
//! The `#[async_trait]` macro desugars `async fn` into a boxed future, allowing
//! trait objects and dynamic dispatch. Every implementation must also use
//! `#[async_trait]` so the signatures match.

use async_trait::async_trait;

use crate::error::PipelineError;
use crate::types::{ExtractedEntity, ExtractionResult, KnownEntity, ResolvedEntity, TextChunk};

/// Model-agnostic LLM interface for extraction.
///
/// Implementations: ClaudeProvider (API), VllmProvider (self-hosted).
/// Each provider handles its own HTTP client, authentication, and response parsing.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a prompt to the LLM and get structured extraction results.
    async fn extract(
        &self,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<ExtractionResult, PipelineError>;

    /// Human-readable model name for logging and cost tracking.
    fn model_name(&self) -> &str;

    /// Cost per input token in USD, if known.
    fn cost_per_input_token(&self) -> Option<f64>;

    /// Cost per output token in USD, if known.
    fn cost_per_output_token(&self) -> Option<f64>;
}

/// Deduplicates extracted entities against known entities.
///
/// Start with exact + normalized matching (sufficient for well-defined domains).
/// Add fuzzy matching (strsim crate) and semantic matching (Qdrant) as needed.
#[async_trait]
pub trait EntityResolver: Send + Sync {
    /// Resolve a set of extracted entities against known entities.
    async fn resolve(
        &self,
        entities: Vec<ExtractedEntity>,
        existing: &[KnownEntity],
    ) -> Result<Vec<ResolvedEntity>, PipelineError>;
}

/// Splits document text into chunks for LLM processing.
///
/// ## Rust Learning: Why a trait?
///
/// Different splitting strategies may be needed for different document
/// types. Legal complaints have numbered paragraphs. Research papers
/// have sections. A trait lets us swap strategies without changing the
/// pipeline code.
pub trait TextSplitter: Send + Sync {
    /// Split text into chunks. Each chunk has the text content and its
    /// position index in the original document.
    fn split(&self, text: &str) -> Vec<TextChunk>;
}

