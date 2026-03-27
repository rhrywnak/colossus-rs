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
use crate::types::{ExtractedEntity, ExtractionResult, KnownEntity, ResolvedEntity};

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
/// Not needed for colossus-legal (documents fit in one call).
/// Designed for colossus-ai where documents may exceed LLM context windows.
pub trait TextSplitter: Send + Sync {
    /// Split text into chunks with optional overlap.
    fn split(&self, text: &str) -> Vec<String>;
}
