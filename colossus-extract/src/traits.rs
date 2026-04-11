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
use crate::types::{
    ChunkExtractionResult, ExtractedEntity, ExtractionResult, KnownEntity, ResolvedEntity,
    TextChunk,
};

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

/// Extracts entities and relationships from a single text chunk.
///
/// This is the provider-agnostic interface for per-chunk extraction.
/// Each LLM provider implements this trait differently:
/// - Anthropic: uses rig's prompt_typed with structured output
/// - OpenAI: uses rig's prompt_typed with structured output
/// - Ollama/vLLM: uses raw JSON prompt with llm_json repair fallback
///
/// The pipeline calls this trait, never a specific provider.
///
/// ## Rust Learning: async_trait + error types
///
/// The trait returns `PipelineError` which the pipeline can match on
/// to decide whether to retry, skip, or fail. The `Send + Sync` bounds
/// allow the trait object to be shared across tokio tasks for concurrent
/// chunk processing.
#[async_trait]
pub trait ChunkExtractor: Send + Sync {
    /// Extract entities and relationships from a single text chunk.
    ///
    /// Arguments:
    /// - chunk_text: the text of this chunk
    /// - schema_json: the extraction schema as a JSON value (entity types,
    ///   relationship types, patterns) — passed to the LLM prompt
    /// - prompt_template: the extraction prompt template with placeholders
    /// - examples: optional few-shot examples for the LLM
    ///
    /// Returns the extracted nodes and relationships, or an error.
    async fn extract_chunk(
        &self,
        chunk_text: &str,
        schema_json: &serde_json::Value,
        prompt_template: &str,
        examples: &str,
    ) -> Result<ChunkExtractionResult, PipelineError>;

    /// Human-readable name of the LLM provider/model.
    fn model_name(&self) -> &str;
}
