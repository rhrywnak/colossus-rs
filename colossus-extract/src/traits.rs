//! Pipeline traits — the extension points for domain-specific implementations.
//!
//! ## Rust Learning: async_trait
//!
//! Rust doesn't natively support async methods in traits (yet — it's stabilizing).
//! The `#[async_trait]` macro desugars `async fn` into a boxed future, allowing
//! trait objects and dynamic dispatch. Every implementation must also use
//! `#[async_trait]` so the signatures match.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::PipelineError;
use crate::types::{ExtractedEntity, KnownEntity, ResolvedEntity, TextChunk};

/// Unified LLM interface for all call sites: extraction, synthesis, decomposition.
///
/// Pattern: neo4j-graphrag-python `LLMInterface` (production canonical reference).
/// All three callers use the same interface:
/// - `LlmExtract` step: builds extraction prompt, calls `invoke()`
/// - `RigSynthesizer` wrapper: builds synthesis prompt, calls `invoke()`
/// - `LlmDecomposer` wrapper: builds decomposition prompt, calls `invoke()`
///
/// Adding a new provider: implement this trait once. It is then available to
/// ALL callers with zero other changes.
///
/// ## Rust Learning: `Send + Sync + 'static` bounds
///
/// Instances of this trait are stored as `Arc<dyn LlmProvider>` in `AppContext`
/// and shared across async tasks for the lifetime of the process. `Send` allows
/// transfer between threads, `Sync` allows shared references across threads,
/// and `'static` ensures the type owns all its data (no borrowed references
/// that could dangle). These three bounds together are what make
/// `Arc<dyn LlmProvider>` legal in async Rust.
#[async_trait]
pub trait LlmProvider: Send + Sync + 'static {
    /// Send a prompt to the LLM and get a raw text response.
    ///
    /// The caller is responsible for building the prompt (via `PromptBuilder`)
    /// and parsing the response (e.g., JSON into `ExtractionResult`).
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` wrapping all HTTP, authentication,
    /// and API failures from the underlying provider.
    /// Returns `PipelineError::RateLimited` when the API returns HTTP 429.
    async fn invoke(&self, prompt: &str, max_tokens: u32) -> Result<LlmResponse, PipelineError>;

    /// Send a prompt with a separate system prompt, returning a raw text response.
    ///
    /// Providers with native system-prompt support (Anthropic Messages API,
    /// OpenAI-compatible Chat Completions) should override this to use the
    /// native field. The default implementation concatenates
    /// `system + "\n\n" + prompt` and delegates to `invoke()`, which is correct
    /// for any provider that makes no distinction between instruction and
    /// user content.
    ///
    /// ## Rust Learning: Default methods delegating to required methods
    ///
    /// Same pattern as `EmbeddingProvider::embed_batch` — a required core method
    /// (`invoke` / `embed`) supplies behavior for a convenience method whose
    /// default implementation forwards to it. Providers get the default for
    /// free; providers with a native implementation override the method.
    ///
    /// # When to override
    ///
    /// Override when the provider's API treats system content differently from
    /// user content (e.g., weights instructions more heavily, applies different
    /// safety filters, or stores system content in a separate API field).
    /// Failure to override for such providers degrades response quality by
    /// flattening the instruction/content distinction.
    ///
    /// # When the default is correct
    ///
    /// Use the default for providers that only accept a single prompt string
    /// and make no distinction between instruction and content layers.
    ///
    /// # Errors
    ///
    /// Same taxonomy as `invoke()`. Returns `PipelineError::LlmProvider` for
    /// network and authentication failures, `PipelineError::RateLimited` for
    /// HTTP 429.
    async fn invoke_with_system(
        &self,
        system: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<LlmResponse, PipelineError> {
        let combined = format!("{system}\n\n{prompt}");
        self.invoke(&combined, max_tokens).await
    }

    /// Human-readable provider identifier (e.g., `"anthropic"`, `"vllm"`).
    ///
    /// Distinct from `model_name()` so metrics, logs, and `pipeline_events`
    /// records can group by backend regardless of which specific model is in
    /// use. Consumers aggregating cost or performance by provider need this
    /// distinction — two different models from the same backend share a
    /// `provider_name` but not a `model_name`.
    ///
    /// Implementations should return a short, stable, lowercase string.
    /// Conventions in this crate: `"anthropic"`, `"vllm"`.
    ///
    /// ## Rust Learning: Why no default
    ///
    /// Unlike `invoke_with_system`, this method has no default. Every provider
    /// has a well-known backend name; a default like `"unknown"` would silently
    /// hide misconfigured providers and corrupt aggregation queries. Forcing
    /// each implementation to state the name explicitly prevents drift.
    fn provider_name(&self) -> &str;

    /// Human-readable provider+model identifier (e.g., `"claude-sonnet-4-6"`).
    ///
    /// Used for cost-tracking logs and `pipeline_events` records.
    fn model_name(&self) -> &str;

    /// Cost per input token in USD, if known.
    ///
    /// Returns `None` for self-hosted models where cost is not tracked per-token.
    fn cost_per_input_token(&self) -> Option<f64>;

    /// Cost per output token in USD, if known.
    ///
    /// Returns `None` for self-hosted models where cost is not tracked per-token.
    fn cost_per_output_token(&self) -> Option<f64>;

    /// Whether this provider supports structured JSON output natively.
    ///
    /// When `true`, extraction uses typed-prompt paths for better reliability
    /// (e.g., Anthropic tool-use mode). When `false`, extraction uses raw
    /// completion followed by JSON repair (e.g., vLLM).
    fn supports_structured_output(&self) -> bool;
}

/// Response from an LLM `invoke()` call.
///
/// This is the unified return type for all LLM interactions in the pipeline.
/// Callers receive raw text and optional token counts; parsing the text into
/// domain-specific types (e.g., `ExtractionResult`) is the caller's responsibility.
///
/// ## Rust Learning: Eager common-trait derives (Rust API Guidelines C-COMMON-TRAITS)
///
/// - `Debug` — required by C-DEBUG for all public types; enables `tracing::debug!`
///   and `{:?}` formatting in error messages.
/// - `Clone` — callers frequently need to both record token counts (move into
///   `pipeline_events`) and parse text (consume the string), so cloning is expected.
/// - `PartialEq` + `Eq` — enable `assert_eq!` in tests; `Eq` is valid because all
///   fields are `Eq` (String, Option<u32>). Follows C-COMMON-TRAITS.
/// - `Serialize` + `Deserialize` — enable logging the response into `pipeline_events`
///   JSONB payloads and follow C-SERDE for data-structure types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Raw text response from the model.
    pub text: String,

    /// Input tokens consumed, if the provider reports them.
    ///
    /// `None` means the provider did not return usage metadata.
    pub input_tokens: Option<u32>,

    /// Output tokens produced, if the provider reports them.
    ///
    /// `None` means the provider did not return usage metadata.
    pub output_tokens: Option<u32>,
}

/// Embedding provider for converting text to vectors.
///
/// Used by the `Index` step in the pipeline, the `QdrantRetriever` in RAG,
/// and the `EmbeddingReranker` in RAG.
///
/// ## Rust Learning: Default methods on traits
///
/// The `embed_batch` method has a default implementation that calls `embed()`
/// serially. Providers with a native batch API can override it for efficiency,
/// but providers without one get serial iteration for free. This is the same
/// pattern as `std::io::Read` — `read()` is required, `read_to_end()` has a
/// default implementation that calls `read()` in a loop.
///
/// ## Rust Learning: `Send + Sync + 'static` bounds
///
/// Same rationale as `LlmProvider` — instances are stored as
/// `Arc<dyn EmbeddingProvider>` and shared across async tasks.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync + 'static {
    /// Embed a single text string into a vector.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` wrapping HTTP and model errors
    /// from the underlying embedding backend.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PipelineError>;

    /// Embed multiple texts in a batch.
    ///
    /// The default implementation calls `embed()` serially over the slice.
    /// Providers with native batch APIs are encouraged to override this
    /// for better throughput.
    ///
    /// # Errors
    ///
    /// Returns the first `PipelineError` encountered during iteration.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, PipelineError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Dimension of the embedding vectors this provider produces.
    ///
    /// MUST equal the dimensions configured in the Qdrant collection —
    /// mismatches cause runtime query failures.
    fn dimensions(&self) -> u32;

    /// Human-readable model name for logging and cost tracking.
    fn model_name(&self) -> &str;
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
