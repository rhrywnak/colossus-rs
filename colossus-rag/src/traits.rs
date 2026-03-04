//! Trait definitions for the Colossus RAG pipeline.
//!
//! These traits define the contract for each stage of the pipeline:
//!
//! ```text
//! QueryRouter ──► VectorRetriever ──► GraphExpander ──► ContextAssembler ──► Synthesizer
//! ```
//!
//! Each trait has exactly one primary method. Implementations are provided
//! by separate modules (behind feature flags) or by the no-op defaults
//! in [`crate`].
//!
//! ## Rust Learning: `#[async_trait]`
//!
//! Rust traits cannot natively have `async fn` methods that work with
//! dynamic dispatch (`dyn Trait`). This is because `async fn` returns an
//! opaque `impl Future` type, and trait objects need to know the concrete
//! return type at compile time to build a vtable.
//!
//! The `#[async_trait]` macro solves this by rewriting:
//! ```rust,ignore
//! #[async_trait]
//! trait Foo {
//!     async fn bar(&self) -> i32;
//! }
//! ```
//! into:
//! ```rust,ignore
//! trait Foo {
//!     fn bar(&self) -> Pin<Box<dyn Future<Output = i32> + Send + '_>>;
//! }
//! ```
//!
//! This adds a small heap allocation per call (the `Box`), but enables
//! `dyn Trait` usage which we need for runtime flexibility — e.g., swapping
//! between a real GraphExpander and a NoOpExpander without recompiling.
//!
//! ## Rust Learning: `Send + Sync` bounds
//!
//! `#[async_trait]` adds `Send` bounds to the returned future by default.
//! This means all trait implementations must be safe to send across threads.
//! This is required because Axum handlers run on a multi-threaded tokio
//! runtime where tasks can migrate between threads.
//!
//! If an implementation holds a non-Send type, you'd use `#[async_trait(?Send)]`
//! instead — but we don't need that here.

use async_trait::async_trait;

use crate::error::RagError;
use crate::types::{
    AssembledContext, ContextChunk, RetrievalStrategy, ScopeFilter, SynthesisResult,
};

/// Analyzes a question and determines the best retrieval strategy.
///
/// The router is the first stage of the pipeline. It examines the user's
/// question and decides HOW to search for relevant context.
///
/// ## Future implementations
/// - `LlmRouter`: Uses a fast LLM call to classify the question
/// - `RuleRouter`: Uses keyword/regex patterns (cheaper, no API call)
/// - `NoOpRouter`: Always returns `Broad` (in [`crate`])
#[async_trait]
pub trait QueryRouter: Send + Sync {
    async fn route(&self, question: &str) -> Result<RetrievalStrategy, RagError>;
}

/// Searches the vector store for chunks relevant to a query.
///
/// The retriever embeds the query text, searches Qdrant, and returns
/// matching chunks with their similarity scores.
///
/// ## Future implementations
/// - `RigRetriever`: Uses rig-core + rig-qdrant + rig-fastembed (T-R.2)
///
/// ## Parameters
/// - `query`: The text to embed and search for
/// - `limit`: Maximum number of results to return
/// - `filters`: Scope filters to narrow the search (from the router's strategy)
#[async_trait]
pub trait VectorRetriever: Send + Sync {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        filters: &[ScopeFilter],
    ) -> Result<Vec<ContextChunk>, RagError>;
}

/// Expands context by traversing the knowledge graph from seed nodes.
///
/// Starting from nodes found by the vector retriever, the expander
/// follows relationships in Neo4j to find additional relevant context
/// that wasn't directly matched by vector similarity.
///
/// For example, if vector search finds "Phillips: Emil wanted $50K returned",
/// the expander might follow a `SUPPORTS` relationship to find the related
/// `Harm` node or `MotionClaim` that this evidence supports.
///
/// ## Future implementations
/// - `Neo4jExpander`: Traverses relationships in Neo4j (T-R.3)
/// - `NoOpExpander`: Returns empty vec (in [`crate`])
///
/// ## Parameters
/// - `seed_ids`: Node IDs from vector search to start expanding from
/// - `max_depth`: How many relationship hops to follow (1 = direct neighbors)
#[async_trait]
pub trait GraphExpander: Send + Sync {
    async fn expand(
        &self,
        seed_ids: &[String],
        max_depth: u32,
    ) -> Result<Vec<ContextChunk>, RagError>;
}

/// Formats context chunks into a text prompt for the LLM.
///
/// This is the only SYNCHRONOUS trait in the pipeline — it does pure
/// string formatting with no I/O. It takes the raw chunks and produces
/// a structured text block with a system prompt, formatted evidence,
/// and a token count estimate.
///
/// ## Rust Learning: Why not `async`?
///
/// Not every operation needs to be async. This trait only does:
/// 1. String concatenation and formatting
/// 2. Token count estimation (a simple heuristic: chars / 4)
///
/// Making it async would add unnecessary overhead — the `Box<dyn Future>`
/// allocation, the `Send` bounds, and the async runtime overhead — for
/// work that completes in microseconds with no I/O.
///
/// In Java terms: this is like a pure function with no database calls,
/// no HTTP requests, no file I/O. It runs entirely in-memory.
///
/// ## Parameters
/// - `question`: The user's original question (included in the prompt)
/// - `chunks`: The context chunks to format
/// - `max_tokens`: Token budget for the context (to avoid exceeding model limits)
pub trait ContextAssembler: Send + Sync {
    fn assemble(
        &self,
        question: &str,
        chunks: &[ContextChunk],
        max_tokens: usize,
    ) -> AssembledContext;
}

/// Sends the assembled context to an LLM and returns the synthesized answer.
///
/// The synthesizer is the final async stage. It calls Claude (or another LLM)
/// with the formatted context and returns the generated answer along with
/// citations and token usage statistics.
///
/// ## Future implementations
/// - `RigSynthesizer`: Uses rig-core's Anthropic provider (T-R.2)
/// - `DirectSynthesizer`: Uses raw reqwest HTTP calls (fallback)
#[async_trait]
pub trait Synthesizer: Send + Sync {
    async fn synthesize(
        &self,
        context: &AssembledContext,
        question: &str,
    ) -> Result<SynthesisResult, RagError>;
}
