//! Error types for the Colossus RAG pipeline.
//!
//! ## Rust Learning: `thiserror` crate
//!
//! The `thiserror` crate provides derive macros that auto-implement the
//! `std::error::Error` and `std::fmt::Display` traits for your enum.
//!
//! Each `#[error("...")]` attribute defines the Display message for that variant.
//! `{0}` refers to the first (and only) unnamed field in the variant.
//!
//! This is much less boilerplate than implementing Error + Display manually,
//! which would require ~30 lines per variant in Java-style code.
//!
//! ## Rust Learning: Error enum vs exception hierarchy
//!
//! In Java/C++, you'd have an exception class hierarchy (RagException → subclasses).
//! In Rust, we use a flat enum where each variant carries its own context string.
//! The `?` operator converts lower-level errors into the appropriate variant.
//!
//! Pattern:
//! ```rust,ignore
//! fn search(&self) -> Result<Vec<Chunk>, RagError> {
//!     let results = qdrant.search(query)
//!         .map_err(|e| RagError::SearchError(e.to_string()))?;
//!     Ok(results)
//! }
//! ```

/// Errors that can occur during RAG pipeline execution.
///
/// Each variant corresponds to a stage of the pipeline. The `String` payload
/// carries a human-readable description of what went wrong, including any
/// underlying error messages from external services.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RagError {
    /// The input question was empty, too long, or otherwise invalid.
    ///
    /// Example: `RagError::InvalidInput("Question must not be empty".into())`
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Failed to generate embeddings for the query text.
    ///
    /// Typically caused by fastembed/ONNX runtime errors or model download
    /// failures. Example: `RagError::EmbeddingError("ONNX model not found".into())`
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Vector search against Qdrant failed.
    ///
    /// Could be a connection error, collection not found, or query timeout.
    /// Example: `RagError::SearchError("Qdrant connection refused on port 6334".into())`
    #[error("Search error: {0}")]
    SearchError(String),

    /// Knowledge graph expansion failed.
    ///
    /// Typically a Neo4j connection or Cypher query error.
    /// Example: `RagError::ExpandError("Neo4j timeout after 5000ms".into())`
    #[error("Expand error: {0}")]
    ExpandError(String),

    /// Context assembly failed.
    ///
    /// Rare — would happen if template rendering fails or token estimation
    /// encounters an unexpected error.
    /// Example: `RagError::AssemblyError("Context exceeds 200k token limit".into())`
    #[error("Assembly error: {0}")]
    AssemblyError(String),

    /// LLM synthesis (Claude API call) failed.
    ///
    /// Could be an API key issue, rate limit, model not found, or network error.
    /// Example: `RagError::SynthesisError("Anthropic API 429: rate limited".into())`
    #[error("Synthesis error: {0}")]
    SynthesisError(String),

    /// Configuration error — missing env vars, invalid URLs, etc.
    ///
    /// Example: `RagError::ConfigError("ANTHROPIC_API_KEY not set".into())`
    #[error("Config error: {0}")]
    ConfigError(String),
}
