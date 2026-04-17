//! # colossus-rag
//!
//! RAG (Retrieval-Augmented Generation) pipeline library for Colossus applications.
//!
//! This crate provides the types, traits, and implementations for building
//! RAG pipelines that combine vector search (Qdrant), knowledge graph
//! traversal (Neo4j), and LLM synthesis (Claude via Rig).
//!
//! ## Pipeline Architecture
//!
//! ```text
//! User Question
//!       │
//!       ▼
//! ┌─────────────┐
//! │ QueryRouter  │  Analyze question → choose retrieval strategy
//! └──────┬──────┘
//!        ▼
//! ┌──────────────────┐
//! │ VectorRetriever   │  Embed query → search Qdrant → return chunks
//! └──────┬───────────┘
//!        ▼
//! ┌──────────────────┐
//! │ GraphExpander     │  Follow Neo4j relationships → find related context
//! └──────┬───────────┘
//!        ▼
//! ┌──────────────────┐
//! │ ContextAssembler  │  Format chunks into a prompt (synchronous)
//! └──────┬───────────┘
//!        ▼
//! ┌──────────────────┐
//! │ Synthesizer       │  Send to Claude → get answer + citations
//! └──────┬───────────┘
//!        ▼
//!    RagResult
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use colossus_rag::{NoOpRouter, NoOpExpander, RagError};
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description | Status |
//! |---------|-------------|--------|
//! | `qdrant` | Qdrant vector search via gRPC; embedding backend injected at runtime via `Arc<dyn EmbeddingProvider>` | T-R.2.1 |
//! | `neo4j` | Enables GraphExpander with Neo4j | T-R.3.1 |
//! | `axum` | Enables Axum handler integration | Planned |
//! | `full` | Enables all features | Available |
//!
//! Enable `qdrant` (or `full`) to get `QdrantRetriever`.
//!
//! ## Rust Learning: Crate organization
//!
//! A Rust library crate (`lib.rs`) serves two purposes:
//!
//! 1. **Module tree root**: `mod types;` declares that `src/types.rs` exists
//!    and is part of this crate. Without this declaration, the file is ignored.
//!
//! 2. **Public API surface**: `pub use types::ContextChunk;` re-exports the
//!    type so consumers can write `use colossus_rag::ContextChunk;` instead
//!    of `use colossus_rag::types::ContextChunk;`.
//!
//! This pattern (declare modules, re-export the public items) is exactly
//! what colossus-auth does, and is idiomatic for Rust libraries.

mod error;
mod noop;
mod traits;
mod types;

// --- Feature-gated modules ---

// Retriever requires qdrant-client for vector search. The embedding provider
// is injected at runtime via Arc<dyn EmbeddingProvider>, so this module has
// no compile-time dependency on any specific embedding backend.
#[cfg(feature = "qdrant")]
mod retriever;

// Reranker uses the EmbeddingProvider trait + cosine similarity math.
// No external crate dependency — always available.
mod reranker;

// The Neo4j expander module requires the `neo4j` feature.
// Split into two files to stay under the 300-line code limit:
// - expander.rs: struct, trait impl, helpers, conversion
// - expander_queries.rs: 7 per-type Cypher expansion functions
#[cfg(feature = "neo4j")]
mod expander;
#[cfg(feature = "neo4j")]
mod expander_queries;
#[cfg(feature = "neo4j")]
mod expander_queries_minor;
#[cfg(feature = "neo4j")]
mod expansion_category;
#[cfg(feature = "neo4j")]
mod graph_retriever;

// The router, assembler, and synthesizer modules use only base dependencies (no feature flags).
// They're always available regardless of which features are enabled.
mod assembler;
mod decomposer;
mod pipeline;
mod pipeline_helpers;
mod router;
mod synthesizer;

// --- Public API re-exports: Error ---

pub use error::RagError;

// --- Public API re-exports: Core types ---

pub use types::{
    AssembledContext, Citation, ContextChunk, DecompositionResult, PipelineStats, RagResult,
    RelatedNode, RelationDirection, RetrievalStrategy, ScopeFilter, ScopeFilterType,
    SourceReference, SubQuery, SynthesisResult,
};

// --- Public API re-exports: Traits ---

pub use traits::{
    ContextAssembler, GraphExpander, QueryDecomposer, QueryRouter, Synthesizer, VectorRetriever,
};

// --- Public API re-exports: No-op implementations ---

pub use noop::{NoOpDecomposer, NoOpExpander, NoOpRouter};

// --- Public API re-exports: Feature-gated implementations ---

#[cfg(feature = "qdrant")]
pub use retriever::{scope_filters_to_qdrant_filter, QdrantRetriever};

#[cfg(feature = "neo4j")]
pub use expander::Neo4jExpander;

#[cfg(feature = "neo4j")]
pub use graph_retriever::GraphDirectRetriever;

pub use reranker::EmbeddingReranker;

pub use assembler::{estimate_tokens, format_chunk, LegalAssembler};
pub use decomposer::LlmDecomposer;
pub use pipeline::{RagPipeline, RagPipelineBuilder};
pub use router::RuleBasedRouter;
pub use synthesizer::RigSynthesizer;
