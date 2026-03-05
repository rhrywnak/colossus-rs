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
//! | `qdrant` | Qdrant vector search via gRPC | T-R.2.1 |
//! | `fastembed` | Local embeddings via rig-fastembed | T-R.2.1 |
//! | `neo4j` | Enables GraphExpander with Neo4j | Planned (T-R.3) |
//! | `axum` | Enables Axum handler integration | Planned |
//! | `full` | Enables all features | Available |
//!
//! Enable both `qdrant` and `fastembed` (or just `full`) to get `QdrantRetriever`.
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

// ## Rust Learning: `#[cfg(all(...))]` for multi-feature gates
//
// The retriever module requires BOTH the `qdrant` and `fastembed` features
// because it uses qdrant-client for search AND rig-fastembed for embedding.
// `#[cfg(all(feature = "a", feature = "b"))]` means "compile this only
// when both features are enabled."
#[cfg(all(feature = "qdrant", feature = "fastembed"))]
mod retriever;

// The assembler and synthesizer modules use only base dependencies (no feature flags).
// They're always available regardless of which features are enabled.
mod assembler;
mod synthesizer;

// ## Rust Learning: `compile_error!` for helpful diagnostics
//
// If someone enables only ONE of the two required features, they'd get
// confusing "type not found" errors. Instead, we detect the mismatch at
// compile time and produce a clear error message explaining what's needed.
//
// `any(...)` matches if EITHER feature is on.
// `not(all(...))` matches if BOTH are NOT on simultaneously.
// Together: "one is on but not both" → helpful error.
#[cfg(any(
    all(feature = "qdrant", not(feature = "fastembed")),
    all(feature = "fastembed", not(feature = "qdrant")),
))]
compile_error!(
    "QdrantRetriever requires both `qdrant` and `fastembed` features. \
     Enable both: features = [\"qdrant\", \"fastembed\"] or features = [\"full\"]"
);

// --- Public API re-exports: Error ---

pub use error::RagError;

// --- Public API re-exports: Core types ---

pub use types::{
    AssembledContext, Citation, ContextChunk, PipelineStats, RagResult, RelatedNode,
    RelationDirection, RetrievalStrategy, ScopeFilter, ScopeFilterType, SourceReference,
    SynthesisResult,
};

// --- Public API re-exports: Traits ---

pub use traits::{ContextAssembler, GraphExpander, QueryRouter, Synthesizer, VectorRetriever};

// --- Public API re-exports: No-op implementations ---

pub use noop::{NoOpExpander, NoOpRouter};

// --- Public API re-exports: Feature-gated implementations ---

#[cfg(all(feature = "qdrant", feature = "fastembed"))]
pub use retriever::{scope_filters_to_qdrant_filter, QdrantRetriever};

pub use assembler::{estimate_tokens, format_chunk, LegalAssembler};
pub use synthesizer::RigSynthesizer;
