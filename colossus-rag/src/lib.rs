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
//! | `fastembed` | Enables VectorRetriever with rig-fastembed | Planned (T-R.2) |
//! | `neo4j` | Enables GraphExpander with Neo4j | Planned (T-R.3) |
//! | `axum` | Enables Axum handler integration | Planned |
//! | `full` | Enables all features | Planned |
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
