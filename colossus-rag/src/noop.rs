//! No-op implementations of pipeline traits.
//!
//! These are default implementations that do nothing useful but satisfy
//! the trait contracts. They're used when a pipeline stage is not needed:
//!
//! - `NoOpRouter`: Always returns `Broad` strategy — useful when you want
//!   to skip routing and always do a broad search.
//! - `NoOpExpander`: Returns an empty vec — useful when you don't have
//!   a knowledge graph (no Neo4j) and only want vector search.
//!
//! ## Rust Learning: Why provide no-op implementations?
//!
//! In Java, you might make these optional with `null` checks. In Rust,
//! we avoid `Option<Box<dyn Trait>>` checks scattered throughout the
//! pipeline. Instead, we provide no-op implementations that the pipeline
//! can call unconditionally — the no-op just returns a harmless default.
//!
//! This follows the Null Object Pattern: instead of checking "is there
//! an expander?", we always have an expander — it just might do nothing.

use async_trait::async_trait;

use crate::error::RagError;
use crate::traits::{GraphExpander, QueryRouter};
use crate::types::{ContextChunk, RetrievalStrategy};

/// A router that always returns [`RetrievalStrategy::Broad`] with no filters.
///
/// Use this when you don't need intelligent routing and want every query
/// to search broadly across the entire knowledge base.
///
/// ## Rust Learning: Unit struct
///
/// `NoOpRouter` has no fields — it's a "unit struct" (like `struct Foo;`).
/// It takes zero bytes of memory. You create it with just `NoOpRouter`,
/// no parentheses or braces needed. Despite having no state, it can still
/// implement traits and have methods.
pub struct NoOpRouter;

#[async_trait]
impl QueryRouter for NoOpRouter {
    async fn route(&self, _question: &str) -> Result<RetrievalStrategy, RagError> {
        // Always broad, no type filtering — search everything.
        Ok(RetrievalStrategy::Broad { node_types: None })
    }
}

/// A graph expander that returns no additional context.
///
/// Use this when you don't have a knowledge graph and want to skip
/// the expansion stage entirely. The pipeline will use only the
/// vector search results.
pub struct NoOpExpander;

#[async_trait]
impl GraphExpander for NoOpExpander {
    async fn expand(
        &self,
        _seed_ids: &[String],
        _max_depth: u32,
    ) -> Result<Vec<ContextChunk>, RagError> {
        // No graph to expand — return empty results.
        Ok(Vec::new())
    }
}
