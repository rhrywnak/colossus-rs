//! Expansion categories — which relationships to follow per strategy.
//!
//! This module maps the router's `RetrievalStrategy` to an `ExpansionCategory`
//! that controls which Neo4j relationship types the expander follows.
//!
//! ## Rust Learning: Strategy Pattern via enum mapping
//!
//! The router produces a `RetrievalStrategy` (Focused, Broad, etc.).
//! Rather than passing the full strategy into every Cypher query function,
//! we map it to an `ExpansionCategory` that directly controls which
//! OPTIONAL MATCH clauses are included. This is the Strategy Pattern:
//! one enum value changes the expansion behavior throughout the pipeline.
//!
//! ## Why not modify the Cypher at runtime?
//!
//! We could build dynamic Cypher strings, but that's fragile and hard to
//! test. Instead, each `expand_*` function gets a `HashSet<&str>` of
//! allowed relationship types and simply skips OPTIONAL MATCHes for
//! relationship types not in the set. The Cypher stays static and testable.

use std::collections::HashSet;

use crate::types::{RetrievalStrategy, ScopeFilterType};

// ---------------------------------------------------------------------------
// ExpansionCategory enum
// ---------------------------------------------------------------------------

/// Determines which Neo4j relationship types the expander follows.
///
/// Derived from the router's `RetrievalStrategy` via `from_strategy()`.
/// Each category maps to a specific set of allowed relationship types,
/// or `None` for Broad mode (allow everything).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExpansionCategory {
    /// Follow CONTRADICTS, REBUTS, STATED_BY, CONTAINED_IN, PROVES.
    /// For questions comparing statements or finding conflicts.
    Contradiction,

    /// Follow STATED_BY, ABOUT, REPRESENTED_BY, EMPLOYED_BY, AUTHORED.
    /// For questions about a specific person's involvement.
    Person,

    /// Follow CONTAINED_IN, APPEARS_IN, DOCUMENTED_IN, EXHIBIT_OF, IN_CASE.
    /// For questions about specific documents.
    Document,

    /// Follow PROVES, RELIES_ON, SUPPORTS, CHARACTERIZES, EVIDENCED_BY,
    /// CAUSED_BY, DAMAGES_FOR.
    /// For questions tracing the legal proof chain.
    ProofChain,

    /// Follow ALL relationship types (current behavior).
    /// Fallback for uncategorized questions.
    Broad,
}

impl ExpansionCategory {
    /// Derive the expansion category from a RetrievalStrategy.
    ///
    /// This is the mapping between router output and expander behavior:
    /// - Focused(person) -> Person
    /// - Focused(document) -> Document
    /// - Focused(node_type) -> ProofChain
    /// - Hybrid (comparison) -> Contradiction
    /// - Broad -> Broad
    /// - Direct -> Broad
    pub(crate) fn from_strategy(strategy: &RetrievalStrategy) -> Self {
        match strategy {
            RetrievalStrategy::Focused { scope } => {
                // Check what kind of scope filters are present.
                // Priority: if ANY person filter exists -> Person category.
                // If only document filters -> Document category.
                // If only node_type -> ProofChain.
                let has_person = scope
                    .iter()
                    .any(|s| matches!(s.filter_type, ScopeFilterType::Person));
                let has_document = scope
                    .iter()
                    .any(|s| matches!(s.filter_type, ScopeFilterType::Document));

                if has_person {
                    ExpansionCategory::Person
                } else if has_document {
                    ExpansionCategory::Document
                } else {
                    ExpansionCategory::ProofChain
                }
            }
            RetrievalStrategy::Hybrid { .. } => ExpansionCategory::Contradiction,
            RetrievalStrategy::Broad { .. } => ExpansionCategory::Broad,
            RetrievalStrategy::Direct { .. } => ExpansionCategory::Broad,
        }
    }

    /// Return the set of relationship types this category allows.
    ///
    /// When the set is `None`, ALL relationship types are allowed (Broad mode).
    /// When `Some(set)`, only relationships in the set are followed.
    pub(crate) fn allowed_relationships(&self) -> Option<HashSet<&'static str>> {
        match self {
            ExpansionCategory::Contradiction => Some(HashSet::from([
                "CONTRADICTS",
                "REBUTS",
                "STATED_BY",
                "CONTAINED_IN",
                "PROVES",
            ])),
            ExpansionCategory::Person => Some(HashSet::from([
                "STATED_BY",
                "ABOUT",
                "REPRESENTED_BY",
                "EMPLOYED_BY",
                "AUTHORED",
            ])),
            ExpansionCategory::Document => Some(HashSet::from([
                "CONTAINED_IN",
                "APPEARS_IN",
                "DOCUMENTED_IN",
                "EXHIBIT_OF",
                "IN_CASE",
            ])),
            ExpansionCategory::ProofChain => Some(HashSet::from([
                "PROVES",
                "RELIES_ON",
                "SUPPORTS",
                "CHARACTERIZES",
                "EVIDENCED_BY",
                "CAUSED_BY",
                "DAMAGES_FOR",
            ])),
            ExpansionCategory::Broad => None, // None = allow everything
        }
    }
}
