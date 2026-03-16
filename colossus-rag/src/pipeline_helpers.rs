//! Helper functions for the RAG pipeline.
//!
//! Extracted from `pipeline.rs` to keep the main module under 300 code lines.

use std::collections::HashSet;

use crate::error::RagError;
use crate::traits::VectorRetriever;
use crate::types::{ContextChunk, RetrievalStrategy, ScopeFilter, SubQuery};

// ---------------------------------------------------------------------------
// Helper: convert RetrievalStrategy into search parameters
// ---------------------------------------------------------------------------

/// Convert a [`RetrievalStrategy`] into Qdrant search parameters.
///
/// Returns `(filters, limit)` where:
/// - `filters`: Scope filters to pass to the retriever (may be empty for Broad)
/// - `limit`: Maximum number of results to request from Qdrant
pub(crate) fn strategy_to_search_params(
    strategy: &RetrievalStrategy,
    default_limit: usize,
) -> (Vec<ScopeFilter>, usize) {
    match strategy {
        RetrievalStrategy::Focused { scope } => (scope.clone(), default_limit),
        RetrievalStrategy::Broad { .. } => (Vec::new(), default_limit),
        RetrievalStrategy::Hybrid { scopes, .. } => (scopes.clone(), default_limit),
        RetrievalStrategy::Direct { .. } => {
            tracing::warn!("Direct strategy not yet implemented — falling back to Broad");
            (Vec::new(), default_limit)
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: merge expanded chunks into retriever results
// ---------------------------------------------------------------------------

/// Merge graph-expanded chunks into the retriever's vector search results.
///
/// Expanded chunks are appended after retriever chunks, with deduplication
/// by `node_id`. If an expanded chunk has the same `node_id` as a retriever
/// chunk, it's skipped (the retriever version has a real similarity score,
/// so it's more useful for ranking).
pub(crate) fn merge_expansion(chunks: &mut Vec<ContextChunk>, expanded: Vec<ContextChunk>) {
    let existing_ids: HashSet<String> = chunks.iter().map(|c| c.node_id.clone()).collect();

    for chunk in expanded {
        if !existing_ids.contains(&chunk.node_id) {
            chunks.push(chunk);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: format strategy name for stats
// ---------------------------------------------------------------------------

/// Produce a human-readable name for the strategy (used in PipelineStats).
pub(crate) fn format_strategy_name(strategy: &RetrievalStrategy) -> String {
    match strategy {
        RetrievalStrategy::Focused { scope } => {
            let types: Vec<&str> = scope
                .iter()
                .map(|s| match s.filter_type {
                    crate::types::ScopeFilterType::Document => "document",
                    crate::types::ScopeFilterType::Person => "person",
                    crate::types::ScopeFilterType::NodeType => "node_type",
                    crate::types::ScopeFilterType::Collection => "collection",
                })
                .collect();
            format!("Focused({})", types.join(", "))
        }
        RetrievalStrategy::Broad { .. } => "Broad".into(),
        RetrievalStrategy::Hybrid { scopes, .. } => {
            format!("Hybrid({} scopes)", scopes.len())
        }
        RetrievalStrategy::Direct { .. } => "Direct".into(),
    }
}

// ---------------------------------------------------------------------------
// Helper: execute sub-queries and collect deduplicated chunks
// ---------------------------------------------------------------------------

/// Dispatch sub-queries to the appropriate retrieval backend.
///
/// Vector sub-queries go to the retriever (Qdrant). Graph sub-queries go
/// to the GraphDirectRetriever (Neo4j Cypher). Results are deduplicated
/// by `node_id` — if two sub-queries return the same node, only the first
/// occurrence is kept.
pub(crate) async fn execute_sub_queries(
    sub_queries: &[SubQuery],
    retriever: &dyn VectorRetriever,
    limit: usize,
    filters: &[ScopeFilter],
    #[cfg(feature = "neo4j")]
    graph_retriever: &Option<crate::graph_retriever::GraphDirectRetriever>,
) -> Result<Vec<ContextChunk>, RagError> {
    let mut chunks: Vec<ContextChunk> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    for sub_query in sub_queries {
        let results = match sub_query {
            SubQuery::VectorSearch { query } => retriever.search(query, limit, filters).await?,

            #[cfg(feature = "neo4j")]
            SubQuery::GraphDocumentContent { document_id, .. } => {
                if let Some(ref gr) = graph_retriever {
                    gr.fetch_document_evidence(document_id).await?
                } else {
                    Vec::new()
                }
            }

            #[cfg(feature = "neo4j")]
            SubQuery::GraphPersonStatements { person_id, .. } => {
                if let Some(ref gr) = graph_retriever {
                    gr.fetch_person_statements(person_id).await?
                } else {
                    Vec::new()
                }
            }

            #[cfg(feature = "neo4j")]
            SubQuery::GraphContradictions { person_name, .. } => {
                if let Some(ref gr) = graph_retriever {
                    gr.fetch_contradictions(person_name).await?
                } else {
                    Vec::new()
                }
            }

            // If neo4j feature not enabled but graph sub-queries present, skip.
            #[cfg(not(feature = "neo4j"))]
            _ => {
                tracing::warn!("Graph sub-query skipped — neo4j feature not enabled");
                Vec::new()
            }
        };

        for chunk in results {
            if seen_ids.insert(chunk.node_id.clone()) {
                chunks.push(chunk);
            }
        }
    }

    Ok(chunks)
}
