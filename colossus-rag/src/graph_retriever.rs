//! GraphDirectRetriever — executes targeted Cypher queries for decomposed sub-queries.
//!
//! Unlike the GraphExpander (which follows relationships from seed nodes),
//! this module runs standalone Cypher queries that directly fetch nodes
//! matching a specific criterion (e.g., "all evidence from document X").
//!
//! Used by the pipeline to fulfill graph-typed sub-queries from the
//! QueryDecomposer.

use std::sync::Arc;

use neo4rs::{query, Graph};

use crate::error::RagError;
use crate::types::{ContextChunk, SourceReference};

// ---------------------------------------------------------------------------
// GraphDirectRetriever struct
// ---------------------------------------------------------------------------

/// Executes direct Cypher queries against Neo4j for decomposed sub-queries.
///
/// ## Rust Learning: Sharing the Neo4j connection
///
/// This struct holds the same `Arc<Graph>` as the `Neo4jExpander`. When
/// the pipeline is built, it clones the Arc and passes it to both the
/// expander and the graph retriever. No extra connections are created.
pub struct GraphDirectRetriever {
    graph: Arc<Graph>,
}

impl GraphDirectRetriever {
    /// Create a new graph retriever with a shared Neo4j connection.
    pub fn new(graph: Arc<Graph>) -> Self {
        Self { graph }
    }
}

// ---------------------------------------------------------------------------
// Query methods — one per graph SubQuery type
// ---------------------------------------------------------------------------

impl GraphDirectRetriever {
    /// Fetch all Evidence nodes contained in a specific document.
    ///
    /// Cypher: MATCH (e:Evidence)-[:CONTAINED_IN]->(d:Document {id: $doc_id})
    /// Returns evidence with title, verbatim_quote, page_number, and the
    /// document title for source reference.
    ///
    /// Limit: 20 nodes max (same as expand_document in expander_queries).
    pub async fn fetch_document_evidence(
        &self,
        document_id: &str,
    ) -> Result<Vec<ContextChunk>, RagError> {
        let cypher = "MATCH (e:Evidence)-[:CONTAINED_IN]->(d:Document {id: $doc_id})
            RETURN e.id AS id, e.title AS title,
                   e.verbatim_quote AS quote, e.significance AS significance,
                   e.page_number AS page, e.statement_date AS date,
                   d.title AS doc_title
            ORDER BY e.page_number
            LIMIT 20";

        let mut result = self
            .graph
            .execute(query(cypher).param("doc_id", document_id))
            .await
            .map_err(|e| RagError::SearchError(format!("Graph document query failed: {e}")))?;

        let mut chunks = Vec::new();

        while let Some(row) = result
            .next()
            .await
            .map_err(|e| RagError::SearchError(format!("Graph row error: {e}")))?
        {
            let id: String = row.get("id").unwrap_or_default();
            if id.is_empty() {
                continue;
            }

            let title: String = row.get("title").unwrap_or_default();
            let quote: String = row.get("quote").unwrap_or_default();
            let significance: String = row.get("significance").unwrap_or_default();
            let doc_title: String = row.get("doc_title").unwrap_or_default();
            let page: Option<u32> = row.get::<i64>("page").ok().map(|p| p as u32);

            let content = if !quote.is_empty() {
                quote
            } else if !significance.is_empty() {
                significance
            } else {
                title.clone()
            };

            chunks.push(ContextChunk {
                node_id: id,
                node_type: "Evidence".to_string(),
                title,
                content,
                score: 0.0, // Graph-retrieved, not vector-scored
                source: SourceReference {
                    document_title: Some(doc_title),
                    page_number: page,
                    ..Default::default()
                },
                relationships: Vec::new(),
                metadata: serde_json::Value::Null,
            });
        }

        tracing::info!(
            document_id,
            results = chunks.len(),
            "GraphDirectRetriever: fetched document evidence"
        );

        Ok(chunks)
    }

    /// Fetch all Evidence nodes stated by a specific person.
    ///
    /// Uses person ID (e.g., "george-phillips") or falls back to name match.
    /// Limit: 20 nodes max.
    pub async fn fetch_person_statements(
        &self,
        person_id: &str,
    ) -> Result<Vec<ContextChunk>, RagError> {
        let cypher = "MATCH (e:Evidence)-[:STATED_BY]->(p)
            WHERE p.id = $person_id OR toLower(p.name) CONTAINS toLower($person_id)
            OPTIONAL MATCH (e)-[:CONTAINED_IN]->(d:Document)
            RETURN e.id AS id, e.title AS title,
                   e.verbatim_quote AS quote, e.significance AS significance,
                   e.page_number AS page,
                   d.title AS doc_title
            ORDER BY e.statement_date
            LIMIT 20";

        let mut result = self
            .graph
            .execute(query(cypher).param("person_id", person_id))
            .await
            .map_err(|e| RagError::SearchError(format!("Graph person query failed: {e}")))?;

        let mut chunks = Vec::new();

        while let Some(row) = result
            .next()
            .await
            .map_err(|e| RagError::SearchError(format!("Graph row error: {e}")))?
        {
            let id: String = row.get("id").unwrap_or_default();
            if id.is_empty() {
                continue;
            }

            let title: String = row.get("title").unwrap_or_default();
            let quote: String = row.get("quote").unwrap_or_default();
            let significance: String = row.get("significance").unwrap_or_default();
            let doc_title: String = row.get("doc_title").unwrap_or_default();
            let page: Option<u32> = row.get::<i64>("page").ok().map(|p| p as u32);

            let content = if !quote.is_empty() {
                quote
            } else if !significance.is_empty() {
                significance
            } else {
                title.clone()
            };

            chunks.push(ContextChunk {
                node_id: id,
                node_type: "Evidence".to_string(),
                title,
                content,
                score: 0.0,
                source: SourceReference {
                    document_title: if doc_title.is_empty() {
                        None
                    } else {
                        Some(doc_title)
                    },
                    page_number: page,
                    ..Default::default()
                },
                relationships: Vec::new(),
                metadata: serde_json::Value::Null,
            });
        }

        tracing::info!(
            person_id,
            results = chunks.len(),
            "GraphDirectRetriever: fetched person statements"
        );

        Ok(chunks)
    }

    /// Fetch Evidence nodes involved in contradictions for a person.
    ///
    /// Finds evidence stated by the person that has CONTRADICTS relationships,
    /// and returns both sides of each contradiction.
    /// Limit: 20 nodes max.
    pub async fn fetch_contradictions(
        &self,
        person_name: &str,
    ) -> Result<Vec<ContextChunk>, RagError> {
        let cypher = "MATCH (e:Evidence)-[:STATED_BY]->(p)
            WHERE toLower(p.name) CONTAINS toLower($person_name)
            MATCH (e)-[:CONTRADICTS]-(other:Evidence)
            OPTIONAL MATCH (e)-[:CONTAINED_IN]->(d:Document)
            OPTIONAL MATCH (other)-[:CONTAINED_IN]->(d2:Document)
            RETURN e.id AS id, e.title AS title, e.verbatim_quote AS quote,
                   d.title AS doc_title, e.page_number AS page,
                   other.id AS other_id, other.title AS other_title,
                   other.verbatim_quote AS other_quote,
                   d2.title AS other_doc_title, other.page_number AS other_page
            LIMIT 20";

        let mut result = self
            .graph
            .execute(query(cypher).param("person_name", person_name))
            .await
            .map_err(|e| {
                RagError::SearchError(format!("Graph contradictions query failed: {e}"))
            })?;

        let mut chunks = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        while let Some(row) = result
            .next()
            .await
            .map_err(|e| RagError::SearchError(format!("Graph row error: {e}")))?
        {
            // Extract the main evidence node.
            let id: String = row.get("id").unwrap_or_default();
            if !id.is_empty() && seen_ids.insert(id.clone()) {
                let title: String = row.get("title").unwrap_or_default();
                let quote: String = row.get("quote").unwrap_or_default();
                let doc_title: String = row.get("doc_title").unwrap_or_default();
                let page: Option<u32> = row.get::<i64>("page").ok().map(|p| p as u32);

                let content = if !quote.is_empty() {
                    quote
                } else {
                    title.clone()
                };

                chunks.push(ContextChunk {
                    node_id: id,
                    node_type: "Evidence".to_string(),
                    title,
                    content,
                    score: 0.0,
                    source: SourceReference {
                        document_title: if doc_title.is_empty() {
                            None
                        } else {
                            Some(doc_title)
                        },
                        page_number: page,
                        ..Default::default()
                    },
                    relationships: Vec::new(),
                    metadata: serde_json::Value::Null,
                });
            }

            // Extract the contradicting evidence node.
            let other_id: String = row.get("other_id").unwrap_or_default();
            if !other_id.is_empty() && seen_ids.insert(other_id.clone()) {
                let other_title: String = row.get("other_title").unwrap_or_default();
                let other_quote: String = row.get("other_quote").unwrap_or_default();
                let other_doc: String = row.get("other_doc_title").unwrap_or_default();
                let other_page: Option<u32> =
                    row.get::<i64>("other_page").ok().map(|p| p as u32);

                let content = if !other_quote.is_empty() {
                    other_quote
                } else {
                    other_title.clone()
                };

                chunks.push(ContextChunk {
                    node_id: other_id,
                    node_type: "Evidence".to_string(),
                    title: other_title,
                    content,
                    score: 0.0,
                    source: SourceReference {
                        document_title: if other_doc.is_empty() {
                            None
                        } else {
                            Some(other_doc)
                        },
                        page_number: other_page,
                        ..Default::default()
                    },
                    relationships: Vec::new(),
                    metadata: serde_json::Value::Null,
                });
            }
        }

        tracing::info!(
            person_name,
            results = chunks.len(),
            "GraphDirectRetriever: fetched contradictions"
        );

        Ok(chunks)
    }
}
