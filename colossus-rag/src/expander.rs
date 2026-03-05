//! Neo4jExpander — implements [`GraphExpander`] using Neo4j graph traversal.
//!
//! This module provides graph-based context expansion for the RAG pipeline.
//! Starting from seed node IDs (found by vector search), it traverses Neo4j
//! relationships to find additional relevant context that wasn't directly
//! matched by vector similarity.
//!
//! This is Colossus's unique value-add — no other Rust RAG library provides
//! graph-based context expansion. The combination of vector search (Qdrant)
//! + graph expansion (Neo4j) gives much richer context than vector-only RAG.
//!
//! ## Migration from colossus-legal
//!
//! This code migrates the graph expansion logic from:
//! - `colossus-legal/backend/src/services/graph_expander.rs`
//! - `colossus-legal/backend/src/services/graph_expansion_queries.rs`
//! - `colossus-legal/backend/src/services/graph_expansion_minor.rs`
//!
//! The Cypher queries and traversal logic are IDENTICAL to the original.
//! The only changes are:
//! 1. Internal types → `ContextChunk` + `RelatedNode` (our shared types)
//! 2. `GraphExpanderError` → `RagError` (our shared error type)
//! 3. The `GraphExpander` trait signature (takes `&[String]` seed IDs,
//!    resolves node types internally via a batch Neo4j query)
//!
//! ## Architecture: Two-file split
//!
//! The expander is split into two files to stay under the 300-line code limit:
//! - `expander.rs` (this file): struct, trait impl, helpers, conversion
//! - `expander_queries.rs`: 7 per-type Cypher expansion functions
//!
//! ## Rust Learning: `Arc<neo4rs::Graph>`
//!
//! `neo4rs::Graph` is the Neo4j connection pool. It's `Clone` (internally
//! uses `Arc`), but we wrap it in `Arc` explicitly to make the sharing
//! semantics clear — multiple pipeline stages can share the same connection.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use neo4rs::{query, Graph};

use crate::error::RagError;
use crate::expander_queries;
use crate::traits::GraphExpander;
use crate::types::{ContextChunk, RelatedNode, RelationDirection, SourceReference};

// ---------------------------------------------------------------------------
// Internal types — mirror the original graph_expander.rs types
// ---------------------------------------------------------------------------

// ## Rust Learning: `pub(crate)` visibility
//
// These types are used across two modules within this crate (expander.rs
// and expander_queries.rs) but should NOT be exposed to consumers.
// `pub(crate)` means "public within this crate only."

/// A node from the Neo4j graph with its properties.
///
/// This is an internal intermediate type — the expansion functions produce
/// these, then `nodes_to_chunks()` converts them to `ContextChunk`.
/// We keep this separate from `ContextChunk` to match the original code
/// structure and make the Cypher query functions easier to maintain.
pub(crate) struct ExpandedNode {
    pub id: String,
    pub node_type: String,
    pub title: String,
    pub properties: HashMap<String, String>,
}

/// A relationship between two expanded nodes (internal).
///
/// Converted to `RelatedNode` entries attached to `ContextChunk`s during
/// the final conversion step. Unlike `RelatedNode` (which is attached to
/// a specific chunk), this is a standalone edge with both endpoints.
pub(crate) struct ExpandedRel {
    pub from_id: String,
    pub to_id: String,
    pub rel_type: String,
}

impl ExpandedRel {
    pub fn new(from_id: &str, to_id: &str, rel_type: &str) -> Self {
        Self {
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            rel_type: rel_type.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Neo4jExpander struct
// ---------------------------------------------------------------------------

/// Expands context by traversing Neo4j knowledge graph relationships.
///
/// Given seed node IDs from vector search, the expander:
/// 1. Resolves each ID to its Neo4j label (node type)
/// 2. Dispatches to a type-specific Cypher query
/// 3. Collects neighbor nodes and relationships
/// 4. Deduplicates across all seeds
/// 5. Converts to `Vec<ContextChunk>` with `RelatedNode` entries
pub struct Neo4jExpander {
    /// The Neo4j connection pool.
    graph: Arc<Graph>,
}

impl Neo4jExpander {
    /// Create a new expander with a shared Neo4j connection.
    ///
    /// ## Example
    /// ```rust,ignore
    /// let graph = Arc::new(neo4rs::Graph::new("bolt://localhost:7687", "neo4j", "password").await?);
    /// let expander = Neo4jExpander::new(graph);
    /// ```
    pub fn new(graph: Arc<Graph>) -> Self {
        Self { graph }
    }
}

// ---------------------------------------------------------------------------
// GraphExpander trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl GraphExpander for Neo4jExpander {
    /// Expand seed nodes through the knowledge graph.
    ///
    /// ## Flow (matches original colossus-legal graph_expander.rs)
    /// 1. Resolve seed IDs → (id, node_type) pairs via batch Cypher query
    /// 2. For each seed, dispatch to type-specific expansion function
    /// 3. Per-seed error resilience — one failed query doesn't kill the batch
    /// 4. Deduplicate nodes across all seeds via shared `HashSet<String>`
    /// 5. Convert internal types to `Vec<ContextChunk>` with relationships
    ///
    /// ## Note on `max_depth`
    /// The current Cypher queries are fixed 1-hop traversals (OPTIONAL MATCH
    /// on direct neighbors). The `max_depth` parameter is accepted for future
    /// use but does not affect the current expansion depth.
    async fn expand(
        &self,
        seed_ids: &[String],
        _max_depth: u32,
    ) -> Result<Vec<ContextChunk>, RagError> {
        if seed_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Resolve seed IDs to (id, node_type) pairs.
        // The trait receives bare IDs but the expansion functions need the
        // Neo4j label to know which Cypher query to run.
        let typed_seeds = resolve_node_types(&self.graph, seed_ids).await?;

        tracing::info!(
            seed_count = typed_seeds.len(),
            "Graph expansion: starting with {} typed seeds",
            typed_seeds.len()
        );

        let mut all_nodes: Vec<ExpandedNode> = Vec::new();
        let mut all_rels: Vec<ExpandedRel> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut seeds_expanded = 0;

        // Step 2: Dispatch each seed to its type-specific expansion.
        //
        // ## Pattern: Per-seed error resilience
        // Each seed expansion is wrapped in a match so one failed Neo4j
        // query doesn't kill the entire expansion. Errors are logged and
        // the seed is skipped — other seeds still expand normally.
        for (node_id, node_type) in &typed_seeds {
            let result = match node_type.as_str() {
                "Evidence" => {
                    expander_queries::expand_evidence(&self.graph, node_id, &mut seen).await
                }
                "ComplaintAllegation" => {
                    expander_queries::expand_allegation(&self.graph, node_id, &mut seen).await
                }
                "MotionClaim" => {
                    expander_queries::expand_motion_claim(&self.graph, node_id, &mut seen).await
                }
                "Harm" => {
                    expander_queries::expand_harm(&self.graph, node_id, &mut seen).await
                }
                "Document" => {
                    expander_queries::expand_document(&self.graph, node_id, &mut seen).await
                }
                "Person" => {
                    expander_queries::expand_person(&self.graph, node_id, &mut seen).await
                }
                "Organization" => {
                    expander_queries::expand_organization(&self.graph, node_id, &mut seen).await
                }
                _ => {
                    tracing::warn!("Unknown node type for expansion: {node_type}");
                    continue;
                }
            };

            match result {
                Ok((nodes, rels)) => {
                    tracing::info!(
                        node_id,
                        node_type,
                        nodes_found = nodes.len(),
                        rels_found = rels.len(),
                        "Seed expanded successfully"
                    );
                    all_nodes.extend(nodes);
                    all_rels.extend(rels);
                    seeds_expanded += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        node_id,
                        node_type,
                        error = %e,
                        "Seed expansion failed (skipping)"
                    );
                }
            }
        }

        tracing::info!(
            seeds_expanded,
            total_nodes = all_nodes.len(),
            total_rels = all_rels.len(),
            "Graph expansion complete"
        );

        // Step 3: Convert internal types → ContextChunks with relationships.
        let mut chunks = nodes_to_chunks(all_nodes);
        attach_relationships(&mut chunks, &all_rels);

        Ok(chunks)
    }
}

// ---------------------------------------------------------------------------
// Helper: resolve node types via batch Neo4j query
// ---------------------------------------------------------------------------

/// Resolve a list of node IDs to (id, node_type) pairs.
///
/// Runs a single Cypher query to look up the Neo4j label for each ID.
/// IDs that don't exist in the graph are silently skipped (the MATCH
/// returns nothing for them).
///
/// ## Cypher: `labels(n)[0]`
///
/// Neo4j nodes can have multiple labels, but in the Awad v. CFS knowledge
/// graph, each node has exactly one label (Evidence, Person, Document, etc.).
/// `labels(n)[0]` gets the primary (first) label.
async fn resolve_node_types(
    graph: &Graph,
    ids: &[String],
) -> Result<Vec<(String, String)>, RagError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    let cypher = "UNWIND $ids AS node_id \
                  MATCH (n {id: node_id}) \
                  RETURN n.id AS id, labels(n)[0] AS node_type";

    let ids_vec: Vec<String> = ids.to_vec();
    let mut result = graph
        .execute(query(cypher).param("ids", ids_vec))
        .await
        .map_err(|e| RagError::ExpandError(format!("Type resolution query failed: {e}")))?;

    let mut pairs = Vec::new();
    while let Some(row) = result
        .next()
        .await
        .map_err(|e| RagError::ExpandError(format!("Type resolution row error: {e}")))?
    {
        let id: String = row.get("id").unwrap_or_default();
        let node_type: String = row.get("node_type").unwrap_or_default();
        if !id.is_empty() && !node_type.is_empty() {
            pairs.push((id, node_type));
        }
    }

    Ok(pairs)
}

// ---------------------------------------------------------------------------
// Conversion: ExpandedNode → ContextChunk
// ---------------------------------------------------------------------------

/// Convert a list of internal `ExpandedNode`s into `ContextChunk`s.
///
/// Each node becomes a `ContextChunk` with:
/// - `score = 0.0` (graph-expanded nodes have no similarity score)
/// - `content` built from type-specific properties
/// - `source` extracted from document/page properties if available
/// - Empty `relationships` (attached later by `attach_relationships`)
fn nodes_to_chunks(nodes: Vec<ExpandedNode>) -> Vec<ContextChunk> {
    nodes.into_iter().map(|node| {
        let content = build_content(&node);
        let source = build_source(&node);
        ContextChunk {
            node_id: node.id,
            node_type: node.node_type,
            title: node.title,
            content,
            score: 0.0,
            source,
            relationships: Vec::new(),
            metadata: serde_json::Value::Null,
        }
    }).collect()
}

/// Build the `content` field from type-specific properties.
///
/// Different node types have different meaningful text fields:
/// - Evidence: verbatim quote or significance
/// - ComplaintAllegation: the allegation text
/// - MotionClaim: the claim text
/// - Harm: description
/// - Others: title is sufficient
fn build_content(node: &ExpandedNode) -> String {
    match node.node_type.as_str() {
        "Evidence" => node.properties.get("verbatim_quote")
            .or(node.properties.get("significance"))
            .cloned()
            .unwrap_or_else(|| node.title.clone()),
        "ComplaintAllegation" => node.properties.get("allegation")
            .cloned()
            .unwrap_or_else(|| node.title.clone()),
        "MotionClaim" => node.properties.get("claim_text")
            .or(node.properties.get("significance"))
            .cloned()
            .unwrap_or_else(|| node.title.clone()),
        "Harm" => node.properties.get("description")
            .cloned()
            .unwrap_or_else(|| node.title.clone()),
        _ => node.title.clone(),
    }
}

/// Build the `source` field from Evidence properties (page number).
fn build_source(node: &ExpandedNode) -> SourceReference {
    let page = node.properties.get("page_number")
        .and_then(|p| p.parse::<u32>().ok());
    SourceReference {
        page_number: page,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Conversion: attach ExpandedRel → RelatedNode on ContextChunks
// ---------------------------------------------------------------------------

/// Attach relationship information to chunks as `RelatedNode` entries.
///
/// For each relationship edge (from → to), we attach:
/// - An **Outbound** `RelatedNode` on the `from` chunk (pointing to `to`)
/// - An **Inbound** `RelatedNode` on the `to` chunk (pointing to `from`)
///
/// This gives Claude full visibility into the graph structure from any
/// chunk's perspective. For example, if evidence STATED_BY speaker:
/// - Evidence chunk shows: "→ STATED_BY speaker-001 (Person)"
/// - Speaker chunk shows: "← STATED_BY evidence-001 (Evidence)"
///
/// ## Rust Learning: Building an owned HashMap to avoid borrow conflicts
///
/// We build `type_map` by cloning data from `chunks`, so it's fully owned.
/// This allows us to mutably borrow `chunks` in the loop below without
/// conflicting with the immutable borrow that would exist if `type_map`
/// held references to `chunks`.
fn attach_relationships(chunks: &mut [ContextChunk], rels: &[ExpandedRel]) {
    // Build a lookup: node_id → node_type (owned, not borrowing chunks).
    let type_map: HashMap<String, String> = chunks
        .iter()
        .map(|c| (c.node_id.clone(), c.node_type.clone()))
        .collect();

    for rel in rels {
        let to_type = type_map.get(&rel.to_id).cloned().unwrap_or_default();
        let from_type = type_map.get(&rel.from_id).cloned().unwrap_or_default();

        // Outbound on the "from" chunk.
        if let Some(chunk) = chunks.iter_mut().find(|c| c.node_id == rel.from_id) {
            chunk.relationships.push(RelatedNode {
                node_id: rel.to_id.clone(),
                node_type: to_type.clone(),
                relationship: rel.rel_type.clone(),
                direction: RelationDirection::Outbound,
                summary: String::new(),
            });
        }

        // Inbound on the "to" chunk.
        if let Some(chunk) = chunks.iter_mut().find(|c| c.node_id == rel.to_id) {
            chunk.relationships.push(RelatedNode {
                node_id: rel.from_id.clone(),
                node_type: from_type,
                relationship: rel.rel_type.clone(),
                direction: RelationDirection::Inbound,
                summary: String::new(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (used by expander_queries.rs)
// ---------------------------------------------------------------------------

/// Safe string extraction from a Neo4j row.
///
/// Returns "" if the column is null, missing, or fails to deserialize.
///
/// ## Rust Learning: `unwrap_or_default()` on `Result`
///
/// `row.get::<String>(key)` returns `Result<String, DeError>`.
/// `unwrap_or_default()` returns the Ok value if present, or
/// `String::default()` (empty string) if it's an Err. This is a safe
/// pattern for optional Neo4j properties — no panics, just empty strings.
pub(crate) fn get_str(row: &neo4rs::Row, key: &str) -> String {
    row.get(key).unwrap_or_default()
}

/// Try to extract a node from a Neo4j result row.
///
/// Returns `None` if:
/// - The ID column is empty (OPTIONAL MATCH didn't find it)
/// - The ID is already in `seen` (deduplication)
///
/// This is the core deduplication mechanism: the shared `HashSet<String>`
/// ensures each node appears at most once across all expansion calls,
/// even when multiple seeds share neighbors.
///
/// ## Parameters
/// - `row`: The Neo4j result row
/// - `id_col`: Column name for the node's ID
/// - `node_type`: The type label for this node (e.g., "Evidence", "Person")
/// - `prop_cols`: Pairs of (row_column, property_name) for extracting properties
/// - `seen`: Shared deduplication set
pub(crate) fn try_extract_node(
    row: &neo4rs::Row,
    id_col: &str,
    node_type: &str,
    prop_cols: &[(&str, &str)],
    seen: &mut HashSet<String>,
) -> Option<ExpandedNode> {
    let id: String = row.get(id_col).unwrap_or_default();
    if id.is_empty() || seen.contains(&id) {
        return None;
    }
    seen.insert(id.clone());

    let mut properties = HashMap::new();
    for (col, prop_name) in prop_cols {
        let val: String = row.get(col).unwrap_or_default();
        if !val.is_empty() {
            properties.insert(prop_name.to_string(), val);
        }
    }

    let title = properties
        .get("title")
        .or_else(|| properties.get("name"))
        .cloned()
        .unwrap_or_default();

    Some(ExpandedNode {
        id,
        node_type: node_type.to_string(),
        title,
        properties,
    })
}
