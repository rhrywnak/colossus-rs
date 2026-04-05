//! Schema-agnostic Neo4j query functions.
//!
//! Replicates the query patterns from Neo4j Labs' llm-graph-builder:
//! - Query by property, not by label
//! - Generic neighborhood traversal
//! - Label introspection via labels(n) and type(r)
//! - Document-scoped queries via source property
//!
//! All functions take a `&neo4rs::Graph` and return domain-agnostic types.
//! No application-specific logic lives here.

use std::collections::HashMap;

use neo4rs::{query, Graph, Node, Row};

use crate::error::GraphAccessError;
use crate::types::{GraphNode, GraphRelationship, LabelCount, NodeNeighborhood};

// ---------------------------------------------------------------------------
// Helpers: property extraction from neo4rs types
// ---------------------------------------------------------------------------

/// Extract all properties from a neo4rs::Node into a HashMap.
///
/// ## Rust Learning: Type probing with neo4rs
///
/// neo4rs::Node stores properties as typed values internally. Since we
/// don't know the types at compile time, we try extracting each property
/// as String, then i64, then f64, then bool. The first successful extraction
/// wins. This is similar to dynamic typing but explicit — Rust makes you
/// handle each case.
///
/// neo4rs also supports Vec<String> for array properties (like
/// `source_documents`). We handle that case too.
fn extract_node_properties(node: &Node) -> HashMap<String, serde_json::Value> {
    let mut props = HashMap::new();
    for key in node.keys() {
        let value = if let Ok(v) = node.get::<String>(key) {
            serde_json::Value::String(v)
        } else if let Ok(v) = node.get::<i64>(key) {
            serde_json::Value::Number(v.into())
        } else if let Ok(v) = node.get::<f64>(key) {
            serde_json::json!(v)
        } else if let Ok(v) = node.get::<bool>(key) {
            serde_json::Value::Bool(v)
        } else if let Ok(v) = node.get::<Vec<String>>(key) {
            serde_json::Value::Array(v.into_iter().map(serde_json::Value::String).collect())
        } else {
            serde_json::Value::Null
        };
        props.insert(key.to_string(), value);
    }
    props
}

/// Extract a GraphNode from a neo4rs Row.
///
/// Expects the row to contain a node (at `node_key`) and a labels array
/// (at `labels_key`). The labels come from `labels(n)` in the Cypher query
/// rather than from `node.labels()`, because the Row-level extraction gives
/// us owned `Vec<String>` directly — no lifetime concerns.
fn row_to_graph_node(row: &Row, node_key: &str, labels_key: &str) -> Option<GraphNode> {
    let node: Node = row.get(node_key).ok()?;
    let labels: Vec<String> = row.get(labels_key).ok().unwrap_or_default();
    let properties = extract_node_properties(&node);
    let id = properties
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Some(GraphNode {
        id,
        labels,
        properties,
    })
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

/// Fetch a single node by its application-level `id` property.
///
/// Returns `None` if no node with this ID exists in the graph.
/// The query is label-agnostic — it matches ANY node with `{id: $id}`.
pub async fn get_node_by_id(
    graph: &Graph,
    id: &str,
) -> Result<Option<GraphNode>, GraphAccessError> {
    let cypher = "MATCH (n {id: $id}) RETURN n, labels(n) AS node_labels LIMIT 1";

    let mut result = graph.execute(query(cypher).param("id", id)).await?;

    if let Some(row) = result.next().await? {
        Ok(row_to_graph_node(&row, "n", "node_labels"))
    } else {
        Ok(None)
    }
}

/// Fetch a node and all directly connected neighbors.
///
/// Returns a `NodeNeighborhood` containing the center node, all neighbor
/// nodes (deduplicated by id), and all relationships between them.
///
/// ## Rust Learning: OPTIONAL MATCH for nullable results
///
/// `OPTIONAL MATCH (n)-[r]-(m)` returns the center node even if it has
/// no relationships. Without `OPTIONAL`, an isolated node would return
/// zero rows. The trade-off: `m`, `r`, and related columns can be null,
/// so we must handle `None` when extracting neighbors.
pub async fn get_node_neighbors(
    graph: &Graph,
    id: &str,
) -> Result<NodeNeighborhood, GraphAccessError> {
    let cypher = "MATCH (n {id: $id})
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, labels(n) AS n_labels,
               m, labels(m) AS m_labels,
               type(r) AS rel_type,
               startNode(r) = n AS outgoing,
               m.id AS m_id";

    let mut result = graph.execute(query(cypher).param("id", id)).await?;

    let mut center_node: Option<GraphNode> = None;
    let mut neighbors: Vec<GraphNode> = Vec::new();
    let mut relationships: Vec<GraphRelationship> = Vec::new();
    let mut seen_neighbor_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    while let Some(row) = result.next().await? {
        // Extract center node (only on first row).
        if center_node.is_none() {
            center_node = row_to_graph_node(&row, "n", "n_labels");
        }

        // Extract neighbor (may be null from OPTIONAL MATCH).
        let m_id: String = row.get("m_id").unwrap_or_default();
        if m_id.is_empty() || seen_neighbor_ids.contains(&m_id) {
            // Null neighbor (isolated node) or already seen — skip neighbor
            // but still no relationship to add.
            continue;
        }

        if let Some(neighbor) = row_to_graph_node(&row, "m", "m_labels") {
            seen_neighbor_ids.insert(m_id.clone());
            neighbors.push(neighbor);
        }

        // Extract relationship direction and type.
        let rel_type: String = row.get("rel_type").unwrap_or_default();
        let outgoing: bool = row.get("outgoing").unwrap_or(true);

        if !rel_type.is_empty() {
            let (source_id, target_id) = if outgoing {
                (id.to_string(), m_id)
            } else {
                (m_id, id.to_string())
            };
            relationships.push(GraphRelationship {
                rel_type,
                source_id,
                target_id,
                properties: HashMap::new(),
            });
        }
    }

    Ok(NodeNeighborhood {
        node: center_node,
        neighbors,
        relationships,
    })
}

/// Fetch all nodes belonging to a specific document.
///
/// Matches on both `source_document` and `source_document_id` properties,
/// since different ingestion pipelines may use either convention.
pub async fn get_nodes_by_document(
    graph: &Graph,
    document_id: &str,
) -> Result<Vec<GraphNode>, GraphAccessError> {
    let cypher = "MATCH (n)
        WHERE n.source_document = $doc_id OR n.source_document_id = $doc_id
        RETURN n, labels(n) AS node_labels";

    let mut result = graph
        .execute(query(cypher).param("doc_id", document_id))
        .await?;

    let mut nodes = Vec::new();
    while let Some(row) = result.next().await? {
        if let Some(node) = row_to_graph_node(&row, "n", "node_labels") {
            nodes.push(node);
        }
    }
    Ok(nodes)
}

/// Fetch all nodes with a specific primary label.
///
/// Uses `labels(n)[0]` to match the primary (first) label on each node.
/// In most Colossus graphs, nodes have exactly one label.
pub async fn get_nodes_by_label(
    graph: &Graph,
    label: &str,
) -> Result<Vec<GraphNode>, GraphAccessError> {
    let cypher = "MATCH (n) WHERE labels(n)[0] = $label RETURN n, labels(n) AS node_labels";

    let mut result = graph.execute(query(cypher).param("label", label)).await?;

    let mut nodes = Vec::new();
    while let Some(row) = result.next().await? {
        if let Some(node) = row_to_graph_node(&row, "n", "node_labels") {
            nodes.push(node);
        }
    }
    Ok(nodes)
}

/// Fetch all nodes that have a specific property (regardless of value).
///
/// ## Rust Learning: Dynamic property names in Cypher
///
/// Neo4j Cypher does NOT support `n[$var]` syntax with parameters for
/// property names. We interpolate the property name into the query string.
/// This is safe ONLY because we validate the name first — rejecting
/// anything that isn't alphanumeric or underscore to prevent injection.
pub async fn get_nodes_with_property(
    graph: &Graph,
    property_name: &str,
) -> Result<Vec<GraphNode>, GraphAccessError> {
    // Validate property name to prevent Cypher injection.
    if !property_name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_')
    {
        return Err(GraphAccessError::QueryFailed(format!(
            "Invalid property name: {}",
            property_name
        )));
    }

    let cypher = format!(
        "MATCH (n) WHERE n.{} IS NOT NULL RETURN n, labels(n) AS node_labels",
        property_name
    );

    let mut result = graph.execute(query(&cypher)).await?;

    let mut nodes = Vec::new();
    while let Some(row) = result.next().await? {
        if let Some(node) = row_to_graph_node(&row, "n", "node_labels") {
            nodes.push(node);
        }
    }
    Ok(nodes)
}

/// Get counts of nodes grouped by their primary label.
///
/// Useful for graph introspection — understanding what kinds of nodes
/// exist and how many of each. Results are sorted by count descending.
pub async fn get_label_counts(graph: &Graph) -> Result<Vec<LabelCount>, GraphAccessError> {
    let cypher =
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC";

    let mut result = graph.execute(query(cypher)).await?;

    let mut counts = Vec::new();
    while let Some(row) = result.next().await? {
        let label: String = row.get("label").unwrap_or_default();
        let count: i64 = row.get("count").unwrap_or(0);
        if !label.is_empty() {
            counts.push(LabelCount { label, count });
        }
    }
    Ok(counts)
}

/// Get counts of relationships grouped by type.
///
/// Complements `get_label_counts` — together they give a full overview
/// of the graph schema. Results are sorted by count descending.
pub async fn get_relationship_type_counts(
    graph: &Graph,
) -> Result<Vec<LabelCount>, GraphAccessError> {
    let cypher =
        "MATCH ()-[r]->() RETURN type(r) AS label, count(r) AS count ORDER BY count DESC";

    let mut result = graph.execute(query(cypher)).await?;

    let mut counts = Vec::new();
    while let Some(row) = result.next().await? {
        let label: String = row.get("label").unwrap_or_default();
        let count: i64 = row.get("count").unwrap_or(0);
        if !label.is_empty() {
            counts.push(LabelCount { label, count });
        }
    }
    Ok(counts)
}

/// Get all relationships connected to a specific node.
///
/// Returns relationships in both directions (incoming and outgoing).
/// The `outgoing` boolean from `startNode(r) = n` determines which
/// end of the relationship is source vs target.
pub async fn get_all_relationships_for_node(
    graph: &Graph,
    node_id: &str,
) -> Result<Vec<GraphRelationship>, GraphAccessError> {
    let cypher = "MATCH (n {id: $id})-[r]-(m)
        RETURN type(r) AS rel_type, startNode(r) = n AS outgoing,
               n.id AS n_id, m.id AS m_id";

    let mut result = graph.execute(query(cypher).param("id", node_id)).await?;

    let mut relationships = Vec::new();
    while let Some(row) = result.next().await? {
        let rel_type: String = row.get("rel_type").unwrap_or_default();
        let outgoing: bool = row.get("outgoing").unwrap_or(true);
        let n_id: String = row.get("n_id").unwrap_or_default();
        let m_id: String = row.get("m_id").unwrap_or_default();

        if !rel_type.is_empty() {
            let (source_id, target_id) = if outgoing {
                (n_id, m_id)
            } else {
                (m_id, n_id)
            };
            relationships.push(GraphRelationship {
                rel_type,
                source_id,
                target_id,
                properties: HashMap::new(),
            });
        }
    }
    Ok(relationships)
}
