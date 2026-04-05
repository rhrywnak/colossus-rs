//! Core types for schema-agnostic graph data access.
//!
//! ## Rust Learning: serde_json::Value as a schema-agnostic property container
//!
//! Since we don't know at compile time what node labels or properties exist
//! in the graph, we store properties as `HashMap<String, serde_json::Value>`.
//! This is Rust's equivalent of a Python dict or JavaScript object — it can
//! hold any JSON-compatible value (strings, numbers, booleans, arrays, nulls).
//! The consumer inspects the node's `labels` to decide how to interpret the
//! properties.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A node retrieved from the graph with its labels and all properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// The node's `id` property (application-level ID, not Neo4j internal ID).
    pub id: String,
    /// Neo4j labels on this node (e.g., ["Person"] or ["ComplaintAllegation"]).
    pub labels: Vec<String>,
    /// All properties on the node as key-value pairs.
    pub properties: HashMap<String, serde_json::Value>,
}

/// A relationship retrieved from the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelationship {
    /// Relationship type name (e.g., "STATED_BY", "CONTAINS").
    pub rel_type: String,
    /// Application-level ID of the source (start) node.
    pub source_id: String,
    /// Application-level ID of the target (end) node.
    pub target_id: String,
    /// Properties on the relationship (often empty).
    pub properties: HashMap<String, serde_json::Value>,
}

/// A node and its directly connected neighbors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeNeighborhood {
    /// The center node (None if ID not found).
    pub node: Option<GraphNode>,
    /// All directly connected neighbor nodes.
    pub neighbors: Vec<GraphNode>,
    /// All relationships between the center node and its neighbors.
    pub relationships: Vec<GraphRelationship>,
}

/// Label and count pair from graph introspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelCount {
    /// The label or relationship type name.
    pub label: String,
    /// Number of nodes/relationships with this label/type.
    pub count: i64,
}
