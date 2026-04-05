//! Error types for colossus-graph.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphAccessError {
    #[error("Neo4j query failed: {0}")]
    QueryFailed(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Property extraction failed: {0}")]
    PropertyExtraction(String),
}

#[cfg(feature = "neo4j")]
impl From<neo4rs::Error> for GraphAccessError {
    fn from(e: neo4rs::Error) -> Self {
        GraphAccessError::QueryFailed(e.to_string())
    }
}
