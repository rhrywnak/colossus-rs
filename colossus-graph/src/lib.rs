//! colossus-graph — schema-agnostic Neo4j graph data access.
//!
//! Provides domain-agnostic types and query functions for interacting
//! with Neo4j graphs where node labels and relationship types are not
//! known at compile time.
//!
//! ## Usage
//!
//! ```toml
//! [dependencies]
//! colossus-graph = { path = "../colossus-rs/colossus-graph", features = ["neo4j"] }
//! ```
//!
//! ## Feature Flags
//!
//! - `neo4j` — enables Neo4j query functions (requires neo4rs)

pub mod error;
pub mod types;

#[cfg(feature = "neo4j")]
pub mod queries;

// Re-export core types at crate root for convenience.
pub use error::GraphAccessError;
pub use types::*;

#[cfg(feature = "neo4j")]
pub use queries::*;
