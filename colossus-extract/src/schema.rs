//! Extraction schema definition and YAML loader.
//!
//! Schemas define what entity types, relationship types, and valid patterns
//! the LLM should extract from a document. One schema file per document type,
//! loaded at runtime from YAML.
//!
//! ## Rust Learning: Deserialize with validation
//!
//! Serde handles the YAML->struct conversion, but we add a `validate()` method
//! to check semantic correctness (e.g., patterns reference valid entity types).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use crate::error::PipelineError;

/// Complete extraction schema loaded from a YAML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    /// Document type this schema applies to (e.g. "complaint", "affidavit")
    pub document_type: String,

    /// Human-readable description of this document type
    #[serde(default)]
    pub description: String,

    /// Entity types the LLM should extract
    pub entity_types: Vec<EntityTypeConfig>,

    /// Relationship types the LLM should extract
    pub relationship_types: Vec<RelationshipTypeConfig>,

    /// Valid patterns constraining which entity types can connect
    pub valid_patterns: Vec<PatternConfig>,

    /// Extraction rules included in the LLM prompt
    #[serde(default)]
    pub extraction_rules: Vec<String>,
}

/// Configuration for a single entity type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTypeConfig {
    /// Entity type name (e.g. "Party", "FactualAllegation")
    pub name: String,

    /// Description included in the LLM prompt to guide extraction
    #[serde(default)]
    pub description: String,

    /// Properties this entity type should have
    #[serde(default)]
    pub properties: Vec<PropertyConfig>,
}

/// Configuration for a single relationship type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipTypeConfig {
    /// Relationship type name (e.g. "STATED_BY", "SUPPORTS")
    pub name: String,

    /// Description included in the LLM prompt
    #[serde(default)]
    pub description: String,

    /// Properties on this relationship type
    #[serde(default)]
    pub properties: Vec<PropertyConfig>,
}

/// Configuration for an entity or relationship property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyConfig {
    /// Property name
    pub name: String,

    /// Property type (string, integer, boolean, etc.)
    #[serde(rename = "type", default = "default_property_type")]
    pub property_type: String,

    /// Whether this property is required
    #[serde(default)]
    pub required: bool,

    /// Description included in the LLM prompt
    #[serde(default)]
    pub description: String,
}

fn default_property_type() -> String {
    "string".to_string()
}

/// A valid connection pattern between entity types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Source entity type
    pub from: String,

    /// Relationship type
    pub relationship: String,

    /// Target entity type
    pub to: String,
}

impl ExtractionSchema {
    /// Load a schema from a YAML file.
    pub fn from_file(path: &Path) -> Result<Self, PipelineError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PipelineError::Schema(
                format!("Failed to read schema file {}: {}", path.display(), e)
            ))?;
        let schema: Self = serde_yaml::from_str(&content)?;
        schema.validate()?;
        Ok(schema)
    }

    /// Load a schema from a YAML string (useful for testing).
    pub fn from_yaml_str(yaml: &str) -> Result<Self, PipelineError> {
        let schema: Self = serde_yaml::from_str(yaml)?;
        schema.validate()?;
        Ok(schema)
    }

    /// Validate semantic correctness of the schema.
    pub fn validate(&self) -> Result<(), PipelineError> {
        let entity_names: HashSet<&str> = self.entity_types
            .iter()
            .map(|e| e.name.as_str())
            .collect();

        let relationship_names: HashSet<&str> = self.relationship_types
            .iter()
            .map(|r| r.name.as_str())
            .collect();

        // Check that all patterns reference valid entity and relationship types
        for pattern in &self.valid_patterns {
            if !entity_names.contains(pattern.from.as_str()) {
                return Err(PipelineError::Schema(
                    format!(
                        "Pattern references unknown entity type '{}' in 'from'",
                        pattern.from,
                    )
                ));
            }
            if !entity_names.contains(pattern.to.as_str()) {
                return Err(PipelineError::Schema(
                    format!(
                        "Pattern references unknown entity type '{}' in 'to'",
                        pattern.to,
                    )
                ));
            }
            if !relationship_names.contains(pattern.relationship.as_str()) {
                return Err(PipelineError::Schema(
                    format!(
                        "Pattern references unknown relationship type '{}'",
                        pattern.relationship,
                    )
                ));
            }
        }

        // Check for duplicate entity type names
        if entity_names.len() != self.entity_types.len() {
            return Err(PipelineError::Schema(
                "Duplicate entity type names found".to_string()
            ));
        }

        // Check for duplicate relationship type names
        if relationship_names.len() != self.relationship_types.len() {
            return Err(PipelineError::Schema(
                "Duplicate relationship type names found".to_string()
            ));
        }

        Ok(())
    }

    /// Convert the schema to a JSON string for inclusion in LLM prompts.
    pub fn to_prompt_json(&self) -> Result<String, PipelineError> {
        serde_json::to_string_pretty(self).map_err(PipelineError::from)
    }
}
