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

// ---------------------------------------------------------------------------
// Domain-agnostic enums for schema metadata
// ---------------------------------------------------------------------------

/// How an extracted entity must be grounded (anchored) in the source document.
///
/// Each variant is domain-agnostic:
/// - `Verbatim` — exact text match required (default)
/// - `NameMatch` — entity name appears somewhere in the document
/// - `HeadingMatch` — entity maps to a section title / heading
/// - `Derived` — entity is inferred; provenance links required
/// - `None` — not groundable in text (human-only annotation)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GroundingMode {
    #[default]
    Verbatim,
    NameMatch,
    HeadingMatch,
    Derived,
    None,
}

/// Extraction priority / exhaustiveness category for an entity type.
///
/// - `Foundation` — must be exhaustive; cannot be rejected by the LLM
/// - `Structural` — grounded with dependency links to other entities
/// - `Evidence` — standard grounded entity (default)
/// - `Reference` — existence-check only (no deep extraction)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EntityCategory {
    Foundation,
    Structural,
    #[default]
    Evidence,
    Reference,
}

/// Whether a document defines the skeleton (foundation) or links to one.
///
/// - `Foundation` — defines the case skeleton; completeness validation applies
/// - `Evidence` — links to an existing foundation document (default)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DocumentCategory {
    Foundation,
    #[default]
    Evidence,
}

/// A rule that checks extraction completeness after the LLM pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompletenessRule {
    /// Require at least `min` entities of the given type.
    EntityCount {
        entity: String,
        min: u32,
        message: String,
    },
    /// Require that at least `min_percentage`% of `from` entities have
    /// a `relationship` edge to a `to` entity.
    RelationshipExists {
        from: String,
        relationship: String,
        to: String,
        min_percentage: u32,
        message: String,
    },
}

/// Complete extraction schema loaded from a YAML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    /// Document type this schema applies to (e.g. "complaint", "affidavit")
    pub document_type: String,

    /// Schema version string (defaults to "1.0" for backward compatibility)
    #[serde(default = "default_version")]
    pub version: String,

    /// Whether this document defines a foundation or links to one
    #[serde(default)]
    pub document_category: DocumentCategory,

    /// Human-readable description of this document type
    #[serde(default)]
    pub description: String,

    /// Entity types the LLM should extract
    pub entity_types: Vec<EntityTypeConfig>,

    /// Relationship types the LLM should extract
    pub relationship_types: Vec<RelationshipTypeConfig>,

    /// Valid patterns constraining which entity types can connect
    pub valid_patterns: Vec<PatternConfig>,

    /// Post-extraction completeness checks
    #[serde(default)]
    pub completeness_rules: Vec<CompletenessRule>,

    /// Extraction rules included in the LLM prompt
    #[serde(default)]
    pub extraction_rules: Vec<String>,
}

/// Configuration for a single entity type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTypeConfig {
    /// Entity type name (e.g. "Party", "FactualAllegation")
    pub name: String,

    /// Extraction priority category (defaults to Evidence)
    #[serde(default)]
    pub category: EntityCategory,

    /// Whether this entity type must be present in a valid extraction.
    ///
    /// Required in YAML. Removed default per silent-fallback audit defect
    /// #4.1.1: completeness schemas silently accepted zero entities when this
    /// field was absent. Schema YAMLs must now declare the value explicitly.
    pub required: bool,

    /// Minimum number of entities expected (0 = no minimum).
    ///
    /// Required in YAML. Removed default per silent-fallback audit defect
    /// #4.1.1: see `required` above.
    pub min_count: u32,

    /// How this entity must be grounded in the source text
    #[serde(default)]
    pub grounding_mode: GroundingMode,

    /// Whether provenance links are required for this entity
    #[serde(default)]
    pub provenance_required: bool,

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

fn default_version() -> String {
    "1.0".to_string()
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
        let content = std::fs::read_to_string(path).map_err(|e| {
            PipelineError::Schema(format!(
                "Failed to read schema file {}: {}",
                path.display(),
                e
            ))
        })?;
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
        let entity_names: HashSet<&str> =
            self.entity_types.iter().map(|e| e.name.as_str()).collect();

        let relationship_names: HashSet<&str> = self
            .relationship_types
            .iter()
            .map(|r| r.name.as_str())
            .collect();

        // Check that all patterns reference valid entity and relationship types
        for pattern in &self.valid_patterns {
            if !entity_names.contains(pattern.from.as_str()) {
                return Err(PipelineError::Schema(format!(
                    "Pattern references unknown entity type '{}' in 'from'",
                    pattern.from,
                )));
            }
            if !entity_names.contains(pattern.to.as_str()) {
                return Err(PipelineError::Schema(format!(
                    "Pattern references unknown entity type '{}' in 'to'",
                    pattern.to,
                )));
            }
            if !relationship_names.contains(pattern.relationship.as_str()) {
                return Err(PipelineError::Schema(format!(
                    "Pattern references unknown relationship type '{}'",
                    pattern.relationship,
                )));
            }
        }

        // Check for duplicate entity type names
        if entity_names.len() != self.entity_types.len() {
            return Err(PipelineError::Schema(
                "Duplicate entity type names found".to_string(),
            ));
        }

        // Check for duplicate relationship type names
        if relationship_names.len() != self.relationship_types.len() {
            return Err(PipelineError::Schema(
                "Duplicate relationship type names found".to_string(),
            ));
        }

        // Foundation documents need at least one foundation + required entity
        if self.document_category == DocumentCategory::Foundation {
            let has_foundation = self
                .entity_types
                .iter()
                .any(|e| e.category == EntityCategory::Foundation && e.required);
            if !has_foundation {
                return Err(PipelineError::Schema(
                    "Document category is foundation but no entity type has \
                     category = foundation with required = true"
                        .to_string(),
                ));
            }
        }

        // Completeness rules must reference existing types
        for rule in &self.completeness_rules {
            match rule {
                CompletenessRule::EntityCount { entity, .. } => {
                    if !entity_names.contains(entity.as_str()) {
                        return Err(PipelineError::Schema(format!(
                            "Completeness rule references unknown entity type '{entity}'"
                        )));
                    }
                }
                CompletenessRule::RelationshipExists {
                    from,
                    relationship,
                    to,
                    ..
                } => {
                    if !entity_names.contains(from.as_str()) {
                        return Err(PipelineError::Schema(format!(
                            "Completeness rule references unknown entity type '{from}'"
                        )));
                    }
                    if !entity_names.contains(to.as_str()) {
                        return Err(PipelineError::Schema(format!(
                            "Completeness rule references unknown entity type '{to}'"
                        )));
                    }
                    if !relationship_names.contains(relationship.as_str()) {
                        return Err(PipelineError::Schema(format!(
                            "Completeness rule references unknown relationship type \
                             '{relationship}'"
                        )));
                    }
                }
            }
        }

        // Required entities must have min_count > 0
        for entity in &self.entity_types {
            if entity.required && entity.min_count == 0 {
                return Err(PipelineError::Schema(format!(
                    "Entity type '{}' is required but min_count is 0",
                    entity.name,
                )));
            }
        }

        // Warn (don't error) if derived grounding lacks provenance flag
        for entity in &self.entity_types {
            if entity.grounding_mode == GroundingMode::Derived && !entity.provenance_required {
                tracing::warn!(
                    entity = %entity.name,
                    "Entity has grounding_mode = derived but provenance_required = false"
                );
            }
        }

        Ok(())
    }

    /// Convert the schema to a JSON string for inclusion in LLM prompts.
    pub fn to_prompt_json(&self) -> Result<String, PipelineError> {
        serde_json::to_string_pretty(self).map_err(PipelineError::from)
    }
}
