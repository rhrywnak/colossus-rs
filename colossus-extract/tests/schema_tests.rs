//! Tests for ExtractionSchema — YAML loading, validation, and JSON export.

use std::path::PathBuf;

use colossus_extract::{DocumentCategory, EntityCategory, ExtractionSchema, GroundingMode};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/schemas")
        .join(name)
}

fn complaint_yaml() -> String {
    std::fs::read_to_string(fixture_path("complaint.yaml"))
        .expect("complaint.yaml fixture should exist")
}

#[test]
fn test_load_complaint_schema_from_str() {
    let schema =
        ExtractionSchema::from_yaml_str(&complaint_yaml()).expect("complaint schema should parse");
    assert_eq!(schema.document_type, "complaint");
}

#[test]
fn test_load_complaint_schema_from_file() {
    let schema = ExtractionSchema::from_file(&fixture_path("complaint.yaml"))
        .expect("complaint schema should load from file");
    assert_eq!(schema.document_type, "complaint");
}

#[test]
fn test_entity_type_count() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    assert_eq!(schema.entity_types.len(), 4, "Expected 4 entity types");
}

#[test]
fn test_relationship_type_count() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    assert_eq!(
        schema.relationship_types.len(),
        6,
        "Expected 6 relationship types"
    );
}

#[test]
fn test_pattern_count() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    assert_eq!(schema.valid_patterns.len(), 5, "Expected 5 patterns");
}

#[test]
fn test_extraction_rules_loaded() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    assert_eq!(
        schema.extraction_rules.len(),
        4,
        "Expected 4 extraction rules"
    );
}

#[test]
fn test_entity_type_properties() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    let party = schema
        .entity_types
        .iter()
        .find(|e| e.name == "Party")
        .expect("Party entity type should exist");
    assert_eq!(party.properties.len(), 3, "Party should have 3 properties");

    let role_prop = party
        .properties
        .iter()
        .find(|p| p.name == "role")
        .expect("role property should exist");
    assert!(role_prop.required, "role should be required");
    assert_eq!(role_prop.property_type, "string");
}

#[test]
fn test_validate_passes_for_valid_schema() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    assert!(schema.validate().is_ok());
}

#[test]
fn test_validate_fails_for_invalid_from_entity() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
relationship_types:
  - name: KNOWS
valid_patterns:
  - from: Ghost
    relationship: KNOWS
    to: Person
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Ghost"), "Error should mention 'Ghost': {msg}");
    assert!(msg.contains("from"), "Error should mention 'from': {msg}");
}

#[test]
fn test_validate_fails_for_invalid_to_entity() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
relationship_types:
  - name: KNOWS
valid_patterns:
  - from: Person
    relationship: KNOWS
    to: Ghost
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Ghost"), "Error should mention 'Ghost': {msg}");
    assert!(msg.contains("to"), "Error should mention 'to': {msg}");
}

#[test]
fn test_validate_fails_for_invalid_relationship() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
relationship_types:
  - name: KNOWS
valid_patterns:
  - from: Person
    relationship: LIKES
    to: Person
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("LIKES"), "Error should mention 'LIKES': {msg}");
}

#[test]
fn test_validate_fails_for_duplicate_entity_names() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
  - name: Person
relationship_types:
  - name: KNOWS
valid_patterns: []
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Duplicate"),
        "Error should mention duplicates: {msg}"
    );
}

#[test]
fn test_validate_fails_for_duplicate_relationship_names() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
relationship_types:
  - name: KNOWS
  - name: KNOWS
valid_patterns: []
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Duplicate"),
        "Error should mention duplicates: {msg}"
    );
}

#[test]
fn test_to_prompt_json_produces_valid_json() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_yaml()).unwrap();
    let json_str = schema.to_prompt_json().expect("should produce JSON");

    // Verify it's valid JSON by parsing it
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("prompt JSON should be valid");
    assert_eq!(parsed["document_type"], "complaint");
    assert!(parsed["entity_types"].is_array());
}

#[test]
fn test_property_type_defaults_to_string() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Thing
    properties:
      - name: label
relationship_types: []
valid_patterns: []
"#;
    let schema = ExtractionSchema::from_yaml_str(yaml).unwrap();
    let prop = &schema.entity_types[0].properties[0];
    assert_eq!(prop.property_type, "string");
}

#[test]
fn test_description_defaults_to_empty() {
    let yaml = r#"
document_type: minimal
entity_types:
  - name: Thing
relationship_types: []
valid_patterns: []
"#;
    let schema = ExtractionSchema::from_yaml_str(yaml).unwrap();
    assert_eq!(schema.description, "");
}

#[test]
fn test_from_file_nonexistent_returns_error() {
    let result = ExtractionSchema::from_file(&PathBuf::from("/nonexistent/schema.yaml"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Failed to read"),
        "Error should describe read failure: {msg}"
    );
}

// ---------------------------------------------------------------------------
// V2 schema tests
// ---------------------------------------------------------------------------

fn complaint_v2_yaml() -> String {
    std::fs::read_to_string(fixture_path("complaint_v2.yaml"))
        .expect("complaint_v2.yaml fixture should exist")
}

#[test]
fn test_load_complaint_v2_schema() {
    let schema =
        ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).expect("v2 schema should parse");
    assert_eq!(schema.document_type, "complaint");
    assert_eq!(schema.version, "2.0");
    assert_eq!(schema.document_category, DocumentCategory::Foundation);
}

#[test]
fn test_v2_entity_categories() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).unwrap();
    let party = schema
        .entity_types
        .iter()
        .find(|e| e.name == "Party")
        .expect("Party should exist");
    assert_eq!(party.category, EntityCategory::Foundation);
    assert!(party.required);
    assert_eq!(party.min_count, 2);
    assert_eq!(party.grounding_mode, GroundingMode::NameMatch);
}

#[test]
fn test_v2_legal_count_grounding() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).unwrap();
    let lc = schema
        .entity_types
        .iter()
        .find(|e| e.name == "LegalCount")
        .expect("LegalCount should exist");
    assert_eq!(lc.grounding_mode, GroundingMode::HeadingMatch);
    assert_eq!(lc.category, EntityCategory::Foundation);
}

#[test]
fn test_v2_allegation_structural() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).unwrap();
    let alleg = schema
        .entity_types
        .iter()
        .find(|e| e.name == "ComplaintAllegation")
        .expect("ComplaintAllegation should exist");
    assert_eq!(alleg.category, EntityCategory::Structural);
    assert_eq!(alleg.grounding_mode, GroundingMode::Verbatim);
}

#[test]
fn test_v2_harm_derived() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).unwrap();
    let harm = schema
        .entity_types
        .iter()
        .find(|e| e.name == "Harm")
        .expect("Harm should exist");
    assert_eq!(harm.grounding_mode, GroundingMode::Derived);
    assert!(harm.provenance_required);
    assert!(!harm.required);
}

#[test]
fn test_v2_completeness_rules() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).unwrap();
    assert_eq!(schema.completeness_rules.len(), 3);
}

#[test]
fn test_backward_compat_v1_schema() {
    let schema =
        ExtractionSchema::from_yaml_str(&complaint_yaml()).expect("v1 schema should still load");
    assert_eq!(schema.version, "1.0");
    assert_eq!(schema.document_category, DocumentCategory::Evidence);
    for entity in &schema.entity_types {
        assert_eq!(
            entity.category,
            EntityCategory::Evidence,
            "Default category should be Evidence for '{}'",
            entity.name
        );
        assert_eq!(
            entity.grounding_mode,
            GroundingMode::Verbatim,
            "Default grounding should be Verbatim for '{}'",
            entity.name
        );
    }
}

#[test]
fn test_validate_foundation_needs_foundation_entity() {
    let yaml = r#"
document_type: test
document_category: foundation
entity_types:
  - name: Thing
    category: evidence
    required: true
    min_count: 1
relationship_types:
  - name: LINKS
valid_patterns: []
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("foundation"),
        "Error should mention 'foundation': {msg}"
    );
}

#[test]
fn test_validate_completeness_rule_unknown_entity() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
relationship_types:
  - name: KNOWS
valid_patterns: []
completeness_rules:
  - type: entity_count
    entity: Ghost
    min: 1
    message: "ghost check"
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Ghost"), "Error should mention 'Ghost': {msg}");
}

#[test]
fn test_validate_completeness_rule_unknown_relationship() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Person
relationship_types:
  - name: KNOWS
valid_patterns: []
completeness_rules:
  - type: relationship_exists
    from: Person
    relationship: GHOST_REL
    to: Person
    min_percentage: 50
    message: "ghost rel check"
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("GHOST_REL"),
        "Error should mention 'GHOST_REL': {msg}"
    );
}

#[test]
fn test_validate_required_without_min_count() {
    let yaml = r#"
document_type: test
entity_types:
  - name: Thing
    required: true
    min_count: 0
relationship_types: []
valid_patterns: []
"#;
    let err = ExtractionSchema::from_yaml_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("min_count"),
        "Error should mention 'min_count': {msg}"
    );
}

#[test]
fn test_v2_prompt_json_includes_new_fields() {
    let schema = ExtractionSchema::from_yaml_str(&complaint_v2_yaml()).unwrap();
    let json_str = schema.to_prompt_json().expect("should produce JSON");
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("prompt JSON should be valid");
    assert!(
        parsed.get("version").is_some(),
        "JSON should have 'version'"
    );
    assert!(
        parsed.get("document_category").is_some(),
        "JSON should have 'document_category'"
    );
    let first_entity = &parsed["entity_types"][0];
    assert!(
        first_entity.get("category").is_some(),
        "Entity should have 'category'"
    );
    assert!(
        first_entity.get("grounding_mode").is_some(),
        "Entity should have 'grounding_mode'"
    );
}
