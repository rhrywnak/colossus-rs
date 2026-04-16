//! Tests for PromptBuilder — template loading and variable substitution.

use std::path::PathBuf;

use colossus_extract::{ExtractionSchema, PromptBuilder};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn template_dir() -> PathBuf {
    fixtures_dir().join("templates")
}

fn load_complaint_schema() -> ExtractionSchema {
    let schema_path = fixtures_dir().join("schemas/complaint.yaml");
    ExtractionSchema::from_file(&schema_path).expect("complaint schema fixture should load")
}

#[test]
fn test_prompt_builder_creates_with_template_dir() {
    let dir = template_dir();
    let _builder = PromptBuilder::new(&dir);
    // No panic — construction succeeds (template loading is lazy)
}

#[test]
fn test_build_extraction_prompt_returns_nonempty() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Sample document text.", None, None, None, None)
        .expect("should build prompt");
    assert!(
        !artifact.prompt_text.is_empty(),
        "Prompt should not be empty"
    );
}

#[test]
fn test_extraction_prompt_contains_schema_json() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Sample document text.", None, None, None, None)
        .unwrap();
    // The schema JSON should contain the document type
    assert!(
        artifact
            .prompt_text
            .contains("\"document_type\": \"complaint\""),
        "Prompt should contain schema JSON with document_type"
    );
    assert!(
        artifact.prompt_text.contains("\"entity_types\""),
        "Prompt should contain entity_types from schema"
    );
}

#[test]
fn test_extraction_prompt_contains_document_text() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let doc_text = "The plaintiff alleges breach of contract on March 1, 2025.";
    let artifact = builder
        .build_extraction_prompt(&schema, doc_text, None, None, None, None)
        .unwrap();
    assert!(
        artifact.prompt_text.contains(doc_text),
        "Prompt should contain the document text verbatim"
    );
}

#[test]
fn test_extraction_prompt_with_rules_file() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(
            &schema,
            "Document text here.",
            None,
            None,
            Some("global_rules.md"),
            None,
        )
        .unwrap();
    assert!(
        artifact.prompt_text.contains("verbatim_quote"),
        "Prompt should include content from global_rules.md"
    );
    assert!(
        artifact.prompt_text.contains("Do not paraphrase"),
        "Prompt should include rules content"
    );
}

#[test]
fn test_extraction_prompt_with_context() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(
            &schema,
            "Document text.",
            Some("Previously processed: Motion to Dismiss"),
            None,
            None,
            None,
        )
        .unwrap();
    assert!(
        artifact.prompt_text.contains("Motion to Dismiss"),
        "Prompt should contain the context"
    );
}

#[test]
fn test_extraction_prompt_none_context_gets_default() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Document text.", None, None, None, None)
        .unwrap();
    assert!(
        artifact
            .prompt_text
            .contains("None — this is the first document"),
        "Prompt should contain default context when None"
    );
}

#[test]
fn test_extraction_prompt_none_admin_instructions_gets_default() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Document text.", None, None, None, None)
        .unwrap();
    assert!(
        artifact.prompt_text.contains("None."),
        "Prompt should contain default admin instructions when None"
    );
}

#[test]
fn test_extraction_prompt_with_admin_instructions() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(
            &schema,
            "Document text.",
            None,
            Some("Focus on damages calculations."),
            None,
            None,
        )
        .unwrap();
    assert!(
        artifact
            .prompt_text
            .contains("Focus on damages calculations"),
        "Prompt should contain admin instructions"
    );
}

#[test]
fn test_missing_template_returns_error() {
    let schema = load_complaint_schema();
    // Point to a directory that doesn't have pass1_template.md
    let mut builder = PromptBuilder::new(&fixtures_dir().join("schemas"));
    let result = builder.build_extraction_prompt(&schema, "Text.", None, None, None, None);
    assert!(result.is_err(), "Should fail when template file is missing");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Template error"),
        "Error should be a template error: {msg}"
    );
}

#[test]
fn test_v2_extraction_prompt_contains_grounding_mode() {
    let schema_path = fixtures_dir().join("schemas/complaint_v2.yaml");
    let schema = ExtractionSchema::from_file(&schema_path).expect("v2 schema should load");
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Sample document text.", None, None, None, None)
        .expect("should build prompt");
    assert!(
        artifact.prompt_text.contains("name_match"),
        "Prompt should contain 'name_match' from Party's grounding_mode"
    );
}

#[test]
fn test_prompt_contains_pass1_structure() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Sample text.", None, None, None, None)
        .unwrap();
    // Verify template structure is present
    assert!(
        artifact.prompt_text.contains("# Document extraction"),
        "Should have heading"
    );
    assert!(
        artifact.prompt_text.contains("## Schema"),
        "Should have Schema section"
    );
    assert!(
        artifact.prompt_text.contains("## Output format"),
        "Should have Output section"
    );
}

#[test]
fn test_extraction_prompt_with_custom_template() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(
            &schema,
            "Document text.",
            None,
            None,
            None,
            Some("custom_template.md"),
        )
        .expect("should build with custom template");
    assert!(
        artifact.prompt_text.starts_with("CUSTOM"),
        "Prompt should start with CUSTOM: got '{}'",
        &artifact.prompt_text[..20.min(artifact.prompt_text.len())]
    );
    assert!(
        artifact.prompt_text.ends_with("END"),
        "Prompt should end with END"
    );
}

// ---------------------------------------------------------------------------
// PromptArtifact hash and metadata tests
// ---------------------------------------------------------------------------

/// Helper: returns true if s is a 64-character lowercase hex string.
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

#[test]
fn test_prompt_artifact_has_template_hash() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Text.", None, None, None, None)
        .unwrap();
    assert!(
        is_sha256_hex(&artifact.template_hash),
        "template_hash should be 64 hex chars, got: {}",
        artifact.template_hash
    );
}

#[test]
fn test_prompt_artifact_has_schema_hash() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Text.", None, None, None, None)
        .unwrap();
    assert!(
        is_sha256_hex(&artifact.schema_hash),
        "schema_hash should be 64 hex chars, got: {}",
        artifact.schema_hash
    );
}

#[test]
fn test_prompt_artifact_has_rules_hash() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Text.", None, None, Some("global_rules.md"), None)
        .unwrap();
    assert_eq!(artifact.rules_name.as_deref(), Some("global_rules.md"));
    let hash = artifact
        .rules_hash
        .as_ref()
        .expect("rules_hash should be Some");
    assert!(
        is_sha256_hex(hash),
        "rules_hash should be 64 hex chars, got: {hash}"
    );
}

#[test]
fn test_prompt_artifact_no_rules_hash_when_none() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Text.", None, None, None, None)
        .unwrap();
    assert!(artifact.rules_name.is_none(), "rules_name should be None");
    assert!(artifact.rules_hash.is_none(), "rules_hash should be None");
}

#[test]
fn test_prompt_artifact_template_name_default() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(&schema, "Text.", None, None, None, None)
        .unwrap();
    assert_eq!(artifact.template_name, "pass1_template.md");
}

#[test]
fn test_prompt_artifact_template_name_custom() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let artifact = builder
        .build_extraction_prompt(
            &schema,
            "Text.",
            None,
            None,
            None,
            Some("custom_template.md"),
        )
        .unwrap();
    assert_eq!(artifact.template_name, "custom_template.md");
}

#[test]
fn test_prompt_artifact_hash_deterministic() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let a1 = builder
        .build_extraction_prompt(
            &schema,
            "Same text.",
            None,
            None,
            Some("global_rules.md"),
            None,
        )
        .unwrap();
    let a2 = builder
        .build_extraction_prompt(
            &schema,
            "Same text.",
            None,
            None,
            Some("global_rules.md"),
            None,
        )
        .unwrap();
    assert_eq!(
        a1.template_hash, a2.template_hash,
        "template hashes should match"
    );
    assert_eq!(a1.schema_hash, a2.schema_hash, "schema hashes should match");
    assert_eq!(a1.rules_hash, a2.rules_hash, "rules hashes should match");
}
