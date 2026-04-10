//! Tests for PromptBuilder — template loading and variable substitution.

use std::path::PathBuf;

use colossus_extract::{ExtractionSchema, PromptBuilder};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
}

fn template_dir() -> PathBuf {
    fixtures_dir().join("templates")
}

fn load_complaint_schema() -> ExtractionSchema {
    let schema_path = fixtures_dir().join("schemas/complaint.yaml");
    ExtractionSchema::from_file(&schema_path)
        .expect("complaint schema fixture should load")
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
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Sample document text.",
        None,
        None,
        None,
        None,
    ).expect("should build prompt");
    assert!(!prompt.is_empty(), "Prompt should not be empty");
}

#[test]
fn test_extraction_prompt_contains_schema_json() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Sample document text.",
        None,
        None,
        None,
        None,
    ).unwrap();
    // The schema JSON should contain the document type
    assert!(prompt.contains("\"document_type\": \"complaint\""),
        "Prompt should contain schema JSON with document_type");
    assert!(prompt.contains("\"entity_types\""),
        "Prompt should contain entity_types from schema");
}

#[test]
fn test_extraction_prompt_contains_document_text() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let doc_text = "The plaintiff alleges breach of contract on March 1, 2025.";
    let prompt = builder.build_extraction_prompt(
        &schema,
        doc_text,
        None,
        None,
        None,
        None,
    ).unwrap();
    assert!(prompt.contains(doc_text),
        "Prompt should contain the document text verbatim");
}

#[test]
fn test_extraction_prompt_with_rules_file() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Document text here.",
        None,
        None,
        Some("global_rules.md"),
        None,
    ).unwrap();
    assert!(prompt.contains("verbatim_quote"),
        "Prompt should include content from global_rules.md");
    assert!(prompt.contains("Do not paraphrase"),
        "Prompt should include rules content");
}

#[test]
fn test_extraction_prompt_with_context() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Document text.",
        Some("Previously processed: Motion to Dismiss"),
        None,
        None,
        None,
    ).unwrap();
    assert!(prompt.contains("Motion to Dismiss"),
        "Prompt should contain the context");
}

#[test]
fn test_extraction_prompt_none_context_gets_default() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Document text.",
        None,
        None,
        None,
        None,
    ).unwrap();
    assert!(prompt.contains("None — this is the first document"),
        "Prompt should contain default context when None");
}

#[test]
fn test_extraction_prompt_none_admin_instructions_gets_default() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Document text.",
        None,
        None,
        None,
        None,
    ).unwrap();
    assert!(prompt.contains("None."),
        "Prompt should contain default admin instructions when None");
}

#[test]
fn test_extraction_prompt_with_admin_instructions() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Document text.",
        None,
        Some("Focus on damages calculations."),
        None,
        None,
    ).unwrap();
    assert!(prompt.contains("Focus on damages calculations"),
        "Prompt should contain admin instructions");
}

#[test]
fn test_missing_template_returns_error() {
    let schema = load_complaint_schema();
    // Point to a directory that doesn't have pass1_template.md
    let mut builder = PromptBuilder::new(&fixtures_dir().join("schemas"));
    let result = builder.build_extraction_prompt(
        &schema,
        "Text.",
        None,
        None,
        None,
        None,
    );
    assert!(result.is_err(), "Should fail when template file is missing");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Template error"),
        "Error should be a template error: {msg}");
}

#[test]
fn test_v2_extraction_prompt_contains_grounding_mode() {
    let schema_path = fixtures_dir().join("schemas/complaint_v2.yaml");
    let schema = ExtractionSchema::from_file(&schema_path)
        .expect("v2 schema should load");
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Sample document text.",
        None,
        None,
        None,
        None,
    ).expect("should build prompt");
    assert!(prompt.contains("name_match"),
        "Prompt should contain 'name_match' from Party's grounding_mode");
}

#[test]
fn test_prompt_contains_pass1_structure() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Sample text.",
        None,
        None,
        None,
        None,
    ).unwrap();
    // Verify template structure is present
    assert!(prompt.contains("# Document extraction"), "Should have heading");
    assert!(prompt.contains("## Schema"), "Should have Schema section");
    assert!(prompt.contains("## Output format"), "Should have Output section");
}

#[test]
fn test_extraction_prompt_with_custom_template() {
    let schema = load_complaint_schema();
    let mut builder = PromptBuilder::new(&template_dir());
    let prompt = builder.build_extraction_prompt(
        &schema,
        "Document text.",
        None,
        None,
        None,
        Some("custom_template.md"),
    ).expect("should build with custom template");
    assert!(prompt.starts_with("CUSTOM"),
        "Prompt should start with CUSTOM: got '{}'", &prompt[..20.min(prompt.len())]);
    assert!(prompt.ends_with("END"),
        "Prompt should end with END");
}
