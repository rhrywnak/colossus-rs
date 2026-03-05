//! Tests for LegalAssembler — all unit tests (no infrastructure needed).
//!
//! The assembler is synchronous and does purely in-memory work, so every
//! test here runs without any external services (no Qdrant, no Neo4j, no API keys).
//!
//! ## Test organization
//!
//! - Tests 1–3: `format_chunk()` — formatting individual chunks
//! - Tests 4–5: `assemble()` — full assembly with empty/truncation scenarios
//! - Test 6: `estimate_tokens()` — token estimation accuracy
//! - Tests 7–8: System prompt configuration

use colossus_rag::{
    estimate_tokens, format_chunk, ContextAssembler, ContextChunk,
    LegalAssembler, RelatedNode, RelationDirection, SourceReference,
};

// ===========================================================================
// Test helpers
// ===========================================================================

/// Create a fully-populated ContextChunk for testing.
///
/// Uses data modeled on the Awad v. CFS case (the real case in colossus-legal).
fn make_full_chunk() -> ContextChunk {
    ContextChunk {
        node_id: "evidence-phillips-q74".to_string(),
        node_type: "Evidence".to_string(),
        title: "Phillips: Emil wanted $50K returned".to_string(),
        content: "Q: Did Emil ever ask for the money back? A: Yes, multiple times he asked for the $50,000 to be returned.".to_string(),
        score: 0.7056,
        source: SourceReference {
            document_title: Some("Phillips Deposition".to_string()),
            document_id: Some("dep-phillips".to_string()),
            page_number: Some(42),
            verbatim_quote: Some("Yes, multiple times he asked for the $50,000 to be returned.".to_string()),
        },
        relationships: Vec::new(),
        metadata: serde_json::Value::Null,
    }
}

/// Create a chunk with relationships for graph expansion testing.
fn make_chunk_with_relationships() -> ContextChunk {
    let mut chunk = make_full_chunk();
    chunk.relationships = vec![
        RelatedNode {
            node_id: "harm-003".to_string(),
            node_type: "Harm".to_string(),
            relationship: "SUPPORTS".to_string(),
            direction: RelationDirection::Outbound,
            summary: "Financial harm from unreturned funds".to_string(),
        },
        RelatedNode {
            node_id: "claim-007".to_string(),
            node_type: "MotionClaim".to_string(),
            relationship: "CITED_IN".to_string(),
            direction: RelationDirection::Inbound,
            summary: "Referenced in breach of fiduciary duty claim".to_string(),
        },
    ];
    chunk
}

/// Create a minimal chunk with only required fields populated.
fn make_minimal_chunk() -> ContextChunk {
    ContextChunk {
        node_id: "node-001".to_string(),
        node_type: "MotionClaim".to_string(),
        title: "Breach of fiduciary duty".to_string(),
        content: "Breach of fiduciary duty".to_string(), // Same as title — should be omitted
        score: 0.5,
        source: SourceReference::default(), // All None
        relationships: Vec::new(),
        metadata: serde_json::Value::Null,
    }
}

// ===========================================================================
// Test 1: Format a single chunk with all fields populated
// ===========================================================================

/// Verify that `format_chunk()` produces the expected format with all fields.
///
/// The format must include:
/// - `=== TYPE: ID ===` header (type uppercased)
/// - Title, Content, Source, Quote, Score lines
#[test]
fn test_format_single_chunk_with_all_fields() {
    let chunk = make_full_chunk();
    let output = format_chunk(&chunk);

    // Header: type is uppercased, ID is exact.
    assert!(
        output.contains("=== EVIDENCE: evidence-phillips-q74 ==="),
        "Should have header with type and ID. Got:\n{output}"
    );

    // Title line.
    assert!(
        output.contains("Title: Phillips: Emil wanted $50K returned"),
        "Should have title line"
    );

    // Content line (different from title, so it should appear).
    assert!(
        output.contains("Content: Q: Did Emil ever ask for the money back?"),
        "Should have content line when content differs from title"
    );

    // Source line (document + page).
    assert!(
        output.contains("Source: Phillips Deposition, p.42"),
        "Should have source line with document and page"
    );

    // Verbatim quote.
    assert!(
        output.contains("Quote: \"Yes, multiple times"),
        "Should have verbatim quote"
    );

    // Score (4 decimal places).
    assert!(
        output.contains("Score: 0.7056"),
        "Should have score with 4 decimal places"
    );
}

// ===========================================================================
// Test 2: Format a chunk with graph relationships
// ===========================================================================

/// Verify that relationships are formatted with direction arrows.
#[test]
fn test_format_chunk_with_relationships() {
    let chunk = make_chunk_with_relationships();
    let output = format_chunk(&chunk);

    // Outbound relationship: → arrow.
    assert!(
        output.contains("Related: → SUPPORTS harm-003 (Harm)"),
        "Should show outbound relationship with → arrow. Got:\n{output}"
    );

    // Inbound relationship: ← arrow.
    assert!(
        output.contains("Related: ← CITED_IN claim-007 (MotionClaim)"),
        "Should show inbound relationship with ← arrow. Got:\n{output}"
    );
}

// ===========================================================================
// Test 3: Format a chunk with missing optional fields
// ===========================================================================

/// When optional fields are None/empty, they should be omitted from output.
///
/// - Content same as title → Content line omitted
/// - No document_title, no page_number → Source line omitted
/// - No verbatim_quote → Quote line omitted
/// - No relationships → Related lines omitted
#[test]
fn test_format_chunk_missing_optional_fields() {
    let chunk = make_minimal_chunk();
    let output = format_chunk(&chunk);

    // Should have header and title.
    assert!(output.contains("=== MOTIONCLAIM: node-001 ==="));
    assert!(output.contains("Title: Breach of fiduciary duty"));

    // Content equals title → should NOT have a Content line.
    assert!(
        !output.contains("Content:"),
        "Content line should be omitted when content equals title"
    );

    // No source info → should NOT have a Source line.
    assert!(
        !output.contains("Source:"),
        "Source line should be omitted when no source info"
    );

    // No quote → should NOT have a Quote line.
    assert!(
        !output.contains("Quote:"),
        "Quote line should be omitted when no verbatim quote"
    );

    // No relationships → should NOT have Related lines.
    assert!(
        !output.contains("Related:"),
        "Related lines should be omitted when no relationships"
    );

    // Should still have score.
    assert!(output.contains("Score: 0.5000"));
}

// ===========================================================================
// Test 4: Empty chunks produce a no-evidence prompt
// ===========================================================================

/// When no chunks are provided, the assembler should append a "no evidence found"
/// message to the system prompt, telling Claude to inform the user.
#[test]
fn test_empty_chunks_produces_no_evidence_prompt() {
    let assembler = LegalAssembler::new();
    let result = assembler.assemble("What happened?", &[], 8000);

    // System prompt should still contain the base instructions.
    assert!(
        result.system_prompt.contains("legal research assistant"),
        "Should contain base system prompt"
    );

    // Should have the no-evidence message appended.
    assert!(
        result.system_prompt.contains("No relevant evidence was found"),
        "Should append no-evidence message. Got:\n{}",
        result.system_prompt
    );

    // Formatted context should be empty (no chunks to format).
    assert!(
        result.formatted_context.is_empty(),
        "Formatted context should be empty when no chunks"
    );

    // Token estimate should be non-zero (the system prompt itself has tokens).
    assert!(
        result.token_estimate > 0,
        "Token estimate should be non-zero even with no chunks"
    );
}

// ===========================================================================
// Test 5: Max tokens truncation drops lowest-scored chunks
// ===========================================================================

/// When the total context exceeds max_tokens, the assembler should drop
/// the lowest-scored chunks (they're sorted by score descending).
#[test]
fn test_max_tokens_truncation_drops_lowest_scored() {
    let assembler = LegalAssembler::new();

    // Create 3 chunks with different scores.
    let mut high = make_full_chunk();
    high.node_id = "high-score".to_string();
    high.score = 0.9;

    let mut medium = make_full_chunk();
    medium.node_id = "medium-score".to_string();
    medium.score = 0.6;

    let mut low = make_full_chunk();
    low.node_id = "low-score".to_string();
    low.score = 0.3;

    let chunks = vec![low, high.clone(), medium];

    // Use a generous budget first — all 3 should appear.
    let result_all = assembler.assemble("test?", &chunks, 100_000);
    assert!(
        result_all.formatted_context.contains("high-score"),
        "High-scored chunk should appear"
    );
    assert!(
        result_all.formatted_context.contains("medium-score"),
        "Medium-scored chunk should appear"
    );
    assert!(
        result_all.formatted_context.contains("low-score"),
        "Low-scored chunk should appear"
    );

    // Now use a very tight budget — only the system prompt + maybe 1 chunk.
    // The system prompt alone is ~500 chars ≈ 125 tokens.
    // Each chunk is ~200-300 chars ≈ 50-75 tokens.
    // Set budget to allow system prompt + 1 chunk only.
    let system_tokens = estimate_tokens(&result_all.system_prompt);
    let chunk_tokens = estimate_tokens(&format_chunk(&high));
    let tight_budget = system_tokens + chunk_tokens + 10; // Just enough for 1 chunk

    let result_tight = assembler.assemble("test?", &chunks, tight_budget);

    // High-scored chunk should always appear (it's first after sorting).
    assert!(
        result_tight.formatted_context.contains("high-score"),
        "Highest-scored chunk should survive truncation"
    );

    // Low-scored chunk should be dropped.
    assert!(
        !result_tight.formatted_context.contains("low-score"),
        "Lowest-scored chunk should be dropped when over budget"
    );
}

// ===========================================================================
// Test 6: Token estimation is in a reasonable range
// ===========================================================================

/// The chars/4 approximation should give reasonable estimates for English text.
///
/// We don't need exactness — just sanity checking the order of magnitude.
#[test]
fn test_token_estimation_reasonable_range() {
    // Empty string → 0 tokens.
    assert_eq!(estimate_tokens(""), 0);

    // 4 characters → 1 token (exactly at the ratio).
    assert_eq!(estimate_tokens("abcd"), 1);

    // 100 characters → ~25 tokens.
    let text_100 = "a".repeat(100);
    assert_eq!(estimate_tokens(&text_100), 25);

    // A realistic prompt (~1000 chars) → ~250 tokens.
    let realistic_prompt = "You are a legal research assistant analyzing case evidence. ".repeat(17);
    let tokens = estimate_tokens(&realistic_prompt);
    assert!(
        (200..=300).contains(&tokens),
        "~1000 char prompt should estimate 200-300 tokens, got {tokens}"
    );
}

// ===========================================================================
// Test 7: Default system prompt includes citation instructions
// ===========================================================================

/// The default prompt must include the key legal analysis rules:
/// evidence-only answers, citation by ID, contradiction handling, honest uncertainty.
#[test]
fn test_default_system_prompt_includes_citation_instructions() {
    let assembler = LegalAssembler::new();
    let chunks = vec![make_full_chunk()];
    let result = assembler.assemble("test?", &chunks, 100_000);

    // Rule 1: Evidence-only.
    assert!(
        result.system_prompt.contains("ONLY the provided evidence"),
        "Should instruct evidence-only answers"
    );

    // Rule 2: Citation by ID.
    assert!(
        result.system_prompt.contains("cite the specific evidence ID"),
        "Should require citation by evidence ID"
    );

    // Rule 3: Contradictions.
    assert!(
        result.system_prompt.contains("contradict"),
        "Should handle contradictions"
    );

    // Rule 4: Honest uncertainty.
    assert!(
        result.system_prompt.contains("does not contain enough information"),
        "Should instruct honest uncertainty"
    );
}

// ===========================================================================
// Test 8: Custom system prompt template
// ===========================================================================

/// `LegalAssembler::with_system_prompt()` should use the custom prompt
/// instead of the default.
#[test]
fn test_custom_system_prompt_template() {
    let custom_prompt = "You are a medical expert. Answer using only the provided clinical data.";
    let assembler = LegalAssembler::with_system_prompt(custom_prompt);

    let chunks = vec![make_full_chunk()];
    let result = assembler.assemble("test?", &chunks, 100_000);

    // Should use our custom prompt, not the default.
    assert!(
        result.system_prompt.contains("medical expert"),
        "Should use custom system prompt"
    );

    // Should NOT contain the default legal prompt.
    assert!(
        !result.system_prompt.contains("legal research assistant"),
        "Should not contain default prompt when custom is provided"
    );

    // Formatted context should still work normally.
    assert!(
        result.formatted_context.contains("evidence-phillips-q74"),
        "Chunks should still be formatted with custom prompt"
    );
}
