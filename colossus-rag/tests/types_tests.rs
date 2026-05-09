//! Integration tests for colossus-rag types, errors, and no-op implementations.
//!
//! These tests verify:
//! 1. All types serialize/deserialize correctly (serde round-trips)
//! 2. Default implementations produce expected values
//! 3. Error Display messages are meaningful
//! 4. No-op implementations satisfy their trait contracts

use colossus_rag::{
    Citation, ContextChunk, RagError, RelatedNode, RelationDirection, RetrievalStrategy,
    ScopeFilter, ScopeFilterType, SourceReference,
};

// ---------------------------------------------------------------------------
// Test 1: RetrievalStrategy serde round-trip
// ---------------------------------------------------------------------------

/// Verify each RetrievalStrategy variant survives JSON serialization
/// and deserialization without data loss.
///
/// ## Rust Learning: Serde round-trip testing
///
/// A "round-trip" test serializes a value to JSON and deserializes it back,
/// then asserts the result equals the original. This catches:
/// - Missing `#[serde(...)]` attributes
/// - Incorrect rename rules
/// - Fields that serialize but don't deserialize (or vice versa)
#[test]
fn test_retrieval_strategy_serde_roundtrip() {
    let strategies = vec![
        RetrievalStrategy::Focused {
            scope: vec![ScopeFilter {
                filter_type: ScopeFilterType::Person,
                value: "Phillips".to_string(),
            }],
        },
        RetrievalStrategy::Broad {
            node_types: Some(vec!["Evidence".to_string(), "MotionClaim".to_string()]),
        },
        RetrievalStrategy::Broad { node_types: None },
        RetrievalStrategy::Hybrid {
            scopes: vec![
                ScopeFilter {
                    filter_type: ScopeFilterType::Person,
                    value: "Phillips".to_string(),
                },
                ScopeFilter {
                    filter_type: ScopeFilterType::Document,
                    value: "motion-001".to_string(),
                },
            ],
            synthesize_across: true,
        },
        RetrievalStrategy::Direct {
            query_hint: "List all exhibits".to_string(),
        },
    ];

    for strategy in &strategies {
        let json = serde_json::to_string(strategy)
            .expect("RetrievalStrategy should serialize");
        let deserialized: RetrievalStrategy = serde_json::from_str(&json)
            .expect("RetrievalStrategy should deserialize");
        assert_eq!(
            strategy, &deserialized,
            "Round-trip failed for: {json}"
        );
    }

    // Also verify the JSON format uses our adjacently-tagged representation.
    let focused = &strategies[0];
    let json = serde_json::to_value(focused).expect("should serialize to Value");
    assert_eq!(json["type"], "focused", "Should use snake_case tag");
    assert!(
        json["params"]["scope"].is_array(),
        "Focused params should contain scope array"
    );
}

// ---------------------------------------------------------------------------
// Test 2: ContextChunk with relationships serializes correctly
// ---------------------------------------------------------------------------

#[test]
fn test_context_chunk_with_relationships_serializes() {
    let chunk = ContextChunk {
        node_id: "evidence-phillips-q74".to_string(),
        node_type: "Evidence".to_string(),
        title: "Phillips: Emil wanted $50K returned".to_string(),
        content: "Q: Did Emil ever ask for the money back? A: Yes.".to_string(),
        score: 0.7056,
        source: SourceReference {
            document_title: Some("Phillips Deposition".to_string()),
            document_id: Some("dep-phillips".to_string()),
            page_number: Some(42),
            verbatim_quote: None,
        },
        relationships: vec![RelatedNode {
            node_id: "harm-003".to_string(),
            node_type: "Harm".to_string(),
            relationship: "SUPPORTS".to_string(),
            direction: RelationDirection::Outbound,
            summary: "Evidence supports unnecessary auction loss harm".to_string(),
        }],
        metadata: serde_json::Value::Null,
    };

    // Serialize and check structure.
    let json = serde_json::to_value(&chunk).expect("ContextChunk should serialize");
    assert_eq!(json["node_id"], "evidence-phillips-q74");
    assert_eq!(json["relationships"][0]["direction"], "outbound");
    assert_eq!(json["source"]["page_number"], 42);

    // Verify verbatim_quote (None) is omitted due to skip_serializing_if.
    assert!(
        json["source"].get("verbatim_quote").is_none(),
        "None fields with skip_serializing_if should be omitted"
    );

    // Verify metadata (Null) is omitted due to skip_serializing_if.
    assert!(
        json.get("metadata").is_none(),
        "Null metadata should be omitted from JSON"
    );

    // Round-trip.
    let json_string = serde_json::to_string(&chunk).expect("should serialize to string");
    let deserialized: ContextChunk =
        serde_json::from_str(&json_string).expect("should deserialize");
    assert_eq!(chunk, deserialized);
}

// ---------------------------------------------------------------------------
// Test 3: Citation with all None fields
// ---------------------------------------------------------------------------

#[test]
fn test_citation_all_none_fields() {
    let citation = Citation::default();

    assert_eq!(citation.evidence_id, None);
    assert_eq!(citation.document, None);
    assert_eq!(citation.page, None);
    assert_eq!(citation.quote_excerpt, None);

    // Verify all None fields are omitted from JSON.
    let json = serde_json::to_value(&citation).expect("Citation should serialize");
    assert_eq!(
        json,
        serde_json::json!({}),
        "Citation with all None fields should serialize to empty object"
    );

    // Also test a Citation with some fields set.
    let citation_with_data = Citation {
        evidence_id: Some("evidence-phillips-q74".to_string()),
        document: Some("Phillips Deposition".to_string()),
        page: Some(42),
        quote_excerpt: None,
    };
    let json = serde_json::to_value(&citation_with_data).expect("should serialize");
    assert_eq!(json["evidence_id"], "evidence-phillips-q74");
    assert!(json.get("quote_excerpt").is_none(), "None quote_excerpt should be omitted");
}

// ---------------------------------------------------------------------------
// Test 5: RagError Display messages are meaningful
// ---------------------------------------------------------------------------

#[test]
fn test_rag_error_display_messages() {
    // Each variant should produce a descriptive message that includes
    // the stage name and the error detail.
    let errors = vec![
        (
            RagError::InvalidInput("empty question".to_string()),
            "Invalid input: empty question",
        ),
        (
            RagError::EmbeddingError("ONNX model not found".to_string()),
            "Embedding error: ONNX model not found",
        ),
        (
            RagError::SearchError("connection refused".to_string()),
            "Search error: connection refused",
        ),
        (
            RagError::ExpandError("Neo4j timeout".to_string()),
            "Expand error: Neo4j timeout",
        ),
        (
            RagError::AssemblyError("token limit exceeded".to_string()),
            "Assembly error: token limit exceeded",
        ),
        (
            RagError::SynthesisError("API 429 rate limited".to_string()),
            "Synthesis error: API 429 rate limited",
        ),
        (
            RagError::ConfigError("ANTHROPIC_API_KEY not set".to_string()),
            "Config error: ANTHROPIC_API_KEY not set",
        ),
    ];

    for (error, expected_display) in errors {
        let display = format!("{error}");
        assert_eq!(
            display, expected_display,
            "Display for {:?} should be '{expected_display}'",
            error
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: ScopeFilterType serde
// ---------------------------------------------------------------------------

#[test]
fn test_scope_filter_type_serde() {
    let types = vec![
        (ScopeFilterType::Document, "\"document\""),
        (ScopeFilterType::Person, "\"person\""),
        (ScopeFilterType::NodeType, "\"node_type\""),
        (ScopeFilterType::Collection, "\"collection\""),
    ];

    for (variant, expected_json) in types {
        let json = serde_json::to_string(&variant).expect("should serialize");
        assert_eq!(json, expected_json, "ScopeFilterType serde mismatch");
    }
}

// ---------------------------------------------------------------------------
// Test 5: SubQuery serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_sub_query_serde_roundtrip() {
    use colossus_rag::SubQuery;

    let queries = vec![
        SubQuery::VectorSearch {
            query: "test query".to_string(),
        },
        SubQuery::GraphDocumentContent {
            document_id: "doc-phillips-coa-response-300891".to_string(),
            description: "Phillips CoA evidence".to_string(),
        },
        SubQuery::GraphPersonStatements {
            person_id: "george-phillips".to_string(),
            description: "Phillips statements".to_string(),
        },
        SubQuery::GraphContradictions {
            person_name: "Phillips".to_string(),
            description: "Phillips contradictions".to_string(),
        },
    ];

    for sq in &queries {
        let json = serde_json::to_string(sq).expect("should serialize");
        let deserialized: SubQuery = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(sq, &deserialized, "Round-trip failed for: {json}");
    }

    // Verify tagged format.
    let vs = &queries[0];
    let json = serde_json::to_value(vs).expect("should serialize");
    assert_eq!(json["type"], "vector_search");
}
