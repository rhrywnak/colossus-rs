//! Integration tests for colossus-rag types, errors, and no-op implementations.
//!
//! These tests verify:
//! 1. All types serialize/deserialize correctly (serde round-trips)
//! 2. Default implementations produce expected values
//! 3. Error Display messages are meaningful
//! 4. No-op implementations satisfy their trait contracts

use colossus_rag::{
    AssembledContext, Citation, ContextChunk, NoOpExpander, NoOpRouter, PipelineStats, RagError,
    RagResult, RelatedNode, RelationDirection, RetrievalStrategy, ScopeFilter, ScopeFilterType,
    SourceReference, SynthesisResult,
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
// Test 3: PipelineStats default has all zeros
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_stats_default_all_zeros() {
    let stats = PipelineStats::default();

    assert_eq!(stats.route_ms, 0);
    assert_eq!(stats.embed_ms, 0);
    assert_eq!(stats.search_ms, 0);
    assert_eq!(stats.expand_ms, 0);
    assert_eq!(stats.assemble_ms, 0);
    assert_eq!(stats.synthesize_ms, 0);
    assert_eq!(stats.total_ms, 0);
    assert_eq!(stats.qdrant_hits, 0);
    assert_eq!(stats.graph_nodes_expanded, 0);
    assert_eq!(stats.context_tokens_approx, 0);
    assert_eq!(stats.input_tokens, None);
    assert_eq!(stats.output_tokens, None);
    assert!(stats.strategy.is_empty());
    assert!(stats.provider.is_empty());
    assert!(stats.model.is_empty());
}

// ---------------------------------------------------------------------------
// Test 4: Citation with all None fields
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
// Test 6: NoOpExpander returns empty vec
// ---------------------------------------------------------------------------

/// ## Rust Learning: Testing async functions
///
/// `#[tokio::test]` creates a tokio runtime for the test. Without it,
/// we can't `.await` the async `expand()` method.
#[tokio::test]
async fn test_noop_expander_returns_empty() {
    // Must import the trait to call its methods on the concrete type.
    use colossus_rag::GraphExpander;

    let expander = NoOpExpander;
    let seed_ids = vec!["node-1".to_string(), "node-2".to_string()];

    let result = expander
        .expand(&seed_ids, 2)
        .await
        .expect("NoOpExpander should never fail");

    assert!(
        result.is_empty(),
        "NoOpExpander should return empty vec, got {} chunks",
        result.len()
    );
}

// ---------------------------------------------------------------------------
// Test 7: NoOpRouter returns Broad strategy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_noop_router_returns_broad() {
    use colossus_rag::QueryRouter;

    let router = NoOpRouter;

    let strategy = router
        .route("What did Phillips say about the $50,000?")
        .await
        .expect("NoOpRouter should never fail");

    // Verify it returns Broad with no type filters.
    match strategy {
        RetrievalStrategy::Broad { node_types } => {
            assert_eq!(
                node_types, None,
                "NoOpRouter should return Broad with no type filters"
            );
        }
        other => {
            panic!("NoOpRouter should return Broad, got: {other:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 8: Full RagResult round-trip
// ---------------------------------------------------------------------------

/// Verify a complete RagResult with all nested types survives a round-trip.
/// This exercises the full type hierarchy at once.
#[test]
fn test_rag_result_full_roundtrip() {
    let result = RagResult {
        answer: "Phillips testified that Emil wanted the $50,000 returned.".to_string(),
        strategy_used: RetrievalStrategy::Focused {
            scope: vec![ScopeFilter {
                filter_type: ScopeFilterType::Person,
                value: "Phillips".to_string(),
            }],
        },
        chunks: vec![ContextChunk {
            node_id: "evidence-phillips-q74".to_string(),
            node_type: "Evidence".to_string(),
            title: "Phillips: Emil wanted $50K returned".to_string(),
            content: "Test content".to_string(),
            score: 0.7056,
            source: SourceReference::default(),
            relationships: Vec::new(),
            metadata: serde_json::Value::Null,
        }],
        citations: vec![Citation {
            evidence_id: Some("evidence-phillips-q74".to_string()),
            document: None,
            page: None,
            quote_excerpt: Some("Emil wanted the money returned".to_string()),
        }],
        stats: PipelineStats {
            strategy: "focused".to_string(),
            total_ms: 1234,
            qdrant_hits: 5,
            provider: "anthropic".to_string(),
            model: "claude-haiku-4-5-20251001".to_string(),
            ..PipelineStats::default()
        },
    };

    let json = serde_json::to_string_pretty(&result).expect("RagResult should serialize");
    let deserialized: RagResult =
        serde_json::from_str(&json).expect("RagResult should deserialize");
    assert_eq!(result, deserialized, "Full RagResult round-trip failed");
}

// ---------------------------------------------------------------------------
// Test 9: ScopeFilterType serde
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
// Test 10: AssembledContext and SynthesisResult round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_assembled_context_default() {
    let ctx = AssembledContext::default();
    assert!(ctx.system_prompt.is_empty());
    assert!(ctx.formatted_context.is_empty());
    assert_eq!(ctx.token_estimate, 0);
}

#[test]
fn test_synthesis_result_roundtrip() {
    let result = SynthesisResult {
        answer: "The answer is 42.".to_string(),
        citations: vec![Citation::default()],
        input_tokens: 100,
        output_tokens: 25,
        provider: "anthropic".to_string(),
        model: "claude-haiku-4-5-20251001".to_string(),
    };

    let json = serde_json::to_string(&result).expect("should serialize");
    let deserialized: SynthesisResult =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(result, deserialized);
}
