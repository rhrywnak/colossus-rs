//! Tests for RagPipeline — unit tests + integration tests.
//!
//! Unit tests use mock implementations of each trait to verify pipeline
//! wiring without any infrastructure. Integration tests (#[ignore]) need
//! Qdrant, Neo4j, and Claude API access.

use colossus_rag::{
    AssembledContext, Citation, ContextChunk, LegalAssembler, NoOpRouter,
    RagError, RagPipeline, RetrievalStrategy, RuleBasedRouter,
    ScopeFilter, SourceReference, SynthesisResult,
};

// ===========================================================================
// Mock implementations for unit tests
// ===========================================================================

// ## Rust Learning: Mock trait implementations
//
// For unit testing the pipeline wiring, we create simple mock implementations
// of each trait. These return canned data without calling any external services.
// This lets us test the pipeline orchestration logic in isolation.

use async_trait::async_trait;
use colossus_rag::{Synthesizer, VectorRetriever};

/// A mock retriever that returns a fixed set of chunks.
struct MockRetriever;

#[async_trait]
impl VectorRetriever for MockRetriever {
    async fn search(
        &self,
        _query: &str,
        _limit: usize,
        _filters: &[ScopeFilter],
    ) -> Result<Vec<ContextChunk>, RagError> {
        Ok(vec![ContextChunk {
            node_id: "evidence-mock-001".into(),
            node_type: "Evidence".into(),
            title: "Mock evidence".into(),
            content: "This is mock evidence content for testing.".into(),
            score: 0.85,
            source: SourceReference::default(),
            relationships: vec![],
            metadata: serde_json::Value::Null,
        }])
    }
}

/// A mock synthesizer that returns a canned answer.
struct MockSynthesizer;

#[async_trait]
impl Synthesizer for MockSynthesizer {
    async fn synthesize(
        &self,
        _context: &AssembledContext,
        _question: &str,
    ) -> Result<SynthesisResult, RagError> {
        Ok(SynthesisResult {
            answer: "Mock answer based on the evidence.".into(),
            citations: vec![Citation {
                evidence_id: Some("evidence-mock-001".into()),
                document: None,
                page: None,
                quote_excerpt: Some("mock evidence content".into()),
            }],
            input_tokens: 100,
            output_tokens: 50,
            provider: "mock".into(),
            model: "mock-model".into(),
        })
    }
}

// ===========================================================================
// Unit Test 1: Builder missing component returns error
// ===========================================================================

/// If a required component is missing, `build()` should return ConfigError.
#[test]
fn test_builder_missing_component_returns_error() {
    // Missing retriever — should fail.
    let result = RagPipeline::builder()
        .router(Box::new(NoOpRouter))
        .assembler(Box::new(LegalAssembler::new()))
        .synthesizer(Box::new(MockSynthesizer))
        .build();

    assert!(result.is_err(), "Should fail without retriever");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("retriever"),
        "Error should mention 'retriever': {err}"
    );

    // Missing router — should fail.
    let result = RagPipeline::builder()
        .retriever(Box::new(MockRetriever))
        .assembler(Box::new(LegalAssembler::new()))
        .synthesizer(Box::new(MockSynthesizer))
        .build();

    assert!(result.is_err(), "Should fail without router");

    // Missing assembler — should fail.
    let result = RagPipeline::builder()
        .router(Box::new(NoOpRouter))
        .retriever(Box::new(MockRetriever))
        .synthesizer(Box::new(MockSynthesizer))
        .build();

    assert!(result.is_err(), "Should fail without assembler");

    // Missing synthesizer — should fail.
    let result = RagPipeline::builder()
        .router(Box::new(NoOpRouter))
        .retriever(Box::new(MockRetriever))
        .assembler(Box::new(LegalAssembler::new()))
        .build();

    assert!(result.is_err(), "Should fail without synthesizer");
}

// ===========================================================================
// Unit Test 2: Builder with all components builds OK
// ===========================================================================

/// With all required components set, `build()` should succeed.
/// Note: expander is optional (defaults to NoOpExpander).
#[test]
fn test_builder_complete_builds_ok() {
    let result = RagPipeline::builder()
        .router(Box::new(NoOpRouter))
        .retriever(Box::new(MockRetriever))
        .assembler(Box::new(LegalAssembler::new()))
        .synthesizer(Box::new(MockSynthesizer))
        .build();

    assert!(result.is_ok(), "Should build successfully without expander (defaults to NoOp)");
}

// ===========================================================================
// Unit Test 3: Full pipeline ask() with mocks
// ===========================================================================

/// End-to-end test with all mock components — verifies the pipeline wiring.
///
/// ## Rust Learning: `#[tokio::test]`
///
/// Unlike the router tests (which used a manual `Runtime::new()`), here we
/// use `#[tokio::test]` directly. This macro creates a tokio runtime and
/// runs the test as an async function. It's cleaner when the test IS async.
#[tokio::test]
async fn test_pipeline_ask_with_mocks() {
    let pipeline = RagPipeline::builder()
        .router(Box::new(NoOpRouter))
        .retriever(Box::new(MockRetriever))
        .assembler(Box::new(LegalAssembler::new()))
        .synthesizer(Box::new(MockSynthesizer))
        .build()
        .expect("Should build");

    let result = pipeline
        .ask("What evidence supports breach of fiduciary duty?")
        .await
        .expect("Should not error");

    // Verify the result has all expected fields populated.
    assert!(
        !result.answer.is_empty(),
        "Answer should not be empty"
    );
    assert!(
        !result.chunks.is_empty(),
        "Should have at least one chunk"
    );
    assert_eq!(
        result.stats.qdrant_hits, 1,
        "MockRetriever returns 1 chunk"
    );
    assert_eq!(
        result.stats.provider, "mock",
        "Should use mock provider"
    );

    // Strategy should be Broad (NoOpRouter always returns Broad).
    match &result.strategy_used {
        RetrievalStrategy::Broad { .. } => {}
        other => panic!("Expected Broad strategy, got {other:?}"),
    }

    // Verify timing stats are populated.
    // (u64 is always >= 0, so we just check that total_ms makes sense.)
    assert!(result.stats.total_ms < 5000, "Total time should be reasonable for mocks");
}

// ===========================================================================
// Unit Test 4: Pipeline with custom search limit and context tokens
// ===========================================================================

/// Verify that builder configuration options are passed through correctly.
#[tokio::test]
async fn test_pipeline_custom_config() {
    let pipeline = RagPipeline::builder()
        .router(Box::new(NoOpRouter))
        .retriever(Box::new(MockRetriever))
        .assembler(Box::new(LegalAssembler::new()))
        .synthesizer(Box::new(MockSynthesizer))
        .max_context_tokens(3000)
        .search_limit(5)
        .build()
        .expect("Should build");

    // Pipeline should work with custom config.
    let result = pipeline.ask("Test question").await.expect("Should not error");
    assert!(!result.answer.is_empty());
}

// ===========================================================================
// Integration Test 1: Broad search (needs all infrastructure)
// ===========================================================================

/// Full end-to-end test with real infrastructure.
///
/// Requires: QDRANT_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ANTHROPIC_API_KEY
#[tokio::test]
#[ignore]
async fn test_pipeline_ask_broad() {
    // Build real components from environment.
    let pipeline = build_real_pipeline().await;

    let result = pipeline
        .ask("What evidence supports breach of fiduciary duty?")
        .await
        .expect("Pipeline ask should succeed");

    // Print full results for milestone verification.
    println!("\n=== PIPELINE INTEGRATION TEST: BROAD ===");
    println!("Question: What evidence supports breach of fiduciary duty?");
    println!("Strategy: {}", result.stats.strategy);
    println!("Answer:\n{}\n", result.answer);
    println!("Chunks retrieved: {}", result.stats.qdrant_hits);
    println!("Graph nodes expanded: {}", result.stats.graph_nodes_expanded);
    println!("Context tokens (approx): {}", result.stats.context_tokens_approx);
    println!("--- Timing ---");
    println!("  Route:      {} ms", result.stats.route_ms);
    println!("  Search:     {} ms", result.stats.search_ms);
    println!("  Expand:     {} ms", result.stats.expand_ms);
    println!("  Assemble:   {} ms", result.stats.assemble_ms);
    println!("  Synthesize: {} ms", result.stats.synthesize_ms);
    println!("  Total:      {} ms", result.stats.total_ms);
    println!("--- LLM ---");
    println!("  Provider: {}", result.stats.provider);
    println!("  Model:    {}", result.stats.model);
    println!(
        "  Input tokens:  {}",
        result.stats.input_tokens.unwrap_or(0)
    );
    println!(
        "  Output tokens: {}",
        result.stats.output_tokens.unwrap_or(0)
    );
    println!("Citations: {}", result.citations.len());
    for (i, cite) in result.citations.iter().enumerate() {
        println!("  [{i}] {cite:?}");
    }
    println!("=== END BROAD TEST ===\n");

    // Assertions.
    assert!(!result.answer.is_empty(), "Answer should not be empty");
    assert!(result.stats.qdrant_hits > 0, "Should have vector search hits");
    assert!(result.stats.total_ms > 0, "Total time should be > 0");
}

// ===========================================================================
// Integration Test 2: Focused search (needs all infrastructure)
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_pipeline_ask_focused() {
    let pipeline = build_real_pipeline().await;

    let result = pipeline
        .ask("What does Phillips' discovery response say about the check?")
        .await
        .expect("Pipeline ask should succeed");

    println!("\n=== PIPELINE INTEGRATION TEST: FOCUSED ===");
    println!("Question: What does Phillips' discovery response say about the check?");
    println!("Strategy: {}", result.stats.strategy);
    println!("Answer:\n{}\n", result.answer);
    println!("Chunks retrieved: {}", result.stats.qdrant_hits);
    println!("Graph nodes expanded: {}", result.stats.graph_nodes_expanded);
    println!("--- Timing ---");
    println!("  Route:      {} ms", result.stats.route_ms);
    println!("  Search:     {} ms", result.stats.search_ms);
    println!("  Expand:     {} ms", result.stats.expand_ms);
    println!("  Assemble:   {} ms", result.stats.assemble_ms);
    println!("  Synthesize: {} ms", result.stats.synthesize_ms);
    println!("  Total:      {} ms", result.stats.total_ms);
    println!("--- LLM ---");
    println!("  Provider: {}", result.stats.provider);
    println!("  Model:    {}", result.stats.model);
    println!("=== END FOCUSED TEST ===\n");

    assert!(!result.answer.is_empty(), "Answer should not be empty");
    // Should have used Focused strategy (Phillips' discovery → document match).
    assert!(
        result.stats.strategy.contains("Focused"),
        "Should use Focused strategy, got: {}",
        result.stats.strategy
    );
}

// ===========================================================================
// Integration Test 3: Timing stats validation (needs all infrastructure)
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_pipeline_stats_timing() {
    let pipeline = build_real_pipeline().await;

    let result = pipeline
        .ask("Summarize the key claims in the Awad v CFS case")
        .await
        .expect("Pipeline ask should succeed");

    println!("\n=== PIPELINE TIMING STATS ===");
    println!("  Route:      {} ms", result.stats.route_ms);
    println!("  Search:     {} ms", result.stats.search_ms);
    println!("  Expand:     {} ms", result.stats.expand_ms);
    println!("  Assemble:   {} ms", result.stats.assemble_ms);
    println!("  Synthesize: {} ms", result.stats.synthesize_ms);
    println!("  Total:      {} ms", result.stats.total_ms);
    println!("=== END TIMING ===\n");

    // All async stages should take measurable time (> 0 ms).
    // Route is in-memory and might be 0 ms on fast hardware, so we skip it.
    assert!(result.stats.search_ms > 0, "Search should take > 0 ms");
    assert!(result.stats.synthesize_ms > 0, "Synthesize should take > 0 ms");
    assert!(result.stats.total_ms > 0, "Total should take > 0 ms");

    // Total should be >= sum of individual stages.
    let sum = result.stats.route_ms
        + result.stats.search_ms
        + result.stats.expand_ms
        + result.stats.assemble_ms
        + result.stats.synthesize_ms;
    assert!(
        result.stats.total_ms >= sum,
        "Total ({}) should be >= sum of stages ({})",
        result.stats.total_ms,
        sum
    );
}

// ===========================================================================
// Helper: build a real pipeline from environment variables
// ===========================================================================

/// Construct a RagPipeline with real components using env vars.
///
/// Required env vars:
/// - QDRANT_URL (e.g., http://10.10.100.200:6333)
/// - NEO4J_URI (e.g., bolt://10.10.100.200:7687)
/// - NEO4J_USER (e.g., neo4j)
/// - NEO4J_PASSWORD
/// - ANTHROPIC_API_KEY
#[cfg(all(feature = "qdrant", feature = "fastembed", feature = "neo4j"))]
async fn build_real_pipeline() -> RagPipeline {
    use colossus_rag::{Neo4jExpander, QdrantRetriever, RigSynthesizer};
    use std::sync::Arc;

    // --- Router ---
    let router = RuleBasedRouter::legal_defaults();

    // --- Retriever (Qdrant + fastembed) ---
    //
    // QdrantRetriever::new() takes pre-built embedding model and Qdrant client.
    // We construct them here the same way the retriever_tests do.
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://10.10.100.200:6333".to_string());
    let qdrant_grpc_url = qdrant_url.replace(":6333", ":6334");

    let fastembed_client = rig_fastembed::Client::new();
    let embedding_model = Arc::new(
        fastembed_client.embedding_model(&rig_fastembed::FastembedModel::NomicEmbedTextV15),
    );

    let qdrant_client = Arc::new(
        qdrant_client::Qdrant::from_url(&qdrant_grpc_url)
            .skip_compatibility_check()
            .build()
            .expect("Failed to create Qdrant client"),
    );

    let retriever = QdrantRetriever::new(
        embedding_model,
        qdrant_client,
        "colossus_evidence",
        0.0,
    );

    // --- Expander (Neo4j) ---
    let neo4j_uri = std::env::var("NEO4J_URI").expect("NEO4J_URI must be set");
    let neo4j_user = std::env::var("NEO4J_USER").expect("NEO4J_USER must be set");
    let neo4j_pass = std::env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD must be set");
    let graph = Arc::new(
        neo4rs::Graph::new(&neo4j_uri, &neo4j_user, &neo4j_pass)
            .await
            .expect("Neo4j should connect"),
    );
    let expander = Neo4jExpander::new(graph);

    // --- Assembler ---
    let assembler = LegalAssembler::new();

    // --- Synthesizer (Claude via Rig) ---
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let synthesizer = RigSynthesizer::claude(&api_key, "claude-sonnet-4-20250514", 4096)
        .expect("RigSynthesizer should build");

    // --- Build pipeline ---
    RagPipeline::builder()
        .router(Box::new(router))
        .retriever(Box::new(retriever))
        .expander(Box::new(expander))
        .assembler(Box::new(assembler))
        .synthesizer(Box::new(synthesizer))
        .build()
        .expect("Pipeline should build")
}

/// Fallback when not all features are enabled — panics with a helpful message.
#[cfg(not(all(feature = "qdrant", feature = "fastembed", feature = "neo4j")))]
async fn build_real_pipeline() -> RagPipeline {
    panic!(
        "Integration tests require --features full. \
         Run: cargo test -p colossus-rag --features full -- --ignored"
    );
}
