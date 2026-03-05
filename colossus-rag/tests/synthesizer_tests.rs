//! Tests for RigSynthesizer — unit tests and integration tests for Claude API.
//!
//! ## Test organization
//!
//! - **Unit tests** (tests 1–2): No API key needed. Test construction and config.
//! - **Integration tests** (tests 3–5): Require ANTHROPIC_API_KEY env var.
//!   Marked with `#[ignore]` — run with:
//!   `cargo test -p colossus-rag --test synthesizer_tests -- --ignored --nocapture`
//!
//! ## Note on feature gates
//!
//! Unlike QdrantRetriever (which needs `qdrant` + `fastembed` features),
//! RigSynthesizer uses only rig-core which is a base dependency. These tests
//! compile with default features — no `--features full` needed.

use colossus_rag::{
    AssembledContext, RigSynthesizer, Synthesizer,
};

// ===========================================================================
// Unit Tests — no API key needed
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1: Construction with valid params succeeds
// ---------------------------------------------------------------------------

/// Verify that `RigSynthesizer::claude()` constructs successfully with
/// valid parameters. The API key doesn't need to be real — the client
/// is only configured here, not connected.
#[test]
fn test_rig_synthesizer_construction() {
    let result = RigSynthesizer::claude(
        "test-api-key-not-real",
        "claude-haiku-4-5-20251001",
        4096,
    );

    assert!(
        result.is_ok(),
        "Construction should succeed with valid params: {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Construction with empty API key still succeeds
// ---------------------------------------------------------------------------

/// The constructor doesn't validate the API key — that happens at call time.
/// An empty key should construct fine (it will fail when `synthesize()` is called).
#[test]
fn test_rig_synthesizer_empty_api_key_constructs() {
    let result = RigSynthesizer::claude("", "claude-haiku-4-5-20251001", 4096);
    assert!(
        result.is_ok(),
        "Empty API key should still construct (fails at call time)"
    );
}

// ===========================================================================
// Integration Tests — require ANTHROPIC_API_KEY
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 3: Synthesize returns a non-empty answer
// ---------------------------------------------------------------------------

/// ## Integration test: Full synthesis pipeline
///
/// Sends a simple context + question to Claude and verifies we get a
/// non-empty answer back.
///
/// Requires: ANTHROPIC_API_KEY env var
#[tokio::test]
#[ignore]
async fn test_synthesize_returns_answer() {
    let synthesizer = create_test_synthesizer();

    let context = AssembledContext {
        system_prompt: "You are a legal analyst. Answer based only on the provided evidence. \
                        Be concise — respond in one sentence."
            .to_string(),
        formatted_context: "Evidence: Phillips testified that Emil Awad requested the return \
                           of $50,000 on multiple occasions."
            .to_string(),
        token_estimate: 50,
    };

    let result = synthesizer
        .synthesize(&context, "What did Phillips say about the $50,000?")
        .await
        .expect("Synthesis should succeed");

    assert!(
        !result.answer.is_empty(),
        "Answer should not be empty"
    );

    println!("\n  === RigSynthesizer Result ===");
    println!("  Answer: {}", result.answer);
    println!("  Input tokens: {}", result.input_tokens);
    println!("  Output tokens: {}", result.output_tokens);
    println!("  Provider: {}", result.provider);
    println!("  Model: {}", result.model);
}

// ---------------------------------------------------------------------------
// Test 4: Token counts are non-zero
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_synthesize_token_counts() {
    let synthesizer = create_test_synthesizer();

    let context = AssembledContext {
        system_prompt: "You are a helpful assistant. Be very concise.".to_string(),
        formatted_context: "The sky is blue.".to_string(),
        token_estimate: 10,
    };

    let result = synthesizer
        .synthesize(&context, "What color is the sky?")
        .await
        .expect("Synthesis should succeed");

    assert!(
        result.input_tokens > 0,
        "Input tokens should be > 0, got {}",
        result.input_tokens
    );
    assert!(
        result.output_tokens > 0,
        "Output tokens should be > 0, got {}",
        result.output_tokens
    );

    println!("\n  === Token Counts ===");
    println!("  Input tokens:  {}", result.input_tokens);
    println!("  Output tokens: {}", result.output_tokens);
}

// ---------------------------------------------------------------------------
// Test 5: Provider and model fields are correct
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_synthesize_provider_and_model() {
    let synthesizer = create_test_synthesizer();

    let context = AssembledContext {
        system_prompt: "Be concise.".to_string(),
        formatted_context: "Test context.".to_string(),
        token_estimate: 5,
    };

    let result = synthesizer
        .synthesize(&context, "Say hello")
        .await
        .expect("Synthesis should succeed");

    assert_eq!(
        result.provider, "anthropic",
        "Provider should be 'anthropic'"
    );

    let expected_model = get_test_model_id();
    assert_eq!(
        result.model, expected_model,
        "Model should match configured model ID"
    );
}

// ===========================================================================
// Test helpers
// ===========================================================================

/// Get the model ID for tests. Uses ANTHROPIC_MODEL env var or defaults
/// to a known-good cheap model.
fn get_test_model_id() -> String {
    std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string())
}

/// Create a RigSynthesizer configured for integration tests.
///
/// Uses ANTHROPIC_API_KEY from environment and a cheap model to minimize cost.
/// max_tokens = 128 keeps responses short and fast.
fn create_test_synthesizer() -> RigSynthesizer {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set for integration tests");
    let model_id = get_test_model_id();

    RigSynthesizer::claude(&api_key, &model_id, 128)
        .expect("Failed to create RigSynthesizer")
}
