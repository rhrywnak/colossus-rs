//! Unit tests for RigSynthesizer (P3-1 rewrite).
//!
//! These tests exercise `RigSynthesizer` through the `LlmProvider` abstraction
//! by injecting a tiny in-memory `TestLlmProvider` stub. No network, no API
//! keys, no feature flags.
//!
//! ## What changed from the old test file
//!
//! The prior suite constructed `RigSynthesizer::claude(api_key, model, max)`
//! which no longer exists. The new constructor takes
//! `Arc<dyn colossus_extract::LlmProvider>`, so tests build a stub provider
//! and verify the mapping from `LlmResponse` to `SynthesisResult`.
//!
//! See the note at the bottom about integration tests.

use std::sync::Arc;

use async_trait::async_trait;

use colossus_extract::{LlmProvider, LlmResponse, PipelineError};
use colossus_rag::{AssembledContext, RagError, RigSynthesizer, Synthesizer};

// ===========================================================================
// TestLlmProvider — minimal stub implementing the LlmProvider trait
// ===========================================================================

/// A tiny in-memory provider for unit tests. Each call to `invoke()` returns
/// a clone of the configured result.
///
/// `LlmResponse` is `Clone + Debug + PartialEq + Eq + Serialize + Deserialize`
/// (verified in colossus-extract's traits.rs), so cloning is free of surprises.
struct TestLlmProvider {
    result: Result<LlmResponse, String>,
}

#[async_trait]
impl LlmProvider for TestLlmProvider {
    async fn invoke(&self, _prompt: &str, _max_tokens: u32) -> Result<LlmResponse, PipelineError> {
        match &self.result {
            Ok(r) => Ok(r.clone()),
            Err(msg) => Err(PipelineError::LlmProvider(msg.clone())),
        }
    }

    fn model_name(&self) -> &str {
        "test-model"
    }

    fn provider_name(&self) -> &str {
        "test-provider"
    }

    fn cost_per_input_token(&self) -> Option<f64> {
        None
    }

    fn cost_per_output_token(&self) -> Option<f64> {
        None
    }

    fn supports_structured_output(&self) -> bool {
        false
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Build an `AssembledContext` with enough substance for the synthesizer to
/// format. Uses `Default` plus explicit overrides for the fields we care about.
fn test_context() -> AssembledContext {
    AssembledContext {
        system_prompt: "You are a test assistant.".into(),
        formatted_context: "Test context".into(),
        token_estimate: 100,
    }
}

fn make_synthesizer(result: Result<LlmResponse, String>) -> RigSynthesizer {
    let provider: Arc<dyn LlmProvider> = Arc::new(TestLlmProvider { result });
    RigSynthesizer::new(provider, 256)
}

// ===========================================================================
// Unit tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1: Construction with a stub provider does not panic
// ---------------------------------------------------------------------------

#[test]
fn test_rig_synthesizer_construction() {
    let _ = make_synthesizer(Ok(LlmResponse {
        text: "ok".into(),
        input_tokens: None,
        output_tokens: None,
    }));
}

// ---------------------------------------------------------------------------
// Test 2: RigSynthesizer implements the Synthesizer trait
// ---------------------------------------------------------------------------

/// Compile-test: if the `Synthesizer` trait impl is broken (wrong signature,
/// missing `#[async_trait]`, etc.) this function will fail to compile.
#[test]
fn test_rig_synthesizer_implements_trait() {
    fn takes_synthesizer(_: Box<dyn Synthesizer>) {}

    let synth = make_synthesizer(Ok(LlmResponse {
        text: "ok".into(),
        input_tokens: None,
        output_tokens: None,
    }));
    takes_synthesizer(Box::new(synth));
}

// ---------------------------------------------------------------------------
// Test 3: Provider errors surface as RagError::SynthesisError
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_synthesize_propagates_provider_error() {
    let synth = make_synthesizer(Err("test".into()));
    let ctx = test_context();

    let result = synth.synthesize(&ctx, "Anything?").await;

    assert!(
        matches!(result, Err(RagError::SynthesisError(_))),
        "Expected SynthesisError, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 4: Empty LLM text produces SynthesisError
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_synthesize_empty_response_is_error() {
    let synth = make_synthesizer(Ok(LlmResponse {
        text: "".into(),
        input_tokens: None,
        output_tokens: None,
    }));

    let result = synth.synthesize(&test_context(), "Anything?").await;

    assert!(
        matches!(result, Err(RagError::SynthesisError(_))),
        "Expected SynthesisError for empty text, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 5: Successful invoke populates all SynthesisResult fields
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_synthesize_success_populates_fields() {
    let synth = make_synthesizer(Ok(LlmResponse {
        text: "hello".into(),
        input_tokens: Some(10),
        output_tokens: Some(5),
    }));

    let result = synth
        .synthesize(&test_context(), "Say hello")
        .await
        .expect("Synthesis should succeed");

    assert_eq!(result.answer, "hello");
    assert_eq!(result.input_tokens, 10);
    assert_eq!(result.output_tokens, 5);
    assert_eq!(result.model, "test-model");
    assert_eq!(result.provider, "test-provider");
    assert!(result.citations.is_empty());
}

// ---------------------------------------------------------------------------
// Test 6: Missing token counts default to zero
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_synthesize_none_tokens_default_to_zero() {
    let synth = make_synthesizer(Ok(LlmResponse {
        text: "hi".into(),
        input_tokens: None,
        output_tokens: None,
    }));

    let result = synth
        .synthesize(&test_context(), "Say hi")
        .await
        .expect("Synthesis should succeed");

    assert_eq!(result.input_tokens, 0);
    assert_eq!(result.output_tokens, 0);
}

// Integration tests (real Anthropic API calls) removed in P3-1. Equivalent
// coverage exists in colossus-extract/tests/ where AnthropicProvider is
// tested directly. If we decide end-to-end synthesis integration tests
// are needed at the colossus-rag layer, open a new task.
