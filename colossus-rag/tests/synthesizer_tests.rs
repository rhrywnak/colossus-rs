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

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use colossus_extract::{LlmProvider, LlmResponse, PipelineError};
use colossus_rag::{AssembledContext, RagError, RigSynthesizer, Synthesizer};

// ===========================================================================
// TestLlmProvider — minimal stub implementing the LlmProvider trait
// ===========================================================================

/// A tiny in-memory provider for unit tests. Each call to `invoke()` or
/// `invoke_with_system()` returns a clone of the configured result and records
/// the call so tests can assert on interaction (which method was called, with
/// what arguments).
///
/// `LlmResponse` is `Clone + Debug + PartialEq + Eq + Serialize + Deserialize`
/// (verified in colossus-extract's traits.rs), so cloning is free of surprises.
///
/// Interior mutability via `std::sync::Mutex` is required because the trait
/// methods take `&self`. Locks are held briefly (push + drop), so a blocking
/// mutex is correct here — no need for `tokio::sync::Mutex`.
struct TestLlmProvider {
    result: Result<LlmResponse, String>,
    invoke_calls: Mutex<Vec<String>>,
    invoke_with_system_calls: Mutex<Vec<(String, String)>>,
}

impl TestLlmProvider {
    fn new_ok(response: LlmResponse) -> Self {
        Self {
            result: Ok(response),
            invoke_calls: Mutex::new(Vec::new()),
            invoke_with_system_calls: Mutex::new(Vec::new()),
        }
    }

    fn new_err(msg: &str) -> Self {
        Self {
            result: Err(msg.to_string()),
            invoke_calls: Mutex::new(Vec::new()),
            invoke_with_system_calls: Mutex::new(Vec::new()),
        }
    }

    fn invoke_call_count(&self) -> usize {
        self.invoke_calls.lock().unwrap().len()
    }

    fn invoke_with_system_call_count(&self) -> usize {
        self.invoke_with_system_calls.lock().unwrap().len()
    }

    fn last_invoke_with_system_args(&self) -> Option<(String, String)> {
        self.invoke_with_system_calls
            .lock()
            .unwrap()
            .last()
            .cloned()
    }
}

#[async_trait]
impl LlmProvider for TestLlmProvider {
    async fn invoke(&self, prompt: &str, _max_tokens: u32) -> Result<LlmResponse, PipelineError> {
        self.invoke_calls.lock().unwrap().push(prompt.to_string());
        match &self.result {
            Ok(r) => Ok(r.clone()),
            Err(msg) => Err(PipelineError::LlmProvider(msg.clone())),
        }
    }

    // Explicit override of the default trait method. The default impl
    // concatenates system+prompt and delegates to `invoke()`, which would
    // hide whether `RigSynthesizer::synthesize()` is calling the preferred
    // `invoke_with_system` path or falling back to `invoke`. By recording
    // calls here separately, the test can distinguish the two paths.
    async fn invoke_with_system(
        &self,
        system: &str,
        prompt: &str,
        _max_tokens: u32,
    ) -> Result<LlmResponse, PipelineError> {
        self.invoke_with_system_calls
            .lock()
            .unwrap()
            .push((system.to_string(), prompt.to_string()));
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
    let provider: Arc<dyn LlmProvider> = match result {
        Ok(r) => Arc::new(TestLlmProvider::new_ok(r)),
        Err(msg) => Arc::new(TestLlmProvider::new_err(&msg)),
    };
    RigSynthesizer::new(provider, 256)
}

// ===========================================================================
// Unit tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: Provider errors surface as RagError::SynthesisError
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
    let provider = Arc::new(TestLlmProvider::new_ok(LlmResponse {
        text: "hello".into(),
        input_tokens: Some(10),
        output_tokens: Some(5),
    }));

    let synth = RigSynthesizer::new(provider.clone(), 256);

    let result = synth
        .synthesize(&test_context(), "What is the test?")
        .await
        .expect("Synthesis should succeed");

    assert_eq!(result.answer, "hello");
    assert_eq!(result.input_tokens, 10);
    assert_eq!(result.output_tokens, 5);
    assert_eq!(result.model, "test-model");
    assert_eq!(result.provider, "test-provider");
    assert!(result.citations.is_empty());

    // Interaction assertions — prove the synthesizer uses invoke_with_system,
    // not invoke directly with a concatenated prompt. Regression guard for
    // P3-1-Followup's behavior change.
    assert_eq!(
        provider.invoke_call_count(),
        0,
        "invoke() should not be called — synthesize() must go through invoke_with_system()"
    );
    assert_eq!(
        provider.invoke_with_system_call_count(),
        1,
        "invoke_with_system() must be called exactly once per synthesize()"
    );

    let (captured_system, captured_prompt) = provider
        .last_invoke_with_system_args()
        .expect("invoke_with_system must have been called");

    // System prompt should match the context's system_prompt verbatim.
    assert_eq!(
        captured_system, "You are a test assistant.",
        "system prompt must pass through unchanged to the provider"
    );

    // User prompt should contain the question and context, but NOT the system.
    assert!(
        captured_prompt.contains("Test context"),
        "user prompt should include formatted_context"
    );
    assert!(
        captured_prompt.contains("What is the test?"),
        "user prompt should include the question"
    );
    assert!(
        !captured_prompt.contains("You are a test assistant."),
        "system prompt must NOT be concatenated into the user prompt"
    );
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

// ---------------------------------------------------------------------------
// Test 7: Standalone interaction guard — synthesize() uses invoke_with_system
// ---------------------------------------------------------------------------

/// Explicitly verifies that RigSynthesizer uses invoke_with_system, not invoke.
///
/// This is a regression guard for P3-1-Followup's behavior change. If someone
/// reverts synthesize() to call invoke() with a concatenated prompt, this test
/// fails immediately with a clear reason, not with a mysterious assertion on
/// a captured-prompt field buried in a larger test.
#[tokio::test]
async fn synthesize_uses_invoke_with_system_not_invoke() {
    let provider = Arc::new(TestLlmProvider::new_ok(LlmResponse {
        text: "x".into(),
        input_tokens: None,
        output_tokens: None,
    }));

    let synth = RigSynthesizer::new(provider.clone(), 1000);
    let _ = synth.synthesize(&test_context(), "q").await.unwrap();

    assert_eq!(provider.invoke_call_count(), 0);
    assert_eq!(provider.invoke_with_system_call_count(), 1);
}

// Integration tests (real Anthropic API calls) removed in P3-1. Equivalent
// coverage exists in colossus-extract/tests/ where AnthropicProvider is
// tested directly. If we decide end-to-end synthesis integration tests
// are needed at the colossus-rag layer, open a new task.
