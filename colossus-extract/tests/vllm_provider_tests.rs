use colossus_extract::{LlmProvider, PipelineError, VllmProvider};

/// Verifies that `VllmProvider` satisfies `LlmProvider` object safety.
///
/// Object safety is required because providers are stored as `Arc<dyn LlmProvider>`
/// in `AppContext`. This test both checks the trait-level object safety and verifies
/// that a real `VllmProvider` instance can be boxed as a trait object.
#[test]
fn vllm_provider_is_object_safe() {
    let _: Option<Box<dyn LlmProvider>> = None;

    let provider = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    )
    .unwrap();
    let _boxed: Box<dyn LlmProvider> = Box::new(provider);
}

/// Verifies that `model_name()` returns the exact string passed to the constructor.
///
/// `model_name()` feeds `pipeline_events` records and cost-tracking logs. If the
/// return value diverges from the constructor input, log queries break silently.
#[test]
fn model_name_returns_constructor_value() {
    let provider = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    )
    .unwrap();
    assert_eq!(provider.model_name(), "llama-3-8b");
}

/// Verifies the `max_tokens_default()` accessor returns the constructor value.
///
/// This accessor is for future factory and diagnostic use — the factory function
/// (P2-6) reads env vars and passes `max_tokens_default` to the constructor.
#[test]
fn max_tokens_default_accessor() {
    let provider = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    )
    .unwrap();
    assert_eq!(provider.max_tokens_default(), 2048);
}

/// Verifies that `cost_per_input_token()` and `cost_per_output_token()` return `None`.
///
/// Local vLLM inference has no per-token dollar cost — hardware amortization is
/// tracked outside this crate. These methods return `None` as contract placeholders.
#[test]
fn cost_methods_return_none() {
    let provider = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    )
    .unwrap();
    assert_eq!(provider.cost_per_input_token(), None);
    assert_eq!(provider.cost_per_output_token(), None);
}

/// Verifies that `supports_structured_output()` returns `false` for vLLM.
///
/// This is the key difference from `AnthropicProvider` (which returns `true`).
/// vLLM's structured output uses `guided_choice` / `structured_outputs` via
/// `extra_body`, a different mechanism from OpenAI's `response_format` JSON mode.
/// Returning `false` signals the `LlmExtract` step to use raw-completion +
/// JSON-repair parsing rather than typed-prompt extraction.
#[test]
fn supports_structured_output_returns_false() {
    let provider = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    )
    .unwrap();
    assert!(!provider.supports_structured_output());
}

/// Verifies that `provider_name()` returns the canonical lowercase `"vllm"`.
///
/// This is the string consumers use to group metrics and costs by backend.
/// Drift here (e.g., `"vLLM"` or `"openai-compat"`) would silently break
/// downstream aggregation queries that filter on `provider_name`.
#[test]
fn provider_name_returns_vllm() {
    let provider = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    )
    .unwrap();
    assert_eq!(provider.provider_name(), "vllm");
}

/// Verifies that the constructor succeeds with valid inputs.
///
/// This test confirms that `reqwest::Client::builder()` builds successfully and
/// that the `base_url` validation accepts a well-formed URL without trailing
/// slash or `/v1` suffix.
#[test]
fn constructor_succeeds_with_valid_inputs() {
    let result = VllmProvider::new(
        "http://localhost:8000".into(),
        "llama-3-8b".into(),
        None,
        2048,
    );
    assert!(result.is_ok());
}

/// Verifies that the constructor rejects `base_url` values that would cause
/// double-`/v1/` in the final URL.
///
/// The common mistake when integrating with vLLM is copy-pasting the OpenAI SDK's
/// `base_url` convention (which ends in `/v1`) into a raw HTTP client that then
/// appends `/v1/chat/completions`, producing `/v1/v1/chat/completions`. The
/// constructor guard prevents this class of bug for all three invalid forms:
/// trailing `/v1`, trailing `/`, and trailing `/v1/`.
#[test]
fn constructor_rejects_base_url_with_v1_suffix() {
    // Trailing /v1
    let result = VllmProvider::new(
        "http://localhost:8000/v1".into(),
        "llama-3-8b".into(),
        None,
        2048,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing slash
    let result = VllmProvider::new(
        "http://localhost:8000/".into(),
        "llama-3-8b".into(),
        None,
        2048,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing /v1/
    let result = VllmProvider::new(
        "http://localhost:8000/v1/".into(),
        "llama-3-8b".into(),
        None,
        2048,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));
}
