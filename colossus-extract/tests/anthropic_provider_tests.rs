use colossus_extract::{AnthropicProvider, LlmProvider};

/// Verifies that `AnthropicProvider` satisfies `LlmProvider` object safety.
///
/// Object safety is required because providers are stored as `Arc<dyn LlmProvider>`
/// in `AppContext`. This test both checks the trait-level object safety (via
/// `Option<Box<dyn LlmProvider>>`) and verifies that a real `AnthropicProvider`
/// instance can be boxed as a trait object — catching any future field or method
/// change that breaks dynamic dispatch.
#[test]
fn anthropic_provider_is_object_safe() {
    let _: Option<Box<dyn LlmProvider>> = None;

    let provider =
        AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None).unwrap();
    let _boxed: Box<dyn LlmProvider> = Box::new(provider);
}

/// Verifies that `model_name()` returns the exact string passed to the constructor.
///
/// `model_name()` feeds `pipeline_events` records and cost-tracking logs. If the
/// return value diverges from the constructor input, log queries and cost rollups
/// break silently.
#[test]
fn model_name_returns_constructor_value() {
    let provider =
        AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None).unwrap();
    assert_eq!(provider.model_name(), "claude-sonnet-4-6");
}

/// Verifies the `max_tokens_default()` accessor returns the constructor value.
///
/// This accessor is for future factory and diagnostic use — the factory function
/// (P2-6) reads env vars and passes max_tokens_default to the constructor, and
/// diagnostics may need to inspect the configured value.
#[test]
fn max_tokens_default_accessor() {
    let provider =
        AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None).unwrap();
    assert_eq!(provider.max_tokens_default(), 4096);
}

/// Verifies that `cost_per_input_token()` and `cost_per_output_token()` return `None`.
///
/// Pricing is intentionally externalized: per-token pricing changes with model
/// and over time, and belongs in application-level config (env vars, database),
/// not in a library crate. These methods return `None` as contract placeholders.
/// A future `AppContext`-level pricing lookup computes cost from
/// `LlmResponse.input_tokens` / `output_tokens`.
#[test]
fn cost_methods_return_none() {
    let provider =
        AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None).unwrap();
    assert_eq!(provider.cost_per_input_token(), None);
    assert_eq!(provider.cost_per_output_token(), None);
}

/// Verifies that `supports_structured_output()` returns `true` for Anthropic.
///
/// Anthropic supports tool-use for structured JSON output. The `LlmExtract` step
/// uses this signal to choose between typed-prompt extraction (reliable, Anthropic)
/// and raw-completion + JSON-repair (fallback for vLLM and similar).
#[test]
fn supports_structured_output_returns_true() {
    let provider =
        AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None).unwrap();
    assert!(provider.supports_structured_output());
}

/// Verifies that `provider_name()` returns the canonical lowercase `"anthropic"`.
///
/// This is the string consumers use to group metrics and costs by backend.
/// Drift here (e.g., `"Anthropic"` or `"anthropic-api"`) would silently break
/// downstream aggregation queries that filter on `provider_name`.
#[test]
fn provider_name_returns_anthropic() {
    let provider =
        AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None).unwrap();
    assert_eq!(provider.provider_name(), "anthropic");
}

/// Verifies that the constructor succeeds with valid inputs.
///
/// This test confirms that `reqwest::Client::builder()` builds successfully in
/// a normal environment (TLS backend available, no poisoned global state). A
/// failure here would indicate an environmental issue, not a code bug.
#[test]
fn constructor_succeeds_with_valid_inputs() {
    let result = AnthropicProvider::new("test-key".into(), "claude-sonnet-4-6".into(), 4096, None, None);
    assert!(result.is_ok());
}
