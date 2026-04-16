//! Integration tests for the provider factories.
//!
//! These tests exercise the full factory path including provider construction.
//! They use a HashMap-backed lookup closure instead of `std::env::var` so they
//! can run in parallel without racing on process-global state.

use std::collections::HashMap;

use colossus_extract::{embedding_provider_from_lookup, llm_provider_from_lookup, PipelineError};

/// Build an env map from pairs.
fn env_from(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// Wrap a map as the `Fn(&str) -> Option<String>` closure that the factory
/// expects. The closure borrows the map and returns owned `String` values —
/// matching the signature but without touching process env state.
fn make_lookup(map: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
    move |key: &str| map.get(key).cloned()
}

/// Assert that an error is `PipelineError::LlmProvider` and that its message
/// contains the given needle. Panics with a clear message on mismatch.
fn assert_error_contains(err: &PipelineError, needle: &str) {
    match err {
        PipelineError::LlmProvider(msg) => {
            assert!(
                msg.contains(needle),
                "error message {msg:?} should contain {needle:?}"
            );
        }
        other => panic!("expected LlmProvider variant with {needle:?}, got {other:?}"),
    }
}

// =========================================================================
// LLM factory — Anthropic branch
// =========================================================================

#[test]
fn llm_defaults_to_anthropic_when_provider_unset() {
    let env = env_from(&[
        ("ANTHROPIC_API_KEY", "test-key"),
        ("LLM_MODEL", "claude-sonnet-4-6"),
    ]);
    let provider = llm_provider_from_lookup(&make_lookup(&env)).expect("should build");
    assert_eq!(provider.model_name(), "claude-sonnet-4-6");
}

#[test]
fn llm_anthropic_constructs_with_required_vars() {
    let env = env_from(&[
        ("LLM_PROVIDER", "anthropic"),
        ("ANTHROPIC_API_KEY", "test-key"),
        ("LLM_MODEL", "claude-sonnet-4-6"),
    ]);
    let provider = llm_provider_from_lookup(&make_lookup(&env)).expect("should build");
    assert_eq!(provider.model_name(), "claude-sonnet-4-6");
}

#[test]
fn llm_anthropic_missing_api_key_errors() {
    let env = env_from(&[
        ("LLM_PROVIDER", "anthropic"),
        ("LLM_MODEL", "claude-sonnet-4-6"),
    ]);
    let err = llm_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "ANTHROPIC_API_KEY");
}

#[test]
fn llm_anthropic_missing_model_errors() {
    // Verifies the research-justified deviation from v5_2: no LLM_MODEL default.
    let env = env_from(&[
        ("LLM_PROVIDER", "anthropic"),
        ("ANTHROPIC_API_KEY", "test-key"),
    ]);
    let err = llm_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "LLM_MODEL");
}

#[test]
fn llm_anthropic_max_tokens_default_when_absent() {
    let env = env_from(&[
        ("ANTHROPIC_API_KEY", "test-key"),
        ("LLM_MODEL", "claude-sonnet-4-6"),
    ]);
    // Construction should succeed. We can't inspect max_tokens through the
    // trait object, so the assertion is simply "does not error."
    let _provider =
        llm_provider_from_lookup(&make_lookup(&env)).expect("should build with default");
}

#[test]
fn llm_anthropic_max_tokens_parsed_from_env() {
    let env = env_from(&[
        ("ANTHROPIC_API_KEY", "test-key"),
        ("LLM_MODEL", "claude-sonnet-4-6"),
        ("LLM_MAX_TOKENS", "16000"),
    ]);
    let _provider =
        llm_provider_from_lookup(&make_lookup(&env)).expect("should build with explicit value");
}

#[test]
fn llm_anthropic_max_tokens_rejects_non_numeric() {
    let env = env_from(&[
        ("ANTHROPIC_API_KEY", "test-key"),
        ("LLM_MODEL", "claude-sonnet-4-6"),
        ("LLM_MAX_TOKENS", "many"),
    ]);
    let err = llm_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "LLM_MAX_TOKENS");
    assert_error_contains(&err, "many");
}

// =========================================================================
// LLM factory — vLLM branch
// =========================================================================

#[test]
fn llm_vllm_constructs_with_required_vars() {
    let env = env_from(&[
        ("LLM_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("LLM_MODEL", "llama-3-8b"),
    ]);
    let provider = llm_provider_from_lookup(&make_lookup(&env)).expect("should build");
    assert_eq!(provider.model_name(), "llama-3-8b");
}

#[test]
fn llm_vllm_missing_base_url_errors() {
    let env = env_from(&[("LLM_PROVIDER", "vllm"), ("LLM_MODEL", "llama-3-8b")]);
    let err = llm_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "VLLM_BASE_URL");
}

#[test]
fn llm_vllm_missing_model_errors() {
    let env = env_from(&[
        ("LLM_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
    ]);
    let err = llm_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "LLM_MODEL");
}

#[test]
fn llm_vllm_api_key_optional() {
    // vLLM without VLLM_API_KEY must succeed — local deployments have no auth.
    let env = env_from(&[
        ("LLM_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("LLM_MODEL", "llama-3-8b"),
    ]);
    let _provider =
        llm_provider_from_lookup(&make_lookup(&env)).expect("missing api key should be ok");
}

#[test]
fn llm_unknown_provider_errors_with_valid_options_message() {
    let env = env_from(&[("LLM_PROVIDER", "gpt-ultra")]);
    let err = llm_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "gpt-ultra");
    assert_error_contains(&err, "anthropic");
    assert_error_contains(&err, "vllm");
}

// =========================================================================
// Embedding factory — fastembed (HuggingFace mode)
// =========================================================================
//
// HuggingFace-mode construction tests are #[ignore] because they trigger a
// model download from HuggingFace Hub (~100-270 MB depending on model).
// Existing P2-4 tests use the same pattern. Run explicitly with:
//   cargo test -p colossus-extract --test factory_tests -- --ignored

#[test]
#[ignore = "downloads from HuggingFace; run with --ignored"]
fn embedding_defaults_to_fastembed_when_provider_unset() {
    let env = env_from(&[("FASTEMBED_MODEL", "AllMiniLML6V2")]);
    let provider = embedding_provider_from_lookup(&make_lookup(&env)).expect("should build");
    assert_eq!(provider.dimensions(), 384);
}

#[test]
#[ignore = "downloads from HuggingFace; run with --ignored"]
fn embedding_fastembed_huggingface_constructs() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "huggingface"),
        ("FASTEMBED_MODEL", "AllMiniLML6V2"),
    ]);
    let provider = embedding_provider_from_lookup(&make_lookup(&env)).expect("should build");
    assert_eq!(provider.dimensions(), 384);
}

#[test]
fn embedding_fastembed_missing_model_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "huggingface"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "FASTEMBED_MODEL");
}

#[test]
fn embedding_fastembed_unknown_model_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "huggingface"),
        ("FASTEMBED_MODEL", "SomeUnknownModel"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "SomeUnknownModel");
    assert_error_contains(&err, "curated whitelist");
}

#[test]
fn embedding_fastembed_unknown_mode_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "rocket-ship"),
        ("FASTEMBED_MODEL", "AllMiniLML6V2"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "rocket-ship");
    assert_error_contains(&err, "FASTEMBED_MODE");
}

// =========================================================================
// Embedding factory — fastembed (local mode)
// =========================================================================

#[test]
fn embedding_fastembed_local_missing_onnx_errors() {
    // All other vars present; FASTEMBED_LOCAL_ONNX_PATH missing.
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "local"),
        ("FASTEMBED_MODEL", "AllMiniLML6V2"),
        ("FASTEMBED_LOCAL_TOKENIZER_PATH", "/tmp/t.json"),
        ("FASTEMBED_LOCAL_CONFIG_PATH", "/tmp/c.json"),
        ("FASTEMBED_LOCAL_SPECIAL_TOKENS_PATH", "/tmp/s.json"),
        ("FASTEMBED_LOCAL_TOKENIZER_CONFIG_PATH", "/tmp/tc.json"),
        ("FASTEMBED_LOCAL_DIMENSIONS", "384"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "FASTEMBED_LOCAL_ONNX_PATH");
}

#[test]
fn embedding_fastembed_local_missing_dimensions_errors() {
    // Create a real dummy file so path validation doesn't fail first.
    let dir = tempfile::tempdir().expect("tempdir");
    let f = dir.path().join("dummy");
    std::fs::write(&f, b"x").expect("write");
    let p = f.to_str().expect("utf8 path");

    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "local"),
        ("FASTEMBED_MODEL", "AllMiniLML6V2"),
        ("FASTEMBED_LOCAL_ONNX_PATH", p),
        ("FASTEMBED_LOCAL_TOKENIZER_PATH", p),
        ("FASTEMBED_LOCAL_CONFIG_PATH", p),
        ("FASTEMBED_LOCAL_SPECIAL_TOKENS_PATH", p),
        ("FASTEMBED_LOCAL_TOKENIZER_CONFIG_PATH", p),
        // FASTEMBED_LOCAL_DIMENSIONS deliberately absent
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "FASTEMBED_LOCAL_DIMENSIONS");
}

#[test]
fn embedding_fastembed_local_nonexistent_path_errors() {
    // File reads should fail with a message including both the env var name
    // and the offending path — the operator needs both to diagnose.
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "fastembed"),
        ("FASTEMBED_MODE", "local"),
        ("FASTEMBED_MODEL", "AllMiniLML6V2"),
        ("FASTEMBED_LOCAL_ONNX_PATH", "/nonexistent/model.onnx"),
        ("FASTEMBED_LOCAL_TOKENIZER_PATH", "/nonexistent/t.json"),
        ("FASTEMBED_LOCAL_CONFIG_PATH", "/nonexistent/c.json"),
        ("FASTEMBED_LOCAL_SPECIAL_TOKENS_PATH", "/nonexistent/s.json"),
        (
            "FASTEMBED_LOCAL_TOKENIZER_CONFIG_PATH",
            "/nonexistent/tc.json",
        ),
        ("FASTEMBED_LOCAL_DIMENSIONS", "384"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "FASTEMBED_LOCAL_ONNX_PATH");
    assert_error_contains(&err, "/nonexistent/model.onnx");
}

// =========================================================================
// Embedding factory — vLLM branch
// =========================================================================

#[test]
fn embedding_vllm_constructs_with_required_vars() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        ("EMBEDDING_DIMENSIONS", "768"),
    ]);
    let provider = embedding_provider_from_lookup(&make_lookup(&env)).expect("should build");
    assert_eq!(provider.dimensions(), 768);
    assert_eq!(provider.model_name(), "nomic-embed-text-v1.5");
}

#[test]
fn embedding_vllm_missing_base_url_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "vllm"),
        ("EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        ("EMBEDDING_DIMENSIONS", "768"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "VLLM_BASE_URL");
}

#[test]
fn embedding_vllm_missing_model_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("EMBEDDING_DIMENSIONS", "768"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "EMBEDDING_MODEL");
}

#[test]
fn embedding_vllm_missing_dimensions_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "EMBEDDING_DIMENSIONS");
}

#[test]
fn embedding_vllm_non_numeric_dimensions_errors() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        ("EMBEDDING_DIMENSIONS", "seven-sixty-eight"),
    ]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "EMBEDDING_DIMENSIONS");
    assert_error_contains(&err, "seven-sixty-eight");
}

#[test]
fn embedding_vllm_api_key_optional() {
    let env = env_from(&[
        ("EMBEDDING_PROVIDER", "vllm"),
        ("VLLM_BASE_URL", "http://localhost:8000"),
        ("EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        ("EMBEDDING_DIMENSIONS", "768"),
        // no VLLM_API_KEY
    ]);
    let _provider =
        embedding_provider_from_lookup(&make_lookup(&env)).expect("no api key should be ok");
}

#[test]
fn embedding_unknown_provider_errors_with_valid_options_message() {
    let env = env_from(&[("EMBEDDING_PROVIDER", "magic-embeddings")]);
    let err = embedding_provider_from_lookup(&make_lookup(&env))
        .err()
        .expect("expected factory to error");
    assert_error_contains(&err, "magic-embeddings");
    assert_error_contains(&err, "fastembed");
    assert_error_contains(&err, "vllm");
}
