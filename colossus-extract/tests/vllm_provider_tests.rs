use colossus_extract::{PipelineError, VllmProvider};

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
        None,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing slash
    let result = VllmProvider::new(
        "http://localhost:8000/".into(),
        "llama-3-8b".into(),
        None,
        2048,
        None,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing /v1/
    let result = VllmProvider::new(
        "http://localhost:8000/v1/".into(),
        "llama-3-8b".into(),
        None,
        2048,
        None,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));
}
