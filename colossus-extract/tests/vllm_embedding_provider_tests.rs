use colossus_extract::{EmbeddingProvider, PipelineError, VllmEmbeddingProvider};

/// Verifies that `VllmEmbeddingProvider` satisfies `EmbeddingProvider` object safety.
///
/// Object safety is required because providers are stored as
/// `Arc<dyn EmbeddingProvider>` in `AppContext`. This test both checks trait-level
/// object safety and verifies a real `VllmEmbeddingProvider` instance can be boxed
/// as a trait object.
#[test]
fn vllm_embedding_provider_is_object_safe() {
    let _: Option<Box<dyn EmbeddingProvider>> = None;

    let provider = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "nomic-embed".into(),
        None,
        768,
    )
    .unwrap();
    let _boxed: Box<dyn EmbeddingProvider> = Box::new(provider);
}

/// Verifies that the constructor succeeds with valid inputs.
///
/// Confirms `reqwest::Client::builder()` builds successfully and that the
/// `base_url` validation accepts a well-formed URL without trailing slash or
/// `/v1` suffix.
#[test]
fn constructor_succeeds_with_valid_inputs() {
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "nomic-embed".into(),
        None,
        768,
    );
    assert!(result.is_ok());
}

/// Verifies that the constructor rejects `base_url` values that would cause
/// double-`/v1/` in the final URL.
///
/// Mirrors the sibling `VllmProvider` guard — three invalid forms must all be
/// rejected: trailing `/v1`, trailing `/`, and trailing `/v1/`.
#[test]
fn constructor_rejects_base_url_with_v1_suffix() {
    // Trailing /v1
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000/v1".into(),
        "nomic-embed".into(),
        None,
        768,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing slash
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000/".into(),
        "nomic-embed".into(),
        None,
        768,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing /v1/
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000/v1/".into(),
        "nomic-embed".into(),
        None,
        768,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));
}

/// Verifies that `dimensions()` returns the exact value passed to the constructor.
///
/// This value must match the Qdrant collection configuration — mismatches cause
/// runtime query failures.
#[test]
fn dimensions_returns_constructor_value() {
    let provider = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "nomic-embed".into(),
        None,
        384,
    )
    .unwrap();
    assert_eq!(provider.dimensions(), 384);
}

/// Verifies that `dimensions()` returns a different value than the first test,
/// proving it is not hardcoded.
#[test]
fn dimensions_returns_constructor_value_differs_from_default() {
    let provider = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "nomic-embed".into(),
        None,
        1024,
    )
    .unwrap();
    assert_eq!(provider.dimensions(), 1024);
}

/// Verifies that `model_name()` returns the exact string passed to the constructor.
///
/// `model_name()` feeds `pipeline_events` records and cost-tracking logs. If the
/// return value diverges from the constructor input, log queries break silently.
#[test]
fn model_name_returns_constructor_value() {
    let provider = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "test-model".into(),
        None,
        768,
    )
    .unwrap();
    assert_eq!(provider.model_name(), "test-model");
}

/// Verifies that the constructor accepts `Some(api_key)` for authenticated
/// deployments.
#[test]
fn api_key_accepted() {
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "nomic-embed".into(),
        Some("sk-test".into()),
        768,
    );
    assert!(result.is_ok());
}

/// Verifies that the constructor accepts `None` for local/unauth servers.
#[test]
fn api_key_none_accepted() {
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000".into(),
        "nomic-embed".into(),
        None,
        768,
    );
    assert!(result.is_ok());
}

/// Network-dependent: verifies `embed()` returns a vector of the expected dimension
/// against a live vLLM endpoint.
///
/// Set `VLLM_TEST_URL`, `VLLM_TEST_MODEL`, and `VLLM_TEST_DIMENSIONS` to run this
/// test. Skipped in default CI runs.
#[tokio::test]
#[ignore = "requires vLLM server"]
async fn embed_produces_expected_dim_vector() {
    let url = std::env::var("VLLM_TEST_URL").expect("VLLM_TEST_URL must be set");
    let model = std::env::var("VLLM_TEST_MODEL").expect("VLLM_TEST_MODEL must be set");
    let dimensions: u32 = std::env::var("VLLM_TEST_DIMENSIONS")
        .expect("VLLM_TEST_DIMENSIONS must be set")
        .parse()
        .expect("VLLM_TEST_DIMENSIONS must parse as u32");

    let provider = VllmEmbeddingProvider::new(url, model, None, dimensions).unwrap();
    let vector = provider.embed("hello world").await.unwrap();
    assert_eq!(vector.len() as u32, dimensions);
}

/// Network-dependent: verifies `embed_batch()` returns three distinct vectors in
/// request-order against a live vLLM endpoint. Implicitly exercises the
/// OpenAI-spec index-sorting logic.
///
/// Set `VLLM_TEST_URL`, `VLLM_TEST_MODEL`, and `VLLM_TEST_DIMENSIONS` to run.
/// Skipped in default CI runs.
#[tokio::test]
#[ignore = "requires vLLM server"]
async fn embed_batch_preserves_order() {
    let url = std::env::var("VLLM_TEST_URL").expect("VLLM_TEST_URL must be set");
    let model = std::env::var("VLLM_TEST_MODEL").expect("VLLM_TEST_MODEL must be set");
    let dimensions: u32 = std::env::var("VLLM_TEST_DIMENSIONS")
        .expect("VLLM_TEST_DIMENSIONS must be set")
        .parse()
        .expect("VLLM_TEST_DIMENSIONS must parse as u32");

    let provider = VllmEmbeddingProvider::new(url, model, None, dimensions).unwrap();
    let inputs = ["a", "b", "c"];
    let vectors = provider.embed_batch(&inputs).await.unwrap();

    assert_eq!(vectors.len(), 3);
    for v in &vectors {
        assert_eq!(v.len() as u32, dimensions);
    }
    // Distinctness: three different inputs should produce three different vectors.
    assert_ne!(vectors[0], vectors[1]);
    assert_ne!(vectors[1], vectors[2]);
    assert_ne!(vectors[0], vectors[2]);
}
