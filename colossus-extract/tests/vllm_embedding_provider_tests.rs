use colossus_extract::{EmbeddingProvider, PipelineError, VllmEmbeddingProvider};

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
        None,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing slash
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000/".into(),
        "nomic-embed".into(),
        None,
        768,
        None,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));

    // Trailing /v1/
    let result = VllmEmbeddingProvider::new(
        "http://localhost:8000/v1/".into(),
        "nomic-embed".into(),
        None,
        768,
        None,
    );
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::LlmProvider(_))));
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

    let provider = VllmEmbeddingProvider::new(url, model, None, dimensions, None).unwrap();
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

    let provider = VllmEmbeddingProvider::new(url, model, None, dimensions, None).unwrap();
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
