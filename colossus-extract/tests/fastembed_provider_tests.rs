use colossus_extract::EmbeddingProvider;

/// Constructs a real `FastembedProvider`, downloads the smallest model, and
/// verifies that `embed()` returns a vector of the expected dimension.
///
/// This test is `#[ignore]` because it requires network access to download the
/// model from HuggingFace (~50MB). Run manually with:
/// `cargo test -p colossus-extract constructor_downloads_and_embeds -- --ignored`
#[ignore = "requires network; downloads ~50MB model from HuggingFace"]
#[test]
fn constructor_downloads_and_embeds() {
    use colossus_extract::FastembedProvider;
    use fastembed::EmbeddingModel;

    let provider =
        FastembedProvider::new_from_huggingface(EmbeddingModel::AllMiniLML6V2, None).unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(provider.embed("Hello world")).unwrap();

    assert_eq!(
        result.len(),
        384,
        "AllMiniLML6V2 should produce 384-dim vectors"
    );
    assert!(
        result.iter().all(|v| v.is_finite()),
        "All embedding values should be finite (no NaN or Inf)"
    );
}

/// Verifies that `model_name()` and `dimensions()` return expected values for
/// `AllMiniLML6V2`.
///
/// This test is `#[ignore]` because it requires network access.
#[ignore = "requires network"]
#[test]
fn model_name_and_dimensions_match_expected() {
    use colossus_extract::FastembedProvider;
    use fastembed::EmbeddingModel;

    let provider =
        FastembedProvider::new_from_huggingface(EmbeddingModel::AllMiniLML6V2, None).unwrap();

    assert!(
        provider.model_name().contains("MiniLM"),
        "model_name should contain 'MiniLM', got: {}",
        provider.model_name()
    );
    assert_eq!(provider.dimensions(), 384);
}
