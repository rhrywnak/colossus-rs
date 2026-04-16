use async_trait::async_trait;
use colossus_extract::{EmbeddingProvider, LlmProvider, LlmResponse, PipelineError};

/// Verifies that `LlmProvider` is object-safe — it can be used as `dyn LlmProvider`.
///
/// This is a compile-only test. If a future change adds a generic method or
/// returns `Self` from a method, this test will fail to compile and catch the
/// regression immediately. Object safety is required because providers are stored
/// as `Arc<dyn LlmProvider>` in `AppContext`.
///
/// Cost: one bound declaration. Benefit: permanent object-safety guard.
#[test]
fn llm_provider_is_object_safe() {
    let _: Option<Box<dyn LlmProvider>> = None;
}

/// Verifies that `EmbeddingProvider` is object-safe — it can be used as `dyn EmbeddingProvider`.
///
/// Same rationale as `llm_provider_is_object_safe`: embedding providers are stored
/// as `Arc<dyn EmbeddingProvider>` and shared across async tasks. A generic method
/// or `Self` return type would break this. This test catches that at compile time.
#[test]
fn embedding_provider_is_object_safe() {
    let _: Option<Box<dyn EmbeddingProvider>> = None;
}

/// Verifies the full C-COMMON-TRAITS derive set on `LlmResponse`:
///
/// - `Debug` — for `tracing::debug!` and `{:?}` formatting in error messages.
/// - `Clone` — callers need to both record token counts and parse text from the same response.
/// - `PartialEq` + `Eq` — enables `assert_eq!` in tests.
/// - `Serialize` + `Deserialize` — enables logging into `pipeline_events` JSONB payloads.
///
/// Each derive is exercised explicitly so a future removal is caught as a test failure,
/// not discovered when a downstream caller breaks.
#[test]
fn llm_response_common_traits() {
    let response = LlmResponse {
        text: "hello".to_string(),
        input_tokens: Some(10),
        output_tokens: Some(5),
    };

    // Clone + PartialEq + Eq
    let cloned = response.clone();
    assert_eq!(response, cloned);

    // Debug
    let debug_str = format!("{response:?}");
    assert!(debug_str.contains("hello"));

    // Serialize + Deserialize round-trip
    let json = serde_json::to_string(&response).unwrap();
    let deserialized: LlmResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(response, deserialized);
}

/// Verifies that the default `embed_batch()` implementation actually iterates
/// over the input slice and calls `embed()` for each element.
///
/// Without this test, a future change to the default method body could silently
/// break all providers that rely on it (any provider that does not override
/// `embed_batch`). The test uses a trivial `TestEmbedder` that returns a fixed
/// vector, so failures point to the default implementation logic, not to any
/// real provider's behavior.
#[tokio::test]
async fn embedding_provider_default_batch_works() {
    struct TestEmbedder;

    #[async_trait]
    impl EmbeddingProvider for TestEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, PipelineError> {
            Ok(vec![1.0_f32, 2.0, 3.0])
        }

        fn dimensions(&self) -> u32 {
            3
        }

        fn model_name(&self) -> &str {
            "test"
        }
    }

    let embedder = TestEmbedder;
    let result = embedder.embed_batch(&["foo", "bar"]).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], vec![1.0_f32, 2.0, 3.0]);
    assert_eq!(result[1], vec![1.0_f32, 2.0, 3.0]);
}

/// Verifies the exact JSON field names produced by serializing `LlmResponse`.
///
/// This locks in the wire format. Any accidental rename of a struct field
/// (or addition of a `#[serde(rename)]` attribute) would break `pipeline_events`
/// log readers that filter on these JSON keys. The test checks for the presence
/// of each expected key in the serialized output.
#[test]
fn llm_response_serde_field_names() {
    let response = LlmResponse {
        text: "test output".to_string(),
        input_tokens: Some(42),
        output_tokens: Some(17),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"text\""), "JSON must contain field 'text'");
    assert!(
        json.contains("\"input_tokens\""),
        "JSON must contain field 'input_tokens'"
    );
    assert!(
        json.contains("\"output_tokens\""),
        "JSON must contain field 'output_tokens'"
    );
}
