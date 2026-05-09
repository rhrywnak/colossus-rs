//! Tests for EmbeddingReranker - unit tests covering provider interaction
//! and cosine similarity filtering behavior.
//!
//! Uses a Mutex-guarded TestEmbeddingProvider stub that captures embed() calls
//! for interaction assertions. This is the same pattern established in
//! synthesizer_tests.rs and decomposer_tests.rs for P3-1-Followup-Amend onward.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use colossus_extract::{EmbeddingProvider, PipelineError};
use colossus_rag::{ContextChunk, EmbeddingReranker, SourceReference};

// ---------------------------------------------------------------------------
// TestEmbeddingProvider — capture-capable stub
// ---------------------------------------------------------------------------

struct TestEmbeddingProvider {
    /// Pre-canned response returned by embed(). A single entry suffices for
    /// the current test suite: every call returns the same vector (or error).
    response: Mutex<Result<Vec<f32>, String>>,
    /// All embed() calls received, in order.
    embed_calls: Mutex<Vec<String>>,
    /// Dimensions this stub reports.
    dims: u32,
}

impl TestEmbeddingProvider {
    /// Create a stub that returns the same embedding for every call.
    fn new_fixed(vector: Vec<f32>, dims: u32) -> Self {
        Self {
            response: Mutex::new(Ok(vector)),
            embed_calls: Mutex::new(Vec::new()),
            dims,
        }
    }

    /// Create a stub that always errors.
    fn new_err(msg: &str, dims: u32) -> Self {
        Self {
            response: Mutex::new(Err(msg.to_string())),
            embed_calls: Mutex::new(Vec::new()),
            dims,
        }
    }

    /// Number of embed() calls received.
    fn embed_call_count(&self) -> usize {
        self.embed_calls.lock().unwrap().len()
    }

    /// All embed() call texts received, in order.
    #[allow(dead_code)]
    fn embed_call_texts(&self) -> Vec<String> {
        self.embed_calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl EmbeddingProvider for TestEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        self.embed_calls.lock().unwrap().push(text.to_string());
        match &*self.response.lock().unwrap() {
            Ok(v) => Ok(v.clone()),
            Err(msg) => Err(PipelineError::LlmProvider(msg.clone())),
        }
    }

    fn dimensions(&self) -> u32 {
        self.dims
    }

    fn model_name(&self) -> &str {
        "test-embedder"
    }
}

// ---------------------------------------------------------------------------
// Helper: build a ContextChunk with a given score
// ---------------------------------------------------------------------------

fn chunk(id: &str, content: &str, score: f32) -> ContextChunk {
    ContextChunk {
        node_id: id.into(),
        node_type: "Evidence".into(),
        title: format!("Chunk {id}"),
        content: content.into(),
        score,
        source: SourceReference::default(),
        relationships: vec![],
        metadata: serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rerank_empty_input_returns_empty() {
    let stub = Arc::new(TestEmbeddingProvider::new_fixed(vec![1.0, 0.0], 2));
    let reranker = EmbeddingReranker::new(stub.clone() as Arc<dyn EmbeddingProvider>, 0.5);

    let (kept, filtered) = reranker.rerank("q", vec![]).await.unwrap();

    assert!(kept.is_empty());
    assert_eq!(filtered, 0);
    assert_eq!(
        stub.embed_call_count(),
        0,
        "no embedding work should be done for zero chunks"
    );
}

#[tokio::test]
async fn test_rerank_only_qdrant_hits_returns_unchanged() {
    let stub = Arc::new(TestEmbeddingProvider::new_fixed(vec![1.0, 0.0], 2));
    let reranker = EmbeddingReranker::new(stub.clone() as Arc<dyn EmbeddingProvider>, 0.5);

    let chunks = vec![
        chunk("a", "alpha", 0.9),
        chunk("b", "beta", 0.8),
        chunk("c", "gamma", 0.7),
    ];

    let (kept, filtered) = reranker.rerank("q", chunks).await.unwrap();

    assert_eq!(kept.len(), 3);
    assert_eq!(filtered, 0);
    assert_eq!(
        stub.embed_call_count(),
        0,
        "no embedding needed when no graph chunks to rerank"
    );
}

#[tokio::test]
async fn test_rerank_only_graph_chunks_embeds_all() {
    let stub = Arc::new(TestEmbeddingProvider::new_fixed(vec![1.0, 0.0], 2));
    let reranker = EmbeddingReranker::new(stub.clone() as Arc<dyn EmbeddingProvider>, 0.5);

    // All graph-expanded (score == 0.0). Identical vectors → similarity 1.0 → all kept.
    let chunks = vec![
        chunk("a", "alpha", 0.0),
        chunk("b", "beta", 0.0),
        chunk("c", "gamma", 0.0),
    ];

    let (kept, filtered) = reranker.rerank("q", chunks).await.unwrap();

    assert_eq!(kept.len(), 3);
    assert_eq!(filtered, 0);
    assert_eq!(
        stub.embed_call_count(),
        4,
        "1 question + 3 chunks = 4 embed calls"
    );
}

#[tokio::test]
async fn test_rerank_filters_below_threshold() {
    let stub = Arc::new(TestEmbeddingProvider::new_fixed(vec![1.0, 0.0], 2));
    // Threshold above 1.0 → even identical vectors (similarity 1.0) filter out.
    let reranker = EmbeddingReranker::new(stub.clone() as Arc<dyn EmbeddingProvider>, 1.1);

    let chunks = vec![
        chunk("a", "alpha", 0.0),
        chunk("b", "beta", 0.0),
        chunk("c", "gamma", 0.0),
    ];
    let chunk_count = chunks.len();

    let (kept, filtered) = reranker.rerank("q", chunks).await.unwrap();

    assert!(kept.is_empty(), "nothing should survive a 1.1 threshold");
    assert_eq!(filtered, chunk_count);
}

#[tokio::test]
async fn test_rerank_mixed_qdrant_and_graph() {
    let stub = Arc::new(TestEmbeddingProvider::new_fixed(vec![1.0, 0.0], 2));
    let reranker = EmbeddingReranker::new(stub.clone() as Arc<dyn EmbeddingProvider>, 0.5);

    // 2 Qdrant hits (score > 0.0) + 2 graph chunks (score == 0.0).
    let chunks = vec![
        chunk("q1", "qdrant one", 0.9),
        chunk("q2", "qdrant two", 0.8),
        chunk("g1", "graph one", 0.0),
        chunk("g2", "graph two", 0.0),
    ];

    let (kept, filtered) = reranker.rerank("q", chunks).await.unwrap();

    assert_eq!(kept.len(), 4);
    assert_eq!(filtered, 0);
    assert_eq!(
        stub.embed_call_count(),
        3,
        "1 question + 2 graph chunks; Qdrant hits NOT re-embedded"
    );
}

#[tokio::test]
async fn test_rerank_question_embedding_error_propagates() {
    let stub = Arc::new(TestEmbeddingProvider::new_err("stub failure", 2));
    let reranker = EmbeddingReranker::new(stub as Arc<dyn EmbeddingProvider>, 0.5);

    let chunks = vec![chunk("g1", "graph one", 0.0)];
    let err = reranker.rerank("q", chunks).await.unwrap_err();

    match err {
        colossus_rag::RagError::EmbeddingError(_) => {}
        other => panic!("expected RagError::EmbeddingError, got {other:?}"),
    }
}

#[tokio::test]
async fn test_rerank_uses_embed_not_embed_batch() {
    let stub = Arc::new(TestEmbeddingProvider::new_fixed(vec![1.0, 0.0], 2));
    let reranker = EmbeddingReranker::new(stub.clone() as Arc<dyn EmbeddingProvider>, 0.0);

    let chunks = vec![
        chunk("g1", "graph one", 0.0),
        chunk("g2", "graph two", 0.0),
        chunk("g3", "graph three", 0.0),
    ];

    let (_kept, _filtered) = reranker.rerank("q", chunks).await.unwrap();

    // Regression guard: 4 serial single calls (1 question + 3 chunks),
    // not a single batched call. If a future optimization switches to
    // embed_batch(), this test flags the behavior change.
    assert_eq!(stub.embed_call_count(), 4);
}
