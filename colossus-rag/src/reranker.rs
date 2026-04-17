//! EmbeddingReranker — filters graph-expanded chunks by semantic similarity.
//!
//! After graph expansion returns neighbor nodes, many may be structurally
//! connected but semantically irrelevant to the question. The reranker
//! embeds both the question and each chunk's content, computes cosine
//! similarity, and drops chunks below a threshold.
//!
//! ## Pipeline position
//!
//! ```text
//! Retriever → Expander → [Reranker] → Assembler → Synthesizer
//!                          ^^^^^^^^
//!                          Optional stage — skipped when no reranker is configured
//! ```
//!
//! ## Three-layer filtering
//!
//! - Layer 1 (Qdrant): Vector similarity on the full corpus → ~10 hits
//! - Layer 2 (WP-2): Structural filtering by relationship type → fewer graph nodes
//! - Layer 3 (WP-3, this module): Semantic filtering on expanded nodes → only relevant content
//!
//! ## Rust Learning: Why not a trait?
//!
//! The expander and retriever are traits because they have multiple
//! implementations (Neo4j vs NoOp, Qdrant vs future providers). The
//! reranker has only one implementation (embedding cosine similarity)
//! and is optional in the pipeline. A concrete struct is simpler —
//! the pipeline stores it as `Option<EmbeddingReranker>` and skips
//! the step when `None`. The embedding backend it calls is abstracted
//! via the `EmbeddingProvider` trait, so a single reranker implementation
//! works with any provider (Fastembed, vLLM, future backends).

use std::sync::Arc;

use colossus_extract::EmbeddingProvider;

use crate::error::RagError;
use crate::types::ContextChunk;

// ---------------------------------------------------------------------------
// EmbeddingReranker struct
// ---------------------------------------------------------------------------

/// Reranks expanded context chunks by semantic similarity to the question.
///
/// ## Rust Learning: Arc sharing the embedding provider
///
/// The same `Arc<dyn EmbeddingProvider>` that the QdrantRetriever uses is
/// cloned (cheap Arc reference count bump, not a provider copy) and passed
/// to the reranker. Both stages share the same underlying embedding backend.
pub struct EmbeddingReranker {
    /// The embedding provider (shared with the retriever via Arc).
    /// Can be any implementation of EmbeddingProvider — FastembedProvider,
    /// VllmEmbeddingProvider, or any future backend.
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Minimum cosine similarity for a graph-expanded chunk to be kept.
    /// Chunks below this threshold are filtered out. Typical range: 0.2–0.5.
    threshold: f32,
}

impl EmbeddingReranker {
    /// Create a new reranker with a shared embedding provider and threshold.
    ///
    /// ## Parameters
    /// - `embedding_provider`: The same Arc used by QdrantRetriever (cheap clone)
    /// - `threshold`: Minimum cosine similarity (0.0–1.0) for graph-expanded
    ///   chunks to pass through. Qdrant hits (score > 0.0) always pass.
    pub fn new(embedding_provider: Arc<dyn EmbeddingProvider>, threshold: f32) -> Self {
        Self {
            embedding_provider,
            threshold,
        }
    }

    /// Rerank chunks by cosine similarity to the question.
    ///
    /// Returns `(kept_chunks, filtered_count)`.
    ///
    /// Chunks from vector search (score > 0.0) are ALWAYS kept —
    /// they already passed Qdrant's similarity filter. Only
    /// graph-expanded chunks (score == 0.0) are subject to reranking.
    ///
    /// ## Why keep Qdrant hits unconditionally?
    ///
    /// Qdrant hits have real similarity scores from the vector search.
    /// Re-embedding and re-scoring them would be redundant and could
    /// actually hurt — the Qdrant score was computed against the same
    /// embedding model on the original indexed text, which may be
    /// better quality than the chunk.content we'd embed here.
    pub async fn rerank(
        &self,
        question: &str,
        chunks: Vec<ContextChunk>,
    ) -> Result<(Vec<ContextChunk>, usize), RagError> {
        // Separate into Qdrant hits (always kept) and graph-expanded (subject to filtering).
        let (qdrant_hits, graph_chunks): (Vec<_>, Vec<_>) =
            chunks.into_iter().partition(|c| c.score > 0.0);

        tracing::info!(
            qdrant_hits = qdrant_hits.len(),
            graph_chunks = graph_chunks.len(),
            threshold = self.threshold,
            "Reranker: starting"
        );

        // If no graph chunks to rerank, return everything immediately.
        if graph_chunks.is_empty() {
            return Ok((qdrant_hits, 0));
        }

        // Embed the question.
        let q_vec = self
            .embedding_provider
            .embed(question)
            .await
            .map_err(|e| RagError::EmbeddingError(e.to_string()))?;

        // Embed each graph chunk and filter by cosine similarity.
        let mut kept_graph: Vec<ContextChunk> = Vec::new();
        let mut filtered_count: usize = 0;

        for chunk in graph_chunks {
            let c_vec = self
                .embedding_provider
                .embed(&chunk.content)
                .await
                .map_err(|e| RagError::EmbeddingError(e.to_string()))?;

            let similarity = cosine_similarity(&q_vec, &c_vec);

            if similarity >= self.threshold {
                tracing::debug!(
                    node_id = chunk.node_id,
                    similarity = similarity,
                    "Reranker: kept"
                );
                kept_graph.push(chunk);
            } else {
                tracing::debug!(
                    node_id = chunk.node_id,
                    similarity = similarity,
                    "Reranker: filtered out"
                );
                filtered_count += 1;
            }
        }

        // Combine: Qdrant hits first, then kept graph chunks.
        let mut result = qdrant_hits;
        result.extend(kept_graph);

        tracing::info!(
            kept = result.len(),
            filtered = filtered_count,
            "Reranker: complete"
        );

        Ok((result, filtered_count))
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity helper
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal (unrelated)
/// - -1.0 = opposite direction
///
/// ## Rust Learning: Iterator zip + map + sum
///
/// This is the idiomatic Rust way to compute a dot product:
/// zip pairs up elements, map multiplies them, sum adds them.
/// No index variable, no bounds checking, no off-by-one errors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Identical vectors should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 0.001,
            "Orthogonal vectors should have similarity ~0.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should return 0.0");
    }
}
