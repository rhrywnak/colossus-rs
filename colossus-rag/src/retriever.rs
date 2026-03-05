//! QdrantRetriever — implements [`VectorRetriever`] using rig-fastembed + qdrant-client.
//!
//! This module provides the first real implementation of the RAG retriever stage.
//! It replaces two hand-rolled services from colossus-legal:
//! - `embedding_service.rs` (~130 lines) — embedded text via fastembed directly
//! - `qdrant_service.rs` (~120 lines) — searched Qdrant via raw HTTP/reqwest
//!
//! ## Architecture: Why rig-fastembed + qdrant-client (not rig-qdrant)?
//!
//! We use **rig-fastembed** for embedding queries (wraps fastembed behind Rig's
//! `EmbeddingModel` trait) and **qdrant-client** directly for vector search.
//!
//! We chose NOT to use rig-qdrant's `QdrantVectorStore` because:
//! 1. rig-qdrant's `VectorStoreIndex::top_n()` returns payloads as generic `T`,
//!    requiring us to deserialize into `serde_json::Value` and extract fields anyway.
//! 2. Using qdrant-client directly gives us full control over `SearchPoints`:
//!    filters, score thresholds, payload selection — without fighting an abstraction.
//! 3. qdrant-client's `ScoredPoint` exposes the payload as a `HashMap<String, Value>`
//!    (protobuf Value), which we can extract fields from directly.
//!
//! ## Rust Learning: Feature-gated modules
//!
//! This entire module is conditionally compiled with:
//! ```rust,ignore
//! #[cfg(all(feature = "qdrant", feature = "fastembed"))]
//! mod retriever;
//! ```
//! If a consumer doesn't enable both features, this module doesn't exist in the
//! compiled binary — zero cost. The `#[cfg]` gate is in `lib.rs`, not here.
//!
//! ## Rig Concept: EmbeddingModel trait
//!
//! `rig::embeddings::EmbeddingModel` defines `embed_text(&self, text) -> Result<Embedding, _>`.
//! An `Embedding` contains `vec: Vec<f64>` — note f64, not f32. Rig uses f64 internally
//! even though fastembed produces f32. We convert back to f32 for Qdrant's search API.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

/// ## Rust Learning: Re-importing the trait to call its methods
///
/// We need `rig::embeddings::EmbeddingModel` in scope to call `.embed_text()`
/// on the concrete fastembed model. Without this import, Rust can't resolve
/// the method — even though the concrete type implements the trait.
use rig::embeddings::EmbeddingModel;

use qdrant_client::qdrant::{
    value::Kind, Condition, Filter, ScoredPoint, SearchPointsBuilder,
};
use qdrant_client::Qdrant;

use crate::error::RagError;
use crate::traits::VectorRetriever;
use crate::types::{ContextChunk, ScopeFilter, ScopeFilterType, SourceReference};

// ---------------------------------------------------------------------------
// QdrantRetriever struct
// ---------------------------------------------------------------------------

/// Searches Qdrant for context chunks relevant to a query.
///
/// Combines rig-fastembed for embedding with qdrant-client for vector search.
/// This is the concrete implementation of [`VectorRetriever`] for use with
/// a Qdrant vector database and local ONNX-based embeddings.
///
/// ## Rust Learning: `Arc` for shared ownership
///
/// Both `embedding_model` and `qdrant_client` are wrapped in `Arc` (Atomic
/// Reference Counted pointer). This lets multiple parts of the application
/// share the same model/client without cloning the heavy inner data.
///
/// `Arc<T>` is `Clone + Send + Sync` when `T` is `Send + Sync`, which means
/// it's safe to share across async tasks and threads — exactly what Axum
/// needs for handler state.
///
/// ## Rust Learning: Concrete type vs `dyn Trait`
///
/// We store the embedding model as a concrete `rig_fastembed::EmbeddingModel`
/// rather than `dyn rig::embeddings::EmbeddingModel`. Why?
///
/// Rig's `EmbeddingModel` trait has generic methods (like `embed_text` which
/// returns `impl Future`), making it NOT dyn-compatible. You can't create
/// `Box<dyn EmbeddingModel>`. So we use the concrete type directly.
///
/// If we needed to support multiple embedding providers, we'd make
/// `QdrantRetriever` generic: `QdrantRetriever<E: EmbeddingModel>`.
/// But for now, fastembed is our only provider, so concrete is simpler.
pub struct QdrantRetriever {
    /// The fastembed embedding model (nomic-embed-text v1.5).
    /// Shared via Arc because model initialization is expensive (~100MB ONNX download).
    embedding_model: Arc<rig_fastembed::EmbeddingModel>,

    /// The Qdrant gRPC client. Shared via Arc for reuse across requests.
    qdrant_client: Arc<Qdrant>,

    /// Which Qdrant collection to search (e.g., "colossus_evidence").
    collection_name: String,

    /// Minimum similarity score for results. Points below this threshold
    /// are filtered out by Qdrant before returning results.
    /// Cosine similarity range: -1.0 to 1.0 (0.0 = no similarity).
    default_score_threshold: f32,
}

impl QdrantRetriever {
    /// Create a new QdrantRetriever.
    ///
    /// ## Parameters
    /// - `embedding_model`: A rig-fastembed model (e.g., NomicEmbedTextV15)
    /// - `qdrant_client`: A connected Qdrant gRPC client (port 6334)
    /// - `collection_name`: The Qdrant collection to search
    /// - `default_score_threshold`: Minimum cosine similarity score (0.0–1.0)
    pub fn new(
        embedding_model: Arc<rig_fastembed::EmbeddingModel>,
        qdrant_client: Arc<Qdrant>,
        collection_name: impl Into<String>,
        default_score_threshold: f32,
    ) -> Self {
        Self {
            embedding_model,
            qdrant_client,
            collection_name: collection_name.into(),
            default_score_threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// VectorRetriever trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl VectorRetriever for QdrantRetriever {
    /// Embed the query, search Qdrant, and return matching context chunks.
    ///
    /// ## Flow
    /// 1. Embed the query text using rig-fastembed → `Vec<f64>`
    /// 2. Convert `f64` → `f32` (Qdrant uses f32 vectors)
    /// 3. Convert `ScopeFilter` slice → optional Qdrant `Filter`
    /// 4. Build and execute `SearchPoints` request
    /// 5. Map each `ScoredPoint` → `ContextChunk`
    async fn search(
        &self,
        query: &str,
        limit: usize,
        filters: &[ScopeFilter],
    ) -> Result<Vec<ContextChunk>, RagError> {
        // --- Step 1: Embed the query text ---
        //
        // ## Rig Concept: embed_text() returns Embedding { vec: Vec<f64> }
        //
        // rig-fastembed runs ONNX inference on the CPU synchronously, but wraps
        // it in an async interface for consistency with remote embedding providers.
        let embedding = self
            .embedding_model
            .embed_text(query)
            .await
            .map_err(|e| RagError::EmbeddingError(e.to_string()))?;

        // --- Step 2: Convert f64 → f32 ---
        //
        // Rig internally uses f64 for embedding vectors, but Qdrant's gRPC API
        // expects f32. The precision loss is negligible for similarity search —
        // cosine similarity is scale-invariant, and the 7 significant digits of
        // f32 are more than enough for ranking.
        let vector_f32: Vec<f32> = embedding.vec.iter().map(|&v| v as f32).collect();

        // --- Step 3: Build Qdrant filter from ScopeFilters ---
        let qdrant_filter = scope_filters_to_qdrant_filter(filters);

        // --- Step 4: Build and execute SearchPoints ---
        //
        // ## Rig Concept: SearchPointsBuilder
        //
        // qdrant-client provides a builder pattern for search requests.
        // `SearchPointsBuilder::new(collection, vector, limit)` sets the three
        // required fields. Optional methods chain on: `.filter()`, `.with_payload()`,
        // `.score_threshold()`.
        //
        // `.with_payload(true)` tells Qdrant to return all payload fields with
        // each result. Without this, we'd only get point IDs and scores.
        let mut search_builder = SearchPointsBuilder::new(
            &self.collection_name,
            vector_f32,
            limit as u64,
        )
        .with_payload(true)
        .score_threshold(self.default_score_threshold);

        if let Some(filter) = qdrant_filter {
            search_builder = search_builder.filter(filter);
        }

        let search_response = self
            .qdrant_client
            .search_points(search_builder)
            .await
            .map_err(|e| RagError::SearchError(e.to_string()))?;

        // --- Step 5: Map ScoredPoints → ContextChunks ---
        let chunks: Vec<ContextChunk> = search_response
            .result
            .into_iter()
            .map(scored_point_to_context_chunk)
            .collect();

        tracing::debug!(
            query = query,
            results = chunks.len(),
            "QdrantRetriever search completed"
        );

        Ok(chunks)
    }
}

// ---------------------------------------------------------------------------
// Helper: ScopeFilter → Qdrant Filter conversion
// ---------------------------------------------------------------------------

/// Convert a slice of [`ScopeFilter`]s into an optional Qdrant [`Filter`].
///
/// Each supported filter type maps to a Qdrant `Condition::matches()` on the
/// corresponding payload field. Multiple filters are combined with AND logic
/// (`Filter::must`).
///
/// ## Unsupported filter types
///
/// `Person` and `Collection` filters are logged as warnings and skipped,
/// because the Qdrant `colossus_evidence` collection has no `person` or
/// `collection` payload fields. Person filtering relies on semantic matching
/// in the query text itself.
///
/// ## Rust Learning: `Condition::matches(field, value)`
///
/// This is qdrant-client's builder for payload field matching. Under the hood,
/// a string value becomes `MatchValue::Keyword("value")`, which does an exact
/// match on a keyword-indexed payload field. For multi-value matching (OR),
/// you'd pass `Vec<String>` which becomes `MatchValue::Keywords`.
pub fn scope_filters_to_qdrant_filter(filters: &[ScopeFilter]) -> Option<Filter> {
    if filters.is_empty() {
        return None;
    }

    let mut conditions: Vec<Condition> = Vec::new();

    for filter in filters {
        match filter.filter_type {
            ScopeFilterType::Document => {
                // Match points where payload "document_id" equals the filter value.
                conditions.push(Condition::matches(
                    "document_id",
                    filter.value.clone(),
                ));
            }
            ScopeFilterType::NodeType => {
                // Match points where payload "node_type" equals the filter value.
                conditions.push(Condition::matches(
                    "node_type",
                    filter.value.clone(),
                ));
            }
            ScopeFilterType::Person => {
                // No "person" field in Qdrant payloads — skip with a warning.
                // The person name in the query text provides semantic filtering.
                tracing::warn!(
                    filter_type = "person",
                    value = filter.value,
                    "Person filter not supported in Qdrant — relying on semantic match"
                );
            }
            ScopeFilterType::Collection => {
                // No "collection" field in Qdrant payloads — skip with a warning.
                tracing::warn!(
                    filter_type = "collection",
                    value = filter.value,
                    "Collection filter not supported in Qdrant — skipping"
                );
            }
        }
    }

    if conditions.is_empty() {
        return None;
    }

    // ## Rig Concept: Filter::must() = AND logic
    //
    // `Filter::must(conditions)` requires ALL conditions to match.
    // If we had OR logic needs, we'd use `Filter::should(conditions)`.
    Some(Filter::must(conditions))
}

// ---------------------------------------------------------------------------
// Helper: ScoredPoint → ContextChunk mapping
// ---------------------------------------------------------------------------

/// Convert a Qdrant [`ScoredPoint`] into a [`ContextChunk`].
///
/// Extracts payload fields by name from the protobuf `HashMap<String, Value>`.
/// Missing fields default to empty strings or None.
///
/// ## Rust Learning: Protobuf `Value` vs `serde_json::Value`
///
/// qdrant-client uses protobuf-generated types, not serde_json. The payload
/// values are `qdrant_client::qdrant::Value` with a `kind` field that's an
/// enum: `Kind::StringValue(String)`, `Kind::IntegerValue(i64)`, etc.
///
/// This is different from the spike (which used rig-qdrant and got
/// `serde_json::Value` payloads). Here we pattern-match on `Kind` directly.
fn scored_point_to_context_chunk(point: ScoredPoint) -> ContextChunk {
    let payload = &point.payload;

    ContextChunk {
        node_id: extract_string(payload, "node_id"),
        node_type: extract_string(payload, "node_type"),
        title: extract_string(payload, "title"),
        // Qdrant stores the embedded text in the "document" field within the
        // payload. If not present, fall back to the title as content.
        content: extract_string_or(payload, "document", &extract_string(payload, "title")),
        score: point.score,
        source: SourceReference {
            document_title: extract_optional_string(payload, "title"),
            document_id: extract_optional_string(payload, "document_id"),
            page_number: extract_optional_u32(payload, "page_number"),
            verbatim_quote: None,
        },
        // Relationships come from graph expansion (Neo4j), not vector search.
        relationships: Vec::new(),
        metadata: serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Payload extraction helpers
// ---------------------------------------------------------------------------

/// Extract a string value from a Qdrant protobuf payload, returning "" if missing.
///
/// ## Rust Learning: Pattern matching on nested enums
///
/// The payload is `HashMap<String, Value>` where `Value.kind` is `Option<Kind>`.
/// We chain `.get()` → `.and_then()` → pattern match on `Kind::StringValue`.
/// This is Rust's version of null-safe chaining (like Java's Optional or Kotlin's `?.`).
fn extract_string(payload: &HashMap<String, qdrant_client::qdrant::Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|kind| match kind {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Extract a string value, returning a fallback if the field is missing.
fn extract_string_or(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
    fallback: &str,
) -> String {
    let value = extract_string(payload, key);
    if value.is_empty() {
        fallback.to_string()
    } else {
        value
    }
}

/// Extract an optional string value from a Qdrant protobuf payload.
fn extract_optional_string(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> Option<String> {
    let value = extract_string(payload, key);
    if value.is_empty() { None } else { Some(value) }
}

/// Extract an optional u32 from a Qdrant protobuf payload.
///
/// The `page_number` field in our Qdrant collection is stored as a string
/// (from the original ingestion), so we try StringValue first and parse it,
/// then fall back to IntegerValue.
fn extract_optional_u32(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> Option<u32> {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|kind| match kind {
            Kind::StringValue(s) => s.parse::<u32>().ok(),
            Kind::IntegerValue(i) => u32::try_from(*i).ok(),
            _ => None,
        })
}
