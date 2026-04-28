//! vLLM OpenAI-compatible Embeddings provider — direct reqwest implementation.
//!
//! ## Why direct reqwest, not an OpenAI SDK
//!
//! Matches the design of the sibling `VllmProvider` (Chat Completions). There is
//! no official Rust OpenAI SDK from OpenAI, and the community `async-openai`
//! crate adds weight for a feature set we do not need. Using reqwest directly
//! keeps consistency across providers and avoids an extra dependency.
//!
//! ## Error taxonomy
//!
//! | HTTP status | Error variant | Caller action |
//! |-------------|--------------|---------------|
//! | 429 (proxied deployment) | `PipelineError::RateLimited { retry_after_secs }` | Wait precisely |
//! | 500 (server/model crash) | `PipelineError::LlmProvider` | Short exponential backoff |
//! | 400/422 (validation) | `PipelineError::LlmProvider` | Do NOT retry — caller bug |
//! | Network/timeout | `PipelineError::LlmProvider` | Caller decides |
//!
//! ## Why caller supplies `dimensions`
//!
//! The constructor takes a `dimensions: u32` argument and the provider verifies
//! every response matches. vLLM does not expose a dimensions query endpoint, and
//! Qdrant collections are created with a fixed dimension — a silent mismatch
//! would corrupt the index. The OpenAI API reference lists `dimensions` as a
//! canonical request parameter, but we intentionally do NOT send it: most vLLM
//! models don't support dimension reduction, and sending it would introduce
//! failure modes on compliant servers. Verification-on-response is strictly a
//! guard against operator misconfiguration.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::PipelineError;
use crate::traits::EmbeddingProvider;

/// OpenAI-compatible Embeddings endpoint path, confirmed by docs.vllm.ai.
const EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// Default request timeout when the caller does not specify one — matches
/// sibling `VllmProvider`.
///
/// 10 minutes accommodates large batch embedding calls. Callers that need a
/// different value pass it via the `request_timeout_secs` constructor parameter.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 600;

/// TCP keep-alive interval in seconds — matches sibling `VllmProvider`.
///
/// Corporate firewalls/NAT may drop idle connections even on LAN.
const TCP_KEEPALIVE_SECS: u64 = 60;

/// Fallback when 429 comes via proxy without retry-after header.
///
/// vLLM itself does not emit 429; this handles the case where vLLM sits
/// behind a rate-limiting proxy (e.g., LiteLLM).
const DEFAULT_RETRY_AFTER_SECS: u64 = 60;

/// OpenAI Embeddings response envelope.
///
/// Private wire-format struct — not part of the public API.
#[derive(Deserialize)]
struct EmbeddingResponse {
    /// Embedding entries, one per input.
    data: Vec<EmbeddingData>,
    /// Echo of the requested model identifier.
    #[allow(dead_code)]
    model: String,
    /// Token usage metadata, if present.
    #[allow(dead_code)]
    usage: Option<EmbeddingUsage>,
}

/// A single embedding entry.
///
/// Private wire-format struct. The `index` field corresponds to the request's
/// input position — the OpenAI spec permits servers to return entries in any
/// order, so batch consumers must sort by this field.
#[derive(Deserialize)]
struct EmbeddingData {
    /// The embedding vector for the input at `index`.
    embedding: Vec<f32>,
    /// Position of the corresponding input in the request.
    index: u32,
}

/// Token usage metadata from an OpenAI-compatible embeddings endpoint.
///
/// Private wire-format struct.
#[derive(Deserialize)]
struct EmbeddingUsage {
    /// Input tokens consumed.
    #[allow(dead_code)]
    prompt_tokens: u32,
    /// Total tokens for this request (equal to `prompt_tokens` for embeddings).
    #[allow(dead_code)]
    total_tokens: u32,
}

/// vLLM OpenAI-compatible Embeddings provider using direct reqwest HTTP calls.
///
/// ## Rust Learning: `Option<String>` for optional authentication
///
/// `api_key: Option<String>` uses `None` to mean "no authentication" — matching
/// vLLM's optional `--api-key` flag and the sibling `VllmProvider`. The anti-pattern
/// of using an empty string to mean "no auth" produces a malformed
/// `Authorization: Bearer ` header with an empty token.
pub struct VllmEmbeddingProvider {
    /// vLLM server base URL without `/v1` suffix (e.g., `"http://host:8000"`).
    base_url: String,
    /// Model identifier served by this vLLM instance.
    model: String,
    /// API key for authenticated deployments, or `None` for local/unauth servers.
    api_key: Option<String>,
    /// Embedding vector dimension, captured at construction time and verified on
    /// every response.
    dimensions: u32,
    /// Reusable HTTP client with connection pool, timeout, and keep-alive.
    http_client: reqwest::Client,
    /// Request timeout in seconds, retained for diagnostic logging and error
    /// messages so operators can see the configured value at the call site.
    request_timeout_secs: u64,
}

impl VllmEmbeddingProvider {
    /// Create a new vLLM embedding provider.
    ///
    /// - `base_url`: vLLM server root, e.g., `"http://colossus-dev-app1:8000"`.
    ///   Must NOT include `/v1` suffix or trailing slash — the provider appends
    ///   `/v1/embeddings` automatically.
    /// - `model`: Embedding model identifier served by vLLM
    ///   (e.g., `"nomic-embed-text-v1.5"`).
    /// - `api_key`: `Some(key)` for authenticated deployments (vLLM `--api-key`),
    ///   `None` for local servers without authentication.
    /// - `dimensions`: Expected embedding vector dimension. Must equal the
    ///   Qdrant collection configuration. Every response is verified against
    ///   this value to catch operator misconfiguration before silent index
    ///   corruption.
    /// - `request_timeout_secs`: Optional reqwest client timeout in seconds.
    ///   `None` uses [`DEFAULT_REQUEST_TIMEOUT_SECS`] (600). Callers issuing
    ///   large batch requests should pass a longer timeout.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` if `base_url` ends with `/v1` or a
    /// trailing slash (common footgun), or if the reqwest client fails to build.
    pub fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        dimensions: u32,
        request_timeout_secs: Option<u64>,
    ) -> Result<Self, PipelineError> {
        if base_url.ends_with('/') || base_url.ends_with("/v1") {
            return Err(PipelineError::LlmProvider(format!(
                "base_url must NOT include /v1 suffix or trailing slash; got '{base_url}'"
            )));
        }

        let request_timeout_secs = request_timeout_secs.unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS);
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(request_timeout_secs))
            .tcp_keepalive(Duration::from_secs(TCP_KEEPALIVE_SECS))
            .build()
            .map_err(|e| {
                PipelineError::LlmProvider(format!(
                    "Failed to build vLLM embedding HTTP client: {e}"
                ))
            })?;

        Ok(Self {
            base_url,
            model,
            api_key,
            dimensions,
            http_client,
            request_timeout_secs,
        })
    }

    /// Issue a POST to the embeddings endpoint and parse the response envelope.
    ///
    /// Centralizes HTTP concerns shared between `embed` and `embed_batch`:
    /// URL construction, bearer-auth injection, timeout mapping, 429 retry-after
    /// parsing, non-success status categorization, and response JSON parsing.
    /// Returns the parsed `EmbeddingResponse`; semantic validation (dimension
    /// check, index ordering, count match) lives in the `validate_*` free
    /// functions so it can be unit-tested without HTTP.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::RateLimited` on HTTP 429 with retry-after, or
    /// `PipelineError::LlmProvider` on timeout, network failure, non-success
    /// status, or response parse failure.
    async fn post_embeddings(
        &self,
        body: serde_json::Value,
    ) -> Result<EmbeddingResponse, PipelineError> {
        let url = format!("{}{EMBEDDINGS_PATH}", self.base_url);

        let mut request = self.http_client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await.map_err(|e| {
            if e.is_timeout() {
                let timeout_secs = self.request_timeout_secs;
                tracing::warn!(
                    model = %self.model,
                    timeout_secs,
                    "vLLM embedding request timed out"
                );
                PipelineError::LlmProvider(format!(
                    "request timeout after {timeout_secs}s: {e}"
                ))
            } else {
                PipelineError::LlmProvider(format!("network error: {e}"))
            }
        })?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(DEFAULT_RETRY_AFTER_SECS);

            tracing::warn!(
                model = %self.model,
                provider = "vllm-embed",
                retry_after_secs,
                "vLLM embedding rate limited (429)"
            );

            return Err(PipelineError::RateLimited { retry_after_secs });
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let category = if status.as_u16() == 400 || status.as_u16() == 422 {
                "vllm validation error"
            } else if status.is_server_error() {
                "vllm server error"
            } else if status.is_client_error() {
                "client error"
            } else {
                "server error"
            };
            return Err(PipelineError::LlmProvider(format!(
                "{category} HTTP {status}: {}",
                &body[..body.len().min(300)]
            )));
        }

        response.json().await.map_err(|e| {
            PipelineError::LlmProvider(format!("Failed to parse vLLM embedding response: {e}"))
        })
    }
}

/// Validate a single-embedding response and extract the embedding vector.
///
/// Pure function: no I/O, no provider state. Takes the parsed wire-format
/// response and the expected dimensionality. Returns the embedding vector
/// on success, or a `PipelineError::LlmProvider` describing the precise
/// failure mode: empty response data, or embedding length disagreeing with
/// `expected_dims`.
///
/// ## Why a free function
///
/// This validator operates on `EmbeddingResponse` (a wire struct), not on
/// `VllmEmbeddingProvider` state. Per Rust community practice
/// (users.rust-lang.org/t/70949), associated functions are for items
/// inherently about a type; free functions are appropriate when the logic
/// is about a different type (here the wire format). Keeping this pure
/// enables unit testing without an HTTP mock, and preserves the option to
/// reuse it across future OpenAI-compatible providers.
///
/// # Errors
///
/// Returns `PipelineError::LlmProvider` on empty data or dimension mismatch.
fn validate_single(
    response: EmbeddingResponse,
    expected_dims: u32,
) -> Result<Vec<f32>, PipelineError> {
    let entry = response
        .data
        .into_iter()
        .next()
        .ok_or_else(|| PipelineError::LlmProvider("vLLM returned no embedding data".into()))?;

    if entry.embedding.len() as u32 != expected_dims {
        return Err(PipelineError::LlmProvider(format!(
            "vLLM returned {}-dim embedding but provider configured for {}-dim \
             — check EMBEDDING_DIMENSIONS env var",
            entry.embedding.len(),
            expected_dims
        )));
    }

    Ok(entry.embedding)
}

/// Validate a batch-embedding response and extract embedding vectors in
/// request-order.
///
/// Pure function: no I/O, no provider state. Performs three checks:
/// response count matches request count, indexes are contiguous from 0
/// after sorting, and every embedding length equals `expected_dims`.
///
/// ## Why sort by index
///
/// The OpenAI embeddings spec permits servers to return `data` entries in
/// any order; each entry carries its request-position index. Correct
/// consumers must sort by `index` rather than assume response-order equals
/// request-order.
///
/// # Errors
///
/// Returns `PipelineError::LlmProvider` on count mismatch, non-contiguous
/// indexes after sort, or any per-entry dimension mismatch.
fn validate_batch(
    response: EmbeddingResponse,
    expected_count: usize,
    expected_dims: u32,
) -> Result<Vec<Vec<f32>>, PipelineError> {
    if response.data.len() != expected_count {
        return Err(PipelineError::LlmProvider(format!(
            "vLLM returned {} embeddings for {} inputs — server dropped or duplicated entries",
            response.data.len(),
            expected_count
        )));
    }

    let mut data = response.data;
    data.sort_by_key(|d| d.index);

    let mut results = Vec::with_capacity(data.len());
    for (i, entry) in data.into_iter().enumerate() {
        if entry.index != i as u32 {
            return Err(PipelineError::LlmProvider(format!(
                "vLLM response indexes not contiguous from 0: got index {} at position {}",
                entry.index, i
            )));
        }
        if entry.embedding.len() as u32 != expected_dims {
            return Err(PipelineError::LlmProvider(format!(
                "vLLM returned {}-dim embedding but provider configured for {}-dim \
                 — check EMBEDDING_DIMENSIONS env var",
                entry.embedding.len(),
                expected_dims
            )));
        }
        results.push(entry.embedding);
    }

    Ok(results)
}

#[async_trait]
impl EmbeddingProvider for VllmEmbeddingProvider {
    /// Embed a single text string via the vLLM Embeddings endpoint.
    ///
    /// Thin composer: builds the request body, delegates HTTP transport to
    /// `post_embeddings`, delegates response validation to `validate_single`.
    ///
    /// # Errors
    ///
    /// See `post_embeddings` (HTTP/transport errors) and `validate_single`
    /// (empty data, dimension mismatch).
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });
        let response = self.post_embeddings(body).await?;
        validate_single(response, self.dimensions)
    }

    /// Embed multiple texts in a single vLLM Embeddings batch request.
    ///
    /// Thin composer: builds the request body with `input` as the slice of
    /// texts, delegates HTTP transport to `post_embeddings`, delegates
    /// response validation (including OpenAI-spec index sorting) to
    /// `validate_batch`. Overrides the trait's default serial implementation
    /// to issue one HTTP request carrying the whole batch.
    ///
    /// # Errors
    ///
    /// See `post_embeddings` (HTTP/transport errors) and `validate_batch`
    /// (count mismatch, non-contiguous indexes, dimension mismatch).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, PipelineError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });
        let response = self.post_embeddings(body).await?;
        validate_batch(response, texts.len(), self.dimensions)
    }

    /// Returns the embedding vector dimension captured at construction time.
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Returns the model identifier passed to the constructor.
    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build an EmbeddingResponse with a specific set of entries.
    fn make_response(entries: Vec<(u32, Vec<f32>)>) -> EmbeddingResponse {
        EmbeddingResponse {
            data: entries
                .into_iter()
                .map(|(index, embedding)| EmbeddingData { embedding, index })
                .collect(),
            model: "test-model".to_string(),
            usage: None,
        }
    }

    /// validate_single returns the embedding vector when the response has
    /// one entry of the expected dimensionality.
    #[test]
    fn validate_single_returns_embedding_on_valid_response() {
        let response = make_response(vec![(0, vec![0.1, 0.2, 0.3, 0.4])]);
        let result = validate_single(response, 4).unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3, 0.4]);
    }

    /// validate_single errors with a clear message when data is empty.
    #[test]
    fn validate_single_errors_on_empty_data() {
        let response = make_response(vec![]);
        let err = validate_single(response, 4).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("no embedding data"), "got: {msg}");
    }

    /// validate_single errors and mentions EMBEDDING_DIMENSIONS when the
    /// returned embedding length disagrees with expected_dims.
    #[test]
    fn validate_single_errors_on_dim_mismatch() {
        let response = make_response(vec![(0, vec![0.1, 0.2, 0.3])]);
        let err = validate_single(response, 4).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("EMBEDDING_DIMENSIONS"), "got: {msg}");
        assert!(msg.contains('3'), "got: {msg}");
        assert!(msg.contains('4'), "got: {msg}");
    }

    /// validate_batch returns embeddings in index order even when the
    /// response `data` array is unsorted (server may reorder per spec).
    #[test]
    fn validate_batch_returns_embeddings_in_index_order() {
        let response = make_response(vec![
            (2, vec![2.0, 2.0]),
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 1.0]),
        ]);
        let result = validate_batch(response, 3, 2).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![0.0, 0.0]);
        assert_eq!(result[1], vec![1.0, 1.0]);
        assert_eq!(result[2], vec![2.0, 2.0]);
    }

    /// validate_batch errors when data.len() disagrees with expected_count.
    #[test]
    fn validate_batch_errors_on_count_mismatch() {
        let response = make_response(vec![(0, vec![0.0, 0.0]), (1, vec![1.0, 1.0])]);
        let err = validate_batch(response, 3, 2).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("dropped or duplicated"), "got: {msg}");
    }

    /// validate_batch errors when indexes are non-contiguous after sorting
    /// (e.g., 0, 1, 3 with 2 missing).
    #[test]
    fn validate_batch_errors_on_non_contiguous_indexes() {
        let response = make_response(vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 1.0]),
            (3, vec![3.0, 3.0]),
        ]);
        let err = validate_batch(response, 3, 2).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not contiguous"), "got: {msg}");
    }

    /// A non-default `request_timeout_secs` must be accepted without error
    /// — covers the new constructor path added by Phase 1c.
    #[test]
    fn provider_new_accepts_custom_timeout() {
        let provider = VllmEmbeddingProvider::new(
            "http://localhost:8000".to_string(),
            "test-embed".to_string(),
            None,
            768,
            Some(1800),
        );
        assert!(provider.is_ok(), "Custom timeout must be accepted");
    }

    /// validate_batch errors if any single entry has the wrong dimension,
    /// even when all other entries are correct.
    #[test]
    fn validate_batch_errors_on_dim_mismatch_any_entry() {
        let response = make_response(vec![
            (0, vec![0.0, 0.0, 0.0, 0.0]),
            (1, vec![1.0, 1.0, 1.0]),
            (2, vec![2.0, 2.0, 2.0, 2.0]),
        ]);
        let err = validate_batch(response, 3, 4).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("EMBEDDING_DIMENSIONS"), "got: {msg}");
    }
}
