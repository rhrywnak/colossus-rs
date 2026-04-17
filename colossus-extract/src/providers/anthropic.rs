//! Anthropic Messages API provider — direct reqwest implementation.
//!
//! ## Why reqwest, not rig
//!
//! rig's LLM abstraction discards the Anthropic `retry-after` header, which is
//! critical for correct rate-limit handling against Anthropic's token-bucket
//! algorithm. Without the exact retry-after value, any backoff strategy is a
//! guess. The deleted colossus-legal `chunk_extractor.rs` (commit 7aef8ec) was
//! the result of this architectural decision — rig was replaced with direct
//! reqwest to preserve the header.
//!
//! ## Error taxonomy
//!
//! | HTTP status | Error variant | Caller action |
//! |-------------|--------------|---------------|
//! | 429 | `PipelineError::RateLimited { retry_after_secs }` | Wait precisely `retry_after_secs` |
//! | 529 (overloaded) | `PipelineError::LlmProvider` | Short exponential backoff |
//! | 5xx (server) | `PipelineError::LlmProvider` | Short exponential backoff |
//! | 4xx (other) | `PipelineError::LlmProvider` | Do NOT retry — permanent error |
//! | Network/timeout | `PipelineError::LlmProvider` | Caller decides |
//!
//! ## Retry is the caller's responsibility
//!
//! This provider is a thin, stateless HTTP client. The orchestrator (Phase 4
//! `LlmExtract` step) owns retry loops, coordinates with `pipeline_events`
//! progress updates, and knows domain-specific retry budgets. Anthropic's own
//! Go SDK acknowledges both patterns valid and leaves retry to the caller for
//! concurrency-sensitive workloads.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::PipelineError;
use crate::traits::{LlmProvider, LlmResponse};

/// Stable Anthropic Messages API endpoint.
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic API version header. Stable value from Anthropic documentation.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Request timeout in seconds — 10 minutes, matching Anthropic SDK default.
///
/// Anthropic's own Python and TypeScript SDKs default to 10 minute timeouts
/// for non-streaming Messages API requests. Large-context extraction (1M tokens)
/// can legitimately take several minutes. Shorter timeouts risk cutting off
/// valid long requests; longer timeouts leave hung connections alive too long.
const REQUEST_TIMEOUT_SECS: u64 = 600;

/// TCP keep-alive interval in seconds.
///
/// Some networks drop idle connections; Anthropic's docs recommend TCP
/// keep-alive to prevent silent timeout during long responses. reqwest's
/// `ClientBuilder::tcp_keepalive` enables this.
const TCP_KEEPALIVE_SECS: u64 = 60;

/// Fallback retry-after value if 429 response lacks the header.
///
/// Anthropic documents that the `retry-after` header is always present on 429,
/// but we default safely in case of API changes or proxy stripping. 60s is the
/// typical rate-limit window.
const DEFAULT_RETRY_AFTER_SECS: u64 = 60;

/// Anthropic content block kind discriminator for text blocks.
///
/// Other kinds (e.g., `tool_use`) are ignored by this provider.
const BLOCK_KIND_TEXT: &str = "text";

/// Anthropic Messages API response envelope.
///
/// Private wire-format struct — not part of the public API.
#[derive(Deserialize)]
struct AnthropicResponse {
    /// Content blocks returned by the model.
    content: Vec<AnthropicContentBlock>,
    /// Token usage metadata, if present.
    usage: Option<AnthropicUsage>,
}

/// A single content block in an Anthropic response.
///
/// Private wire-format struct. The `kind` field distinguishes text from tool_use
/// and other block types; only text blocks are consumed by this provider.
///
/// ## Rust Learning: `#[serde(rename = "type")]`
///
/// The Anthropic API returns a field called `type` in each content block, but
/// `type` is a reserved keyword in Rust. `#[serde(rename)]` tells serde to
/// read the JSON key `"type"` into the Rust field `kind`, avoiding the keyword
/// conflict without requiring raw identifiers (`r#type`).
#[derive(Deserialize)]
struct AnthropicContentBlock {
    /// Block type: "text", "tool_use", etc. Renamed from JSON `"type"`.
    #[serde(rename = "type")]
    kind: String,
    /// Text content, present only for text blocks.
    text: Option<String>,
}

/// Token usage metadata from Anthropic.
///
/// Private wire-format struct.
#[derive(Deserialize)]
struct AnthropicUsage {
    /// Number of input tokens consumed.
    input_tokens: u32,
    /// Number of output tokens produced.
    output_tokens: u32,
}

/// Anthropic Messages API provider using direct reqwest HTTP calls.
///
/// ## Rust Learning: Reusing `reqwest::Client`
///
/// The `http_client` is a struct field, not built per call, because reqwest
/// docs explicitly advise reuse: "The Client holds a connection pool
/// internally ... it is advised that you create one and reuse it." reqwest's
/// `Client` is already `Arc` internally, so cloning is cheap (reference count
/// increment, no data copy).
pub struct AnthropicProvider {
    /// Anthropic API key, read from environment at app startup.
    api_key: String,
    /// Model identifier (e.g., "claude-sonnet-4-6").
    model: String,
    /// Default max tokens, reserved for future factory use.
    max_tokens_default: u32,
    /// Reusable HTTP client with connection pool, timeout, and keep-alive.
    http_client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    ///
    /// - `api_key`: Anthropic API key, typically from `ANTHROPIC_API_KEY` env var.
    /// - `model`: Model identifier (e.g., `"claude-sonnet-4-6"`).
    /// - `max_tokens_default`: Default max tokens for future factory use; exposed
    ///   via `max_tokens_default()` accessor. `invoke()` uses its own `max_tokens`
    ///   parameter, not this default.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` if the reqwest client fails to build
    /// (e.g., TLS backend unavailable).
    pub fn new(
        api_key: String,
        model: String,
        max_tokens_default: u32,
    ) -> Result<Self, PipelineError> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .tcp_keepalive(Duration::from_secs(TCP_KEEPALIVE_SECS))
            .build()
            .map_err(|e| {
                PipelineError::LlmProvider(format!("Failed to build Anthropic HTTP client: {e}"))
            })?;

        Ok(Self {
            api_key,
            model,
            max_tokens_default,
            http_client,
        })
    }

    /// Configured default max tokens, for factory and diagnostic use.
    ///
    /// `invoke()` uses its own `max_tokens` parameter, not this default.
    pub fn max_tokens_default(&self) -> u32 {
        self.max_tokens_default
    }

    /// POST a prebuilt request body to the Anthropic Messages API and parse
    /// the text response.
    ///
    /// Private transport helper shared by `invoke()` and `invoke_with_system()`.
    /// Centralizes HTTP, rate-limit, overload, and response-parsing logic so a
    /// future change to error handling or token extraction lands in one place
    /// instead of being duplicated across every trait method that sends a prompt.
    ///
    /// The caller builds the request body — the helper is agnostic to whether
    /// the body carries a top-level `system` field, a tool_use block, or just
    /// `messages` — and receives back a parsed `LlmResponse` or a typed error.
    ///
    /// # Errors
    ///
    /// See module-level error taxonomy.
    async fn post_and_parse(&self, body: serde_json::Value) -> Result<LlmResponse, PipelineError> {
        let response = self
            .http_client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    tracing::warn!(
                        model = %self.model,
                        timeout_secs = REQUEST_TIMEOUT_SECS,
                        "Anthropic request timed out"
                    );
                    PipelineError::LlmProvider(format!(
                        "request timeout after {REQUEST_TIMEOUT_SECS}s: {e}"
                    ))
                } else {
                    PipelineError::LlmProvider(format!("network error: {e}"))
                }
            })?;

        let status = response.status();

        // 429 Too Many Requests — extract retry-after header
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(DEFAULT_RETRY_AFTER_SECS);

            tracing::warn!(
                model = %self.model,
                retry_after_secs,
                "Anthropic rate limited (429)"
            );

            return Err(PipelineError::RateLimited { retry_after_secs });
        }

        // 529 Overloaded — Anthropic-specific capacity signal
        if status.as_u16() == 529 {
            let body = response.text().await.unwrap_or_default();
            tracing::warn!(
                model = %self.model,
                "Anthropic overloaded (529)"
            );
            return Err(PipelineError::LlmProvider(format!(
                "overloaded (529): {}",
                &body[..body.len().min(200)]
            )));
        }

        // Other non-success statuses
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let category = if status.is_client_error() {
                "client error"
            } else {
                "server error"
            };
            return Err(PipelineError::LlmProvider(format!(
                "{category} HTTP {status}: {}",
                &body[..body.len().min(300)]
            )));
        }

        // Success — parse response
        let parsed: AnthropicResponse = response.json().await.map_err(|e| {
            PipelineError::LlmProvider(format!("Failed to parse Anthropic response: {e}"))
        })?;

        // Extract text from content blocks (ignores tool_use and other block types)
        let text: String = parsed
            .content
            .iter()
            .filter(|block| block.kind == BLOCK_KIND_TEXT)
            .filter_map(|block| block.text.as_deref())
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return Err(PipelineError::LlmProvider(
                "Anthropic response contained no text content blocks".into(),
            ));
        }

        Ok(LlmResponse {
            text,
            input_tokens: parsed.usage.as_ref().map(|u| u.input_tokens),
            output_tokens: parsed.usage.as_ref().map(|u| u.output_tokens),
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    /// Send a prompt to the Anthropic Messages API and return the raw text response.
    ///
    /// Builds a Messages API request with `temperature: 0.0` (deterministic output
    /// for extraction). No system prompt is injected — prompt engineering is the
    /// caller's concern per the `LlmProvider` trait contract.
    ///
    /// Multi-block responses are concatenated: Anthropic may return multiple text
    /// blocks for long responses. Non-text blocks (tool_use) are ignored because
    /// this provider is for text extraction, not tool-calling.
    ///
    /// # Errors
    ///
    /// See module-level error taxonomy. Returns typed errors that the caller
    /// (LlmExtract step) can match on for retry decisions.
    async fn invoke(&self, prompt: &str, max_tokens: u32) -> Result<LlmResponse, PipelineError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]
        });
        self.post_and_parse(body).await
    }

    /// Send a prompt with a separate system prompt via Anthropic's native
    /// top-level `system` field.
    ///
    /// The Anthropic Messages API treats the `system` field as a distinct
    /// instruction layer — concatenating system + user content into a single
    /// user message loses that distinction and has measurable quality impact
    /// on instruction-following tasks (synthesis, decomposition).
    ///
    /// # Errors
    ///
    /// Same taxonomy as `invoke()` — all HTTP, rate-limit, overload, and
    /// parsing errors are funneled through the shared `post_and_parse` helper.
    async fn invoke_with_system(
        &self,
        system: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<LlmResponse, PipelineError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "system": system,
            "messages": [{"role": "user", "content": prompt}]
        });
        self.post_and_parse(body).await
    }

    /// Returns `"anthropic"` — the canonical lowercase backend identifier.
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    /// Returns the model identifier passed to the constructor.
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Returns `None` — pricing is not exposed by this provider.
    ///
    /// Per-token pricing is deployment configuration (changes with model and
    /// over time) and belongs in application-level config (env vars, database,
    /// or config file), not in a library crate. A future `AppContext`-level
    /// pricing lookup can compute cost from `LlmResponse.input_tokens` /
    /// `output_tokens`.
    fn cost_per_input_token(&self) -> Option<f64> {
        None
    }

    /// Returns `None` — pricing is not exposed by this provider.
    ///
    /// See `cost_per_input_token` for rationale.
    fn cost_per_output_token(&self) -> Option<f64> {
        None
    }

    /// Returns `true` — Anthropic supports tool-use for structured output.
    ///
    /// The `LlmExtract` step in Phase 4 uses this signal to decide between
    /// typed-prompt and raw-completion parsing paths.
    fn supports_structured_output(&self) -> bool {
        true
    }
}
