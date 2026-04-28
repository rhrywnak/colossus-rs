//! vLLM OpenAI-compatible Chat Completions provider — direct reqwest implementation.
//!
//! ## Why direct reqwest, not an OpenAI SDK
//!
//! There is no official Rust OpenAI SDK from OpenAI. The community crate
//! `async-openai` exists but adds dependency weight for a feature set we do not
//! need. Using reqwest directly keeps consistency with `AnthropicProvider` and
//! avoids an extra dependency.
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
//! ## Why `supports_structured_output` returns false
//!
//! vLLM supports structured output via `guided_choice` / `structured_outputs`
//! in `extra_body`, which is a different mechanism from OpenAI's `response_format`
//! JSON mode. This provider returns `false` as a conservative signal; a future
//! provider method can expose vLLM's guided-choice capability.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::PipelineError;
use crate::traits::{LlmProvider, LlmResponse};

/// OpenAI-compatible Chat Completions endpoint path, confirmed by docs.vllm.ai.
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// Default request timeout when the caller does not specify one — matches
/// `AnthropicProvider`.
///
/// 10 minutes accommodates long-context extraction. vLLM has no different
/// recommendation. Callers that need a different value pass it via the
/// `request_timeout_secs` constructor parameter.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 600;

/// TCP keep-alive interval in seconds — matches AnthropicProvider.
///
/// Corporate firewalls/NAT may drop idle connections even on LAN.
const TCP_KEEPALIVE_SECS: u64 = 60;

/// Fallback when 429 comes via proxy without retry-after header.
///
/// vLLM itself does not emit 429; this handles the case where vLLM sits
/// behind a rate-limiting proxy (e.g., LiteLLM).
const DEFAULT_RETRY_AFTER_SECS: u64 = 60;

/// OpenAI Chat Completions `finish_reason` value indicating `max_tokens` was hit.
///
/// Surfaced in the error message when content is empty, to help the caller
/// diagnose truncation vs. model failure.
const FINISH_REASON_LENGTH: &str = "length";

/// OpenAI Chat Completions response envelope.
///
/// Private wire-format struct — not part of the public API.
#[derive(Deserialize)]
struct ChatCompletionResponse {
    /// Completion choices returned by the model.
    choices: Vec<ChatChoice>,
    /// Token usage metadata, if present.
    usage: Option<ChatUsage>,
}

/// A single completion choice.
///
/// Private wire-format struct. `finish_reason` is `Option` because some proxies
/// omit it.
#[derive(Deserialize)]
struct ChatChoice {
    /// The assistant's response message.
    message: ChatMessage,
    /// Why generation stopped: "stop", "length", etc.
    finish_reason: Option<String>,
}

/// Assistant message from a Chat Completions response.
///
/// Private wire-format struct.
///
/// ## Rust Learning: Why `Option<String>` and not `String`
///
/// The OpenAI spec documents `content` as `Optional[Union[str, List[...], null]]`.
/// In practice: (a) vLLM returns `content: ""` when `finish_reason` is `"length"`,
/// (b) some OpenAI models return `content: null` for tool-call-only responses.
/// Using `Option<String>` lets serde deserialize both `null` (→ `None`) and
/// `""` (→ `Some("")`) without failure. We treat both as "no useful content."
#[derive(Deserialize)]
struct ChatMessage {
    /// Text content, if present. `None` when the API returns `null`.
    content: Option<String>,
}

/// Token usage metadata from an OpenAI-compatible endpoint.
///
/// Private wire-format struct.
///
/// ## Rust Learning: Field name differences across providers
///
/// OpenAI uses `prompt_tokens` / `completion_tokens`; Anthropic uses
/// `input_tokens` / `output_tokens`. The `LlmProvider` trait abstracts this
/// via `LlmResponse.input_tokens` / `output_tokens`. Each provider maps its
/// wire-format fields to the trait's abstract names.
#[derive(Deserialize)]
struct ChatUsage {
    /// Input tokens consumed (OpenAI calls this `prompt_tokens`).
    prompt_tokens: u32,
    /// Output tokens produced (OpenAI calls this `completion_tokens`).
    completion_tokens: u32,
}

/// vLLM OpenAI-compatible Chat Completions provider using direct reqwest HTTP calls.
///
/// ## Rust Learning: `Option<String>` for optional authentication
///
/// `api_key: Option<String>` uses `None` to mean "no authentication" — matching
/// vLLM's optional `--api-key` flag. This is contrasted with the anti-pattern of
/// using an empty string to mean "no auth," which would produce a malformed
/// `Authorization: Bearer ` header with an empty token.
pub struct VllmProvider {
    /// vLLM server base URL without `/v1` suffix (e.g., `"http://host:8000"`).
    base_url: String,
    /// Model identifier served by this vLLM instance.
    model: String,
    /// API key for authenticated deployments, or `None` for local/unauth servers.
    api_key: Option<String>,
    /// Default max tokens, reserved for future factory use.
    max_tokens_default: u32,
    /// Reusable HTTP client with connection pool, timeout, and keep-alive.
    http_client: reqwest::Client,
    /// Request timeout in seconds, retained for diagnostic logging and error
    /// messages so operators can see the configured value at the call site.
    request_timeout_secs: u64,
}

impl VllmProvider {
    /// Create a new vLLM provider.
    ///
    /// - `base_url`: vLLM server root, e.g., `"http://colossus-dev-app1:8000"`.
    ///   Must NOT include `/v1` suffix or trailing slash — the provider appends
    ///   `/v1/chat/completions` automatically.
    /// - `model`: Model identifier served by vLLM (e.g., `"llama-3-8b"`).
    /// - `api_key`: `Some(key)` for authenticated deployments (vLLM `--api-key`),
    ///   `None` for local servers without authentication.
    /// - `max_tokens_default`: Default max tokens for future factory use; exposed
    ///   via `max_tokens_default()` accessor.
    /// - `request_timeout_secs`: Optional reqwest client timeout in seconds.
    ///   `None` uses [`DEFAULT_REQUEST_TIMEOUT_SECS`] (600). Callers extracting
    ///   large documents should pass a longer timeout.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` if `base_url` ends with `/v1` or a
    /// trailing slash (common footgun), or if the reqwest client fails to build.
    pub fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        max_tokens_default: u32,
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
                PipelineError::LlmProvider(format!("Failed to build vLLM HTTP client: {e}"))
            })?;

        Ok(Self {
            base_url,
            model,
            api_key,
            max_tokens_default,
            http_client,
            request_timeout_secs,
        })
    }

    /// Configured default max tokens, for factory and diagnostic use.
    ///
    /// `invoke()` uses its own `max_tokens` parameter, not this default.
    pub fn max_tokens_default(&self) -> u32 {
        self.max_tokens_default
    }

    /// POST a prebuilt request body to the vLLM Chat Completions endpoint and
    /// parse the text response.
    ///
    /// Private transport helper shared by `invoke()` and `invoke_with_system()`.
    /// Centralizes URL construction, conditional bearer-auth, rate-limit,
    /// validation-error handling, and response parsing so a future change to
    /// (for example) `finish_reason` handling lands in one place instead of
    /// being duplicated across every trait method that sends a prompt.
    ///
    /// The caller builds the request body — the helper does not care whether
    /// `messages` contains a single user entry or a system + user pair.
    ///
    /// # Errors
    ///
    /// See module-level error taxonomy.
    async fn post_and_parse(&self, body: serde_json::Value) -> Result<LlmResponse, PipelineError> {
        let url = format!("{}{CHAT_COMPLETIONS_PATH}", self.base_url);

        // Build request, conditionally adding auth header
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
                    "vLLM request timed out"
                );
                PipelineError::LlmProvider(format!(
                    "request timeout after {timeout_secs}s: {e}"
                ))
            } else {
                PipelineError::LlmProvider(format!("network error: {e}"))
            }
        })?;

        let status = response.status();

        // 429 Too Many Requests — from proxy (vLLM itself does not rate-limit)
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(DEFAULT_RETRY_AFTER_SECS);

            tracing::warn!(
                model = %self.model,
                provider = "vllm",
                retry_after_secs,
                "vLLM rate limited (429)"
            );

            return Err(PipelineError::RateLimited { retry_after_secs });
        }

        // Non-success statuses
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

        // Success — parse response
        let parsed: ChatCompletionResponse = response.json().await.map_err(|e| {
            PipelineError::LlmProvider(format!("Failed to parse vLLM response: {e}"))
        })?;

        // Defensive: choices array can be empty
        if parsed.choices.is_empty() {
            return Err(PipelineError::LlmProvider(
                "vLLM response contained no choices".into(),
            ));
        }

        let first_choice = &parsed.choices[0];
        let finish_reason_str = first_choice.finish_reason.as_deref().unwrap_or("unknown");

        // Extract content — None or empty string are both "no useful content"
        let content = first_choice.message.content.as_deref().unwrap_or("");
        if content.is_empty() {
            let context = if finish_reason_str == FINISH_REASON_LENGTH {
                " — max_tokens may be too low"
            } else {
                ""
            };
            return Err(PipelineError::LlmProvider(format!(
                "vLLM returned empty content (finish_reason: {finish_reason_str}){context}"
            )));
        }

        Ok(LlmResponse {
            text: content.to_string(),
            input_tokens: parsed.usage.as_ref().map(|u| u.prompt_tokens),
            output_tokens: parsed.usage.as_ref().map(|u| u.completion_tokens),
        })
    }
}

#[async_trait]
impl LlmProvider for VllmProvider {
    /// Send a prompt to the vLLM Chat Completions endpoint and return the text response.
    ///
    /// Builds a request matching the OpenAI Chat Completions API with
    /// `temperature: 0.0` (deterministic output for extraction). No system prompt
    /// is injected — prompt engineering is the caller's concern per the
    /// `LlmProvider` trait contract.
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

    /// Send a prompt with a separate system prompt using the OpenAI-compatible
    /// `role: "system"` first-message convention.
    ///
    /// OpenAI-compatible Chat Completions endpoints (including vLLM) treat the
    /// leading `"system"` message as a distinct instruction layer.
    /// Concatenating system + user content into a single user message loses
    /// that distinction.
    ///
    /// # Errors
    ///
    /// Same taxonomy as `invoke()` — all transport and parsing errors are
    /// funneled through the shared `post_and_parse` helper.
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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        });
        self.post_and_parse(body).await
    }

    /// Returns `"vllm"` — the canonical lowercase backend identifier.
    fn provider_name(&self) -> &str {
        "vllm"
    }

    /// Returns the model identifier passed to the constructor.
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Returns `None` — local vLLM inference has no per-token dollar cost.
    ///
    /// Hardware amortization is tracked outside this crate. A future
    /// `AppContext`-level pricing lookup can compute cost from
    /// `LlmResponse.input_tokens` / `output_tokens` if needed.
    fn cost_per_input_token(&self) -> Option<f64> {
        None
    }

    /// Returns `None` — see `cost_per_input_token` for rationale.
    fn cost_per_output_token(&self) -> Option<f64> {
        None
    }

    /// Returns `false` — vLLM structured output uses a different mechanism.
    ///
    /// vLLM supports structured output via `guided_choice` / `structured_outputs`
    /// in `extra_body`, which is not the same as OpenAI's `response_format` JSON
    /// mode. Returning `false` signals the `LlmExtract` step to use
    /// raw-completion + JSON-repair parsing rather than typed-prompt extraction.
    fn supports_structured_output(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-default `request_timeout_secs` must be accepted without error
    /// — covers the new constructor path added by Phase 1c.
    #[test]
    fn provider_new_accepts_custom_timeout() {
        let provider = VllmProvider::new(
            "http://localhost:8000".to_string(),
            "test-model".to_string(),
            None,
            32_000,
            Some(1800),
        );
        assert!(provider.is_ok(), "Custom timeout must be accepted");
    }
}
