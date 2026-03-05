//! RigSynthesizer — implements [`Synthesizer`] using Rig's Anthropic provider.
//!
//! This module provides the synthesis stage of the RAG pipeline: sending
//! assembled context + a user question to Claude and getting back a
//! natural-language answer with token usage metrics.
//!
//! It replaces: `colossus-legal/backend/src/services/claude_client.rs` (~120 lines)
//! which used raw reqwest HTTP calls to the Anthropic Messages API.
//!
//! ## Architecture: Low-level `completion_request()` API
//!
//! Rig offers two levels of API for calling Claude:
//!
//! 1. **High-level (Agent)**: `agent.prompt("text").await` → `String`
//!    - Simple but returns ONLY the answer text — no token counts.
//!    - Preamble (system prompt) is set at Agent BUILD time, not per-request.
//!    - `agent.prompt("text").extended_details().await` → `PromptResponse`
//!      which includes token usage, but still requires rebuilding the Agent
//!      per request (because our system prompt changes per question).
//!
//! 2. **Low-level (CompletionModel)**: `model.completion_request("text").send().await`
//!    → `CompletionResponse` with `.usage.input_tokens`, `.usage.output_tokens`.
//!    - Preamble can be set per-request via `.preamble(system_prompt)`.
//!    - Direct access to the full `CompletionResponse` including token usage.
//!
//! We use the **low-level API** because:
//! - Our `system_prompt` changes per request (it includes the assembled context)
//! - We need token counts for `PipelineStats`
//! - It maps directly to how the Anthropic API works, making it easier to understand
//!
//! ## Rig Concept: CompletionModel vs Agent
//!
//! Think of `CompletionModel` as a thin wrapper around the HTTP client.
//! It knows the API URL, auth headers, and model ID, but nothing about
//! system prompts or tools. Each call is stateless.
//!
//! An `Agent` wraps a `CompletionModel` with persistent configuration:
//! preamble, tools, static context. It's great for chatbots, but overkill
//! when your "system prompt" changes on every request (like in RAG).

use async_trait::async_trait;

/// ## Rust Learning: Importing traits for method resolution
///
/// `CompletionModel` trait must be in scope to call `.completion_request()`.
/// `CompletionClient` trait must be in scope to call `.completion_model()`.
/// Even though the Anthropic Client has these methods, Rust won't find them
/// unless the defining trait is imported.
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Text};
use rig::client::CompletionClient;

use crate::error::RagError;
use crate::traits::Synthesizer;
use crate::types::{AssembledContext, SynthesisResult};

// ---------------------------------------------------------------------------
// RigSynthesizer struct
// ---------------------------------------------------------------------------

/// Sends assembled context to Claude and returns a synthesized answer.
///
/// Uses Rig's Anthropic provider with the low-level `completion_request()` API
/// for per-request system prompts and token usage access.
///
/// ## Rust Learning: No `Arc` needed
///
/// Unlike `QdrantRetriever` (which wraps heavy clients in `Arc`), the Anthropic
/// `CompletionModel` is already cheap to clone — it's a small struct wrapping
/// an `Arc<reqwest::Client>` internally. So we store it directly.
///
/// If you needed to share a `RigSynthesizer` across Axum handlers, you'd wrap
/// the entire `RigSynthesizer` in `Arc<RigSynthesizer>` at the handler level.
pub struct RigSynthesizer {
    /// The Rig Anthropic completion model (e.g., claude-sonnet-4-5-20250929).
    ///
    /// ## Rig Concept: `rig::providers::anthropic::completion::CompletionModel`
    ///
    /// This is the provider-specific type that implements Rig's `CompletionModel`
    /// trait. It wraps the Anthropic HTTP client, model ID, and API configuration.
    /// Created via `client.completion_model("model-id")`.
    ///
    /// The generic parameter defaults to `reqwest::Client` for the HTTP transport.
    model: rig::providers::anthropic::completion::CompletionModel,

    /// Maximum tokens for the completion response.
    ///
    /// ## Rig Concept: `max_tokens` is REQUIRED for Anthropic
    ///
    /// Unlike OpenAI, the Anthropic API REQUIRES `max_tokens` on every request.
    /// Rig enforces this — omitting `.max_tokens()` causes a runtime error.
    /// A good default for RAG synthesis is 4096 (enough for a detailed answer
    /// with citations, but not excessively large).
    max_tokens: u64,

    /// Provider name for `SynthesisResult` metadata (always "anthropic").
    provider_name: String,

    /// Model ID for `SynthesisResult` metadata (e.g., "claude-sonnet-4-5-20250929").
    model_name: String,
}

impl RigSynthesizer {
    /// Create a synthesizer configured for Claude via Anthropic's API.
    ///
    /// ## Parameters
    /// - `api_key`: Your ANTHROPIC_API_KEY
    /// - `model_id`: The model to use (e.g., "claude-sonnet-4-5-20250929").
    ///   Use a specific model ID string, NOT Rig's built-in constants — they
    ///   can be stale and return 404 errors.
    /// - `max_tokens`: Maximum completion tokens (e.g., 4096)
    ///
    /// ## Rig Concept: Client creation
    ///
    /// `rig::providers::anthropic::Client::new(api_key)` creates an HTTP client
    /// configured for the Anthropic API. It returns `Result` because it builds
    /// the reqwest HTTP client (which can fail if TLS initialization fails —
    /// extremely rare but possible on exotic platforms).
    ///
    /// It sets the `x-api-key` and `anthropic-version` headers automatically.
    /// Then `.completion_model(model_id)` wraps that client with a specific model ID.
    ///
    /// Alternatively, `Client::from_env()` reads ANTHROPIC_API_KEY from
    /// the environment, but we take the key explicitly for clarity and
    /// testability — the caller decides where the key comes from.
    pub fn claude(api_key: &str, model_id: &str, max_tokens: u64) -> Result<Self, RagError> {
        let client = rig::providers::anthropic::Client::new(api_key)
            .map_err(|e| RagError::ConfigError(format!("Failed to create Anthropic client: {e}")))?;
        let model = client.completion_model(model_id);

        Ok(Self {
            model,
            max_tokens,
            provider_name: "anthropic".to_string(),
            model_name: model_id.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Synthesizer trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Synthesizer for RigSynthesizer {
    /// Send the assembled context and question to Claude, return the answer.
    ///
    /// ## Flow
    /// 1. Combine `formatted_context` and `question` into a user message
    /// 2. Set `system_prompt` as the preamble (Rig maps this to Anthropic's `system` field)
    /// 3. Call `completion_request().preamble().max_tokens().send()`
    /// 4. Extract answer text from `CompletionResponse.choice`
    /// 5. Extract token usage from `CompletionResponse.usage`
    /// 6. Return `SynthesisResult`
    async fn synthesize(
        &self,
        context: &AssembledContext,
        question: &str,
    ) -> Result<SynthesisResult, RagError> {
        // --- Step 1: Build the user message ---
        //
        // We combine the formatted evidence context with the user's question
        // into a single user message. The system prompt (preamble) is separate.
        //
        // This mirrors the Anthropic API structure:
        //   system: "You are a legal analyst..." (preamble)
        //   messages: [{ role: "user", content: "Context: ...\n\nQuestion: ..." }]
        let user_message = format!(
            "{}\n\nQuestion: {}",
            context.formatted_context, question
        );

        // --- Step 2: Build and send the completion request ---
        //
        // ## Rig Concept: CompletionRequestBuilder chain
        //
        // `.completion_request(prompt)` creates a builder with the user message.
        // `.preamble(text)` sets the system prompt (Anthropic's `system` field).
        // `.max_tokens(n)` sets the maximum completion length (REQUIRED for Anthropic).
        // `.send()` sends the HTTP request and returns `CompletionResponse`.
        //
        // The builder consumes `self` at each step (ownership transfer), which
        // is why you chain the calls in a single expression. This is Rust's
        // builder pattern — each method returns a new builder.
        let response = self
            .model
            .completion_request(&user_message)
            .preamble(context.system_prompt.clone())
            .max_tokens(self.max_tokens)
            .send()
            .await
            .map_err(|e| RagError::SynthesisError(e.to_string()))?;

        // --- Step 3: Extract the answer text ---
        let answer = extract_text_from_response(&response.choice);

        if answer.is_empty() {
            return Err(RagError::SynthesisError(
                "Empty response from Claude".to_string(),
            ));
        }

        // --- Step 4: Build SynthesisResult with token usage ---
        //
        // ## Rig Concept: `Usage` struct
        //
        // `response.usage` contains:
        // - `input_tokens: u64` — tokens consumed by the prompt (system + user)
        // - `output_tokens: u64` — tokens generated in the response
        // - `total_tokens: u64` — sum (some providers report this separately)
        // - `cached_input_tokens: u64` — from prompt caching (if applicable)
        //
        // We convert u64 → u32 for our `SynthesisResult` type. This is safe
        // because token counts never approach u32::MAX (~4 billion).
        Ok(SynthesisResult {
            answer,
            citations: Vec::new(), // Deferred to a future task (citation parsing)
            input_tokens: response.usage.input_tokens as u32,
            output_tokens: response.usage.output_tokens as u32,
            provider: self.provider_name.clone(),
            model: self.model_name.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: extract text from CompletionResponse
// ---------------------------------------------------------------------------

/// Extract text content from a Rig `CompletionResponse.choice`.
///
/// ## Rig Concept: `OneOrMany<AssistantContent>`
///
/// Claude can return multiple content blocks (e.g., text + tool calls).
/// Rig wraps these in `OneOrMany<AssistantContent>`, which is an iterable
/// container that always has at least one element.
///
/// `AssistantContent` is an enum with variants:
/// - `Text(Text { text })` — the text content we want
/// - `ToolCall(...)` — tool use requests (not applicable for our RAG use case)
/// - `Reasoning(...)` — chain-of-thought reasoning blocks
/// - `Image(...)` — image content
///
/// We iterate over all blocks, extract text from `Text` variants, and join
/// them. In practice, a simple RAG synthesis response will have exactly one
/// `Text` block.
///
/// ## Rust Learning: Pattern matching on enum variants
///
/// `if let AssistantContent::Text(Text { text }) = item` is a "pattern match"
/// that both checks the variant AND destructures the inner `Text` struct
/// to extract the `text` field. If the variant doesn't match, the block is
/// skipped. This replaces Java's `instanceof` + cast pattern.
fn extract_text_from_response(
    choice: &rig::one_or_many::OneOrMany<AssistantContent>,
) -> String {
    let mut parts: Vec<&str> = Vec::new();

    for item in choice.iter() {
        if let AssistantContent::Text(Text { text }) = item {
            parts.push(text);
        }
    }

    parts.join("\n")
}
