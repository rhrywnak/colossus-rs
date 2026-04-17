//! RigSynthesizer — implements [`Synthesizer`] by delegating to an [`LlmProvider`].
//!
//! This module sends the assembled context + a user question to an LLM and
//! returns a synthesized answer along with token-usage metrics. The provider
//! abstraction lets us swap between Anthropic, vLLM, or any other
//! `LlmProvider` implementation without touching the RAG pipeline.
//!
//! ## What changed (P3-1 → P3-1-Followup)
//!
//! Previously `RigSynthesizer` held a concrete
//! `rig::providers::anthropic::completion::CompletionModel` and called it
//! directly, which coupled the RAG pipeline to Rig's Anthropic client. The
//! struct now holds `Arc<dyn colossus_extract::LlmProvider>` and passes the
//! system prompt and the user content separately via `invoke_with_system()`.
//! Each provider routes them through its native system-prompt mechanism
//! (Anthropic's top-level `system` field, OpenAI's `role: "system"` message,
//! etc.), preserving the system/user distinction the model was trained on.
//! The Anthropic-specific code lives in
//! `colossus_extract::providers::AnthropicProvider` as of colossus-rs v0.10.0;
//! `invoke_with_system()` and `provider_name()` were added in v0.10.1.
//!
//! The type keeps the name `RigSynthesizer` to avoid churn in re-exports and
//! downstream callsites. A rename is a separate future task.
//!
//! ## Rust Learning: Why `Arc<dyn Trait>`?
//!
//! A trait object like `dyn LlmProvider` is "unsized" — the compiler doesn't
//! know at compile time how large the concrete implementing struct is (one
//! impl might be a fat HTTP client, another a tiny test stub). Rust can't
//! put unsized values on the stack, so we need indirection: either a
//! reference (`&dyn Trait`), a Box (`Box<dyn Trait>`), or an Arc
//! (`Arc<dyn Trait>`).
//!
//! We use `Arc` specifically because the provider lives in `AppState` and
//! gets cloned into every Axum handler. `Arc::clone` is a cheap refcount
//! bump — it does NOT clone the underlying HTTP client, it just shares
//! ownership. A `Box<dyn LlmProvider>` would force each handler to own its
//! own provider and rebuild the HTTP client, which is wasteful.
//!
//! Contrast with the rig `EmbeddingModel` trait used by `QdrantRetriever`:
//! that trait is NOT object-safe (its methods return `impl Future`, which
//! is a compile-time-generic return type), so `dyn EmbeddingModel` is
//! illegal. Our `LlmProvider` is object-safe because `#[async_trait]`
//! rewrites `async fn` to return `Pin<Box<dyn Future>>`, removing the
//! generic.

use std::sync::Arc;

use async_trait::async_trait;

use colossus_extract::LlmProvider;

use crate::error::RagError;
use crate::traits::Synthesizer;
use crate::types::{AssembledContext, SynthesisResult};

// ---------------------------------------------------------------------------
// RigSynthesizer struct
// ---------------------------------------------------------------------------

/// Sends an assembled context to an LLM via the `LlmProvider` abstraction
/// and returns a synthesized answer.
///
/// ## Rust Learning: dyn Trait storage
///
/// `provider` is typed as `Arc<dyn LlmProvider>` — a trait object. We can
/// store any concrete type that implements `LlmProvider` (Anthropic, vLLM,
/// a test stub) without the `RigSynthesizer` struct itself being generic.
/// This is "dynamic dispatch": the actual `invoke()` implementation is
/// looked up at runtime via a vtable pointer carried by the `Arc`.
///
/// The alternative — making `RigSynthesizer` generic over `P: LlmProvider` —
/// would force every handler and test signature to carry that generic
/// parameter, and every callsite would monomorphize a fresh copy. Dynamic
/// dispatch gives us a single compiled `RigSynthesizer` and lets provider
/// choice be a runtime decision (from `LLM_PROVIDER=anthropic|vllm|...`).
pub struct RigSynthesizer {
    provider: Arc<dyn LlmProvider>,
    max_tokens: u32,
}

impl RigSynthesizer {
    /// Create a synthesizer that delegates to the given LLM provider.
    ///
    /// ## Parameters
    /// - `provider`: any `Arc<dyn LlmProvider>` — typically constructed upstream
    ///   in `main.rs` via `colossus_extract::llm_provider_from_env()`.
    /// - `max_tokens`: maximum completion tokens (e.g., 4096 for synthesis).
    ///
    /// ## Note on the `max_tokens` type change
    ///
    /// The previous Rig-backed constructor took `u64` because Rig's Anthropic
    /// builder wanted `u64`. `LlmProvider::invoke()` takes `u32`. `u32::MAX`
    /// is about 4 billion tokens, vastly larger than any real model window,
    /// so this is a safe narrowing. Callers passing small integer literals
    /// (e.g. `4096`) will type-infer to `u32` automatically.
    pub fn new(provider: Arc<dyn LlmProvider>, max_tokens: u32) -> Self {
        Self {
            provider,
            max_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Synthesizer trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Synthesizer for RigSynthesizer {
    /// Send the assembled context and question to the LLM, return the answer.
    ///
    /// ## Flow
    /// 1. Format the user message as `Context:\n{...}\n\nQuestion: {...}`.
    /// 2. Call `provider.invoke_with_system(system_prompt, user_message,
    ///    max_tokens)` so the provider routes the system content through
    ///    its native system-prompt channel.
    /// 3. Guard against an empty response.
    /// 4. Build a `SynthesisResult`, defaulting missing token counts to 0.
    async fn synthesize(
        &self,
        context: &AssembledContext,
        question: &str,
    ) -> Result<SynthesisResult, RagError> {
        // --- Step 1: Format the user message ---
        //
        // System and user content are passed separately to the provider via
        // `invoke_with_system()`. The provider routes the system content
        // through its native channel (Anthropic's top-level `system` field,
        // OpenAI's `role: "system"` message, etc.). We still wrap the
        // retrieved context and the user's question into a single user-side
        // message labelled "Context:" / "Question:" so the model sees them
        // as one structured user turn.
        let user_message = format!(
            "Context:\n{}\n\nQuestion: {}",
            context.formatted_context, question
        );

        // --- Step 2: Invoke the provider ---
        let response = self
            .provider
            .invoke_with_system(&context.system_prompt, &user_message, self.max_tokens)
            .await
            .map_err(|e| RagError::SynthesisError(e.to_string()))?;

        if response.text.is_empty() {
            return Err(RagError::SynthesisError(
                "Empty response from LLM provider".to_string(),
            ));
        }

        // --- Step 3: Build SynthesisResult ---
        //
        // ## Compatibility note
        //
        // `LlmResponse` has `Option<u32>` token counts because some providers
        // don't report usage (self-hosted vLLM with streaming disabled, for
        // instance). `SynthesisResult` requires `u32`, so we unwrap to 0 —
        // the same convention used by the v0.10.0 providers when a field is
        // genuinely unknown.
        Ok(SynthesisResult {
            answer: response.text,
            citations: Vec::new(), // Deferred to a future task (citation parsing)
            input_tokens: response.input_tokens.unwrap_or(0),
            output_tokens: response.output_tokens.unwrap_or(0),
            provider: self.provider.provider_name().to_string(),
            model: self.provider.model_name().to_string(),
        })
    }
}
