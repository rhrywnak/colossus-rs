//! LlmDecomposer — uses a fast LLM call to decompose complex questions.
//!
//! When a question references a specific document ("What did Phillips state
//! in his CoA response?"), the decomposer identifies the document and creates
//! a graph sub-query to fetch its content directly, rather than relying
//! solely on vector similarity.
//!
//! ## Architecture
//!
//! The decomposer is LLM-only — it produces a `DecompositionResult` with
//! structured sub-queries but does NOT execute them. The pipeline handles
//! execution, dispatching vector sub-queries to the retriever and graph
//! sub-queries to the `GraphDirectRetriever`.
//!
//! ## System / user prompt split
//!
//! Persona, rules, and known-entity lists go in the system prompt (stable
//! across calls). Question and strategy go in the user prompt (vary per
//! call). The split reduces prompt token duplication on repeat calls and
//! lets Anthropic-style APIs weight instructional content in the native
//! `system` field. The provider's `invoke_with_system()` method owns that
//! split: `AnthropicProvider` maps `system` to the Messages-API `system`
//! field; `VllmProvider` emits an OpenAI `role: "system"` message; the
//! trait default concatenates and falls back to `invoke`.
//!
//! ## When decomposition helps
//!
//! - "What did Phillips state in his CoA response?" — identifies document,
//!   creates GraphDocumentContent sub-query
//! - "What statements did Marie make?" — identifies person, creates
//!   GraphPersonStatements sub-query
//! - "Where did Phillips contradict himself?" — creates GraphContradictions
//!   sub-query
//!
//! ## When decomposition doesn't help (and correctly passes through)
//!
//! - "What evidence supports breach of fiduciary duty?" — broad question,
//!   no specific document or person target, returns single VectorSearch

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;

use colossus_extract::LlmProvider;

use crate::error::RagError;
use crate::traits::QueryDecomposer;
use crate::types::{DecompositionResult, RetrievalStrategy, SubQuery};

// ---------------------------------------------------------------------------
// Default prompt templates
// ---------------------------------------------------------------------------

/// Default system prompt — persona, rules, and known-entity lists.
/// Used when no external `system_template` is supplied.
///
/// Placeholders resolved via `.replace()` at runtime:
/// - `{docs}` — formatted document alias list
/// - `{persons}` — formatted person name list
///
/// The JSON example below uses single braces because runtime `.replace()`
/// does not interpret `{{`/`}}` escapes (unlike the compile-time `format!`
/// macro). The wire content the LLM receives is identical to the old
/// `format!`-based prompt.
const DEFAULT_SYSTEM_TEMPLATE: &str = r#"You are a legal research query planner for the Awad v. CFS/Phillips case.

Given a question, decide whether it needs to be decomposed into sub-queries for better document retrieval.

AVAILABLE DOCUMENTS (alias -> id):
{docs}

KNOWN PERSONS:
{persons}

RULES:
1. If the question asks about content FROM a specific document, add a "graph_document_content" sub-query with the document_id.
2. If the question asks what a specific person STATED or SAID, add a "graph_person_statements" sub-query. Use the person's graph ID format (lowercase, hyphenated, e.g. "george-phillips").
3. If the question asks about CONTRADICTIONS by a person, add a "graph_contradictions" sub-query with the person's name.
4. ALWAYS also include a "vector_search" sub-query with a refined version of the question.
5. For broad questions with no specific document or person target, set needs_decomposition to false and include only a single vector_search.
6. Use EXACT document_id values from the AVAILABLE DOCUMENTS list.

Respond with ONLY valid JSON matching this structure:
{
  "needs_decomposition": true/false,
  "sub_queries": [
    {"type": "vector_search", "query": "..."},
    {"type": "graph_document_content", "document_id": "...", "description": "..."},
    {"type": "graph_person_statements", "person_id": "...", "description": "..."},
    {"type": "graph_contradictions", "person_name": "...", "description": "..."}
  ]
}"#;

/// Default user prompt — question and strategy for this specific call.
/// Used when no external `user_template` is supplied.
///
/// Placeholders resolved via `.replace()` at runtime:
/// - `{question}` — the user's question verbatim
/// - `{strategy}` — the `RetrievalStrategy` as Debug output
const DEFAULT_USER_TEMPLATE: &str = r#"QUESTION: {question}
STRATEGY: {strategy}"#;

// ---------------------------------------------------------------------------
// LlmDecomposer struct
// ---------------------------------------------------------------------------

/// Decomposes questions using a fast LLM call (e.g., Claude Sonnet).
///
/// ## Rust Learning: `Arc<dyn LlmProvider>` instead of a concrete model
///
/// The previous version held a `rig::providers::anthropic::completion::CompletionModel`
/// directly. That tied the decomposer to one provider (Anthropic) and made
/// it untestable without an HTTP client. The new design holds an
/// `Arc<dyn LlmProvider>` — a reference-counted pointer to any type that
/// implements the provider trait. Tests inject a stub; production wires up
/// `AnthropicProvider` or `VllmProvider`. Neither the struct nor the
/// `decompose()` method needs to change when a new provider is added.
///
/// Same trait-object pattern as `RigSynthesizer` (see P3-1).
pub struct LlmDecomposer {
    provider: Arc<dyn LlmProvider>,

    /// Formatted document alias list: `  - "alias" -> id` lines. Built once
    /// at construction from the input HashMap.
    document_list: String,

    /// Formatted person name list: `  - Name` lines. Built once at construction.
    person_list: String,

    /// Optional externalized SYSTEM prompt template. When `None`, uses the
    /// hardcoded default.
    ///
    /// Must contain `{docs}` and `{persons}` placeholders (literal text —
    /// replaced at runtime via `.replace()`).
    system_template: Option<String>,

    /// Optional externalized USER prompt template. When `None`, uses the
    /// hardcoded default.
    ///
    /// Must contain `{question}` and `{strategy}` placeholders (literal
    /// text — replaced at runtime via `.replace()`).
    user_template: Option<String>,
}

impl LlmDecomposer {
    /// Create a decomposer with an LLM provider and knowledge graph metadata.
    ///
    /// ## Rust Learning: infallible construction
    ///
    /// This constructor returns `Self`, not `Result<Self, RagError>`. The
    /// previous version built an Anthropic HTTP client internally (a fallible
    /// operation). Now the caller supplies an already-constructed
    /// `Arc<dyn LlmProvider>` — a pointer to something that can never fail
    /// to exist. Removing the `Result` wrapper when there's nothing to fail
    /// is idiomatic Rust and reduces noise at call sites (no `.unwrap()` /
    /// `?` needed).
    ///
    /// ## Parameters
    ///
    /// - `provider`: any `LlmProvider` implementation (production: Anthropic
    ///   or vLLM; tests: an in-memory stub)
    /// - `document_aliases`: Map of alias -> document_id (from router config)
    /// - `person_names`: List of known person names (from router config)
    /// - `system_template`: optional external SYSTEM template. If `Some`,
    ///   must contain `{docs}` and `{persons}` placeholders.
    /// - `user_template`: optional external USER template. If `Some`, must
    ///   contain `{question}` and `{strategy}` placeholders.
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        document_aliases: &std::collections::HashMap<String, String>,
        person_names: &[String],
        system_template: Option<String>,
        user_template: Option<String>,
    ) -> Self {
        let document_list = document_aliases
            .iter()
            .map(|(alias, id)| format!("  - \"{alias}\" -> {id}"))
            .collect::<Vec<_>>()
            .join("\n");

        let person_list = person_names
            .iter()
            .map(|name| format!("  - {name}"))
            .collect::<Vec<_>>()
            .join("\n");

        Self {
            provider,
            document_list,
            person_list,
            system_template,
            user_template,
        }
    }

    /// Build the system prompt (persona + rules + known-entity lists).
    ///
    /// Uses `{docs}` and `{persons}` placeholders. Does NOT include
    /// `{question}` or `{strategy}` — those go in the user prompt.
    fn build_system_prompt(&self) -> String {
        let template = self
            .system_template
            .as_deref()
            .unwrap_or(DEFAULT_SYSTEM_TEMPLATE);
        template
            .replace("{docs}", &self.document_list)
            .replace("{persons}", &self.person_list)
    }

    /// Build the user prompt (question + strategy for this specific call).
    ///
    /// Uses `{question}` and `{strategy}` placeholders. Does NOT include
    /// `{docs}` or `{persons}` — those are in the system prompt.
    fn build_user_prompt(&self, question: &str, strategy: &RetrievalStrategy) -> String {
        let strategy_str = format!("{strategy:?}");
        let template = self
            .user_template
            .as_deref()
            .unwrap_or(DEFAULT_USER_TEMPLATE);
        template
            .replace("{question}", question)
            .replace("{strategy}", &strategy_str)
    }
}

// ---------------------------------------------------------------------------
// QueryDecomposer trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl QueryDecomposer for LlmDecomposer {
    async fn decompose(
        &self,
        question: &str,
        strategy: &RetrievalStrategy,
    ) -> Result<DecompositionResult, RagError> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(question, strategy);

        tracing::info!(
            provider = %self.provider.provider_name(),
            model = %self.provider.model_name(),
            question_len = question.len(),
            "Decomposer: analyzing question"
        );

        // Call the provider with separate system + user content so Anthropic's
        // native `system` field is used, weighting the instructional content.
        let response = self
            .provider
            .invoke_with_system(&system_prompt, &user_prompt, 500)
            .await;

        // On LLM error, fall back to no decomposition (preserve existing behavior).
        let response = match response {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "Decomposer LLM call failed, using fallback");
                return Ok(fallback_result(question));
            }
        };

        // Parse JSON — strip markdown fences if present.
        let clean_json = response
            .text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        match serde_json::from_str::<DecompositionParsed>(clean_json) {
            Ok(parsed) => {
                tracing::info!(
                    needs_decomposition = parsed.needs_decomposition,
                    sub_queries = parsed.sub_queries.len(),
                    "Decomposer: analysis complete"
                );

                // Ensure at least one sub-query exists.
                let sub_queries = if parsed.sub_queries.is_empty() {
                    vec![SubQuery::VectorSearch {
                        query: question.to_string(),
                    }]
                } else {
                    parsed.sub_queries
                };

                Ok(DecompositionResult {
                    needs_decomposition: parsed.needs_decomposition,
                    sub_queries,
                    original_question: question.to_string(),
                })
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    raw_response = %clean_json,
                    "Decomposer: failed to parse LLM response, falling back"
                );
                Ok(fallback_result(question))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a fallback DecompositionResult (no decomposition, original question).
fn fallback_result(question: &str) -> DecompositionResult {
    DecompositionResult {
        needs_decomposition: false,
        sub_queries: vec![SubQuery::VectorSearch {
            query: question.to_string(),
        }],
        original_question: question.to_string(),
    }
}

/// Internal struct for parsing the LLM's JSON response.
///
/// Separate from `DecompositionResult` because the LLM doesn't produce
/// `original_question` — we add that ourselves.
#[derive(Debug, Deserialize)]
struct DecompositionParsed {
    needs_decomposition: bool,
    sub_queries: Vec<SubQuery>,
}
