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

use async_trait::async_trait;
use serde::Deserialize;

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Text};

use crate::error::RagError;
use crate::traits::QueryDecomposer;
use crate::types::{DecompositionResult, RetrievalStrategy, SubQuery};

// ---------------------------------------------------------------------------
// LlmDecomposer struct
// ---------------------------------------------------------------------------

/// Decomposes questions using a fast LLM call (e.g., Claude Sonnet).
///
/// ## Rust Learning: Same Rig pattern as RigSynthesizer
///
/// Both `LlmDecomposer` and `RigSynthesizer` use Rig's Anthropic
/// `CompletionModel`. The difference is the prompt and token budget:
/// - Synthesizer: rich system prompt, 4096 max tokens, uses the "main" model
/// - Decomposer: structured decomposition prompt, 500 max tokens, uses a fast model
pub struct LlmDecomposer {
    model: rig::providers::anthropic::completion::CompletionModel,
    model_name: String,

    /// Known document aliases for the prompt context.
    /// Format: "phillips coa response -> doc-phillips-coa-response-300891"
    document_list: String,

    /// Known person names for the prompt context.
    person_list: String,

    /// Optional externalized prompt template loaded from disk.
    ///
    /// If `Some`, this template is used instead of the hardcoded default.
    /// The template must contain these exact placeholders (literal text,
    /// NOT Rust format syntax — we use `.replace()` at runtime):
    /// - `{docs}` — replaced with formatted document alias list
    /// - `{persons}` — replaced with formatted person name list
    /// - `{question}` — replaced with the user's question
    /// - `{strategy}` — replaced with the retrieval strategy as a string
    prompt_template: Option<String>,
}

impl LlmDecomposer {
    /// Create a decomposer with an Anthropic model and knowledge graph metadata.
    ///
    /// ## Parameters
    ///
    /// - `api_key`: Anthropic API key (same as synthesizer)
    /// - `model_id`: Model to use (e.g., "claude-sonnet-4-6" — fast and cheap)
    /// - `document_aliases`: Map of alias -> document_id (from router config)
    /// - `person_names`: List of known person names (from router config)
    /// - `prompt_template`: Optional externalized prompt template loaded from disk.
    ///   If `None`, the hardcoded default is used. If `Some`, must contain
    ///   `{docs}`, `{persons}`, `{question}`, and `{strategy}` placeholders.
    pub fn new(
        api_key: &str,
        model_id: &str,
        document_aliases: &std::collections::HashMap<String, String>,
        person_names: &[String],
        prompt_template: Option<String>,
    ) -> Result<Self, RagError> {
        let client = rig::providers::anthropic::Client::new(api_key).map_err(|e| {
            RagError::ConfigError(format!(
                "Failed to create Anthropic client for decomposer: {e}"
            ))
        })?;
        let model = client.completion_model(model_id);

        // Format document list for the prompt.
        let document_list = document_aliases
            .iter()
            .map(|(alias, id)| format!("  - \"{alias}\" -> {id}"))
            .collect::<Vec<_>>()
            .join("\n");

        // Format person list for the prompt.
        let person_list = person_names
            .iter()
            .map(|name| format!("  - {name}"))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(Self {
            model,
            model_name: model_id.to_string(),
            document_list,
            person_list,
            prompt_template,
        })
    }

    /// Build the decomposition prompt.
    ///
    /// ## Rust Learning: format!() vs .replace() for templates
    ///
    /// `format!()` uses compile-time syntax like `{strategy:?}` — this only
    /// works on string literals the compiler can see. For runtime-loaded
    /// templates (read from a file), we use `.replace("{strategy}", &val)`
    /// which does simple string substitution at runtime.
    fn build_prompt(&self, question: &str, strategy: &RetrievalStrategy) -> String {
        // Convert strategy to a display string once — used by both paths.
        let strategy_str = format!("{strategy:?}");

        if let Some(template) = &self.prompt_template {
            // Externalized template: use .replace() for runtime substitution.
            template
                .replace("{docs}", &self.document_list)
                .replace("{persons}", &self.person_list)
                .replace("{question}", question)
                .replace("{strategy}", &strategy_str)
        } else {
            // Hardcoded default (original prompt).
            format!(
                r#"You are a legal research query planner for the Awad v. CFS/Phillips case.

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
{{
  "needs_decomposition": true/false,
  "sub_queries": [
    {{"type": "vector_search", "query": "..."}},
    {{"type": "graph_document_content", "document_id": "...", "description": "..."}},
    {{"type": "graph_person_statements", "person_id": "...", "description": "..."}},
    {{"type": "graph_contradictions", "person_name": "...", "description": "..."}}
  ]
}}

QUESTION: {question}
STRATEGY: {strategy}"#,
                docs = self.document_list,
                persons = self.person_list,
                question = question,
                strategy = strategy_str,
            )
        }
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
        let prompt = self.build_prompt(question, strategy);

        tracing::info!(
            model = %self.model_name,
            question_len = question.len(),
            "Decomposer: analyzing question"
        );

        // Call the LLM.
        let response = self
            .model
            .completion_request(&prompt)
            .max_tokens(500)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!("Decomposer LLM call failed: {e}, falling back to pass-through");
                RagError::SynthesisError(format!("Decomposer call failed: {e}"))
            });

        // On LLM error, fall back to no decomposition.
        let response = match response {
            Ok(r) => r,
            Err(_) => {
                return Ok(fallback_result(question));
            }
        };

        // Extract text from response (same pattern as synthesizer.rs).
        let answer_text: String = response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(Text { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        // Parse JSON — strip markdown fences if present.
        let clean_json = answer_text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        // Try to parse the LLM's JSON response.
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
