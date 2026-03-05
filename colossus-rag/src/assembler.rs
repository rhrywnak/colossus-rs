//! LegalAssembler — implements [`ContextAssembler`] for legal case analysis.
//!
//! This module formats retrieved context chunks into an LLM-ready prompt
//! string with system instructions, formatted evidence, and citation rules.
//!
//! It replaces: the `build_system_prompt()` and `format_node()` functions
//! in `colossus-legal/backend/src/api/ask.rs` and `graph_expander.rs`.
//!
//! ## Why this is synchronous (not async)
//!
//! The [`ContextAssembler`] trait is the ONLY synchronous trait in the pipeline.
//! It does purely in-memory work:
//! 1. String concatenation and formatting
//! 2. Token count estimation (a simple heuristic)
//! 3. Score-based chunk selection
//!
//! No network calls, no disk I/O, no database queries. Making this async
//! would add unnecessary overhead — the `Box<dyn Future>` heap allocation,
//! the `Send` bounds, and the async runtime overhead — for work that
//! completes in microseconds.
//!
//! ## Prompt engineering principles
//!
//! The system prompt is carefully designed for legal analysis:
//!
//! 1. **Evidence-only answers**: Claude must NOT infer facts not in the evidence.
//!    This is critical for legal work — hallucinated facts are worse than no answer.
//!
//! 2. **Citation by evidence ID**: Every factual claim must cite the specific
//!    evidence ID (e.g., `evidence-phillips-q74`). This enables verification
//!    and audit trails.
//!
//! 3. **Contradiction handling**: When evidence conflicts (common in depositions),
//!    Claude must note the contradiction and attribute each position to its source.
//!
//! 4. **Honest uncertainty**: If evidence is insufficient, Claude must say so
//!    rather than speculate. This builds trust with legal professionals.
//!
//! ## Token estimation
//!
//! We use the standard approximation of **1 token ≈ 4 characters** for English
//! text. This is not exact (tokenizers like BPE can vary), but it's sufficient
//! for our purposes:
//! - Estimating context size for `PipelineStats`
//! - Budget-based chunk truncation (approximate is fine — we're not trying to
//!   hit an exact token limit, just avoid sending grossly oversized contexts)

use crate::traits::ContextAssembler;
use crate::types::{AssembledContext, ContextChunk};

/// Average characters per token for English text.
/// This is a standard approximation used across the industry.
/// GPT/Claude tokenizers average ~3.5–4.5 chars/token for English.
const CHARS_PER_TOKEN: usize = 4;

// ---------------------------------------------------------------------------
// Default system prompt
// ---------------------------------------------------------------------------

/// The default system prompt template for legal case analysis.
///
/// This prompt establishes Claude's role and rules for handling evidence.
/// The `{context}` placeholder is replaced with the formatted evidence sections.
///
/// ## Prompt engineering: Why these specific rules?
///
/// - **Rule 1** (evidence-only): Prevents hallucination — the #1 risk in legal AI.
/// - **Rule 2** (cite by ID): Enables the frontend to link citations to evidence nodes.
/// - **Rule 3** (contradictions): Depositions often contradict each other; Claude
///   must surface this rather than silently picking a side.
/// - **Rule 4** (honest uncertainty): Better to say "I don't know" than fabricate.
/// - **Rule 5** (plain language): The end user may not be a lawyer.
/// - **Rule 6** (enumerate patterns): Prevents vague claims like "Phillips repeatedly..."
///   without backing each instance.
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a legal research assistant analyzing case evidence.

You have been given evidence from a case knowledge graph, including verbatim quotes from sworn testimony, court filings, and documentary evidence. Each piece of evidence includes its source document and page number where available.

RULES:
1. Answer using ONLY the provided evidence. Do not infer facts not present in the evidence.
2. For every factual claim in your answer, cite the specific evidence ID in parentheses, e.g. (evidence-phillips-q74).
3. When evidence items contradict each other, note the contradiction explicitly and identify which party made each statement.
4. If the provided evidence does not contain enough information to answer the question, say so clearly. Do not speculate.
5. Use plain language accessible to a non-lawyer, but maintain legal precision for citations.
6. When describing patterns (e.g., "Phillips repeatedly..."), list each specific instance with its citation."#;

/// Message appended to the system prompt when no evidence chunks are available.
const NO_EVIDENCE_MESSAGE: &str = "\n\nNo relevant evidence was found for this question. \
Inform the user that no matching evidence was found in the knowledge graph, \
and suggest they rephrase their question or broaden the search scope.";

// ---------------------------------------------------------------------------
// LegalAssembler struct
// ---------------------------------------------------------------------------

/// Formats context chunks into an LLM-ready prompt for legal case analysis.
///
/// The assembler is stateless — the only configuration is an optional custom
/// system prompt template. If not provided, it uses a battle-tested default
/// prompt designed for legal evidence analysis.
///
/// ## Rust Learning: Unit struct vs struct with fields
///
/// `LegalAssembler` has one optional field (the custom system prompt).
/// If we had no configurable state at all, we could use a unit struct
/// like `NoOpRouter`. But the custom prompt option justifies a regular struct.
pub struct LegalAssembler {
    /// Custom system prompt, or None to use the default.
    /// The prompt should include instructions for how Claude should handle
    /// evidence citations and contradictions.
    system_prompt: String,
}

impl LegalAssembler {
    /// Create a new LegalAssembler with the default legal analysis system prompt.
    pub fn new() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Create a LegalAssembler with a custom system prompt.
    ///
    /// Use this when you need different instructions for Claude — for example,
    /// a different case name, different citation format, or a non-legal domain.
    pub fn with_system_prompt(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
        }
    }
}

/// ## Rust Learning: `Default` trait implementation
///
/// Implementing `Default` lets callers write `LegalAssembler::default()`
/// which is equivalent to `LegalAssembler::new()`. This is idiomatic Rust —
/// many generic containers and builders expect `T: Default`.
impl Default for LegalAssembler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ContextAssembler trait implementation
// ---------------------------------------------------------------------------

impl ContextAssembler for LegalAssembler {
    /// Format context chunks into an assembled prompt for Claude.
    ///
    /// ## Algorithm
    /// 1. Sort chunks by score (highest first) — best evidence gets priority
    /// 2. Format each chunk into a text section with ID, title, content, source
    /// 3. Accumulate sections until the token budget is exhausted
    /// 4. Build the system prompt with the formatted context appended
    /// 5. Estimate total token count
    ///
    /// ## Parameters
    /// - `question`: The user's question (not currently included in the context,
    ///   but available for future use — e.g., highlighting relevant passages)
    /// - `chunks`: The context chunks from vector search and graph expansion
    /// - `max_tokens`: Token budget for the context (drop low-scored chunks if over)
    fn assemble(
        &self,
        _question: &str,
        chunks: &[ContextChunk],
        max_tokens: usize,
    ) -> AssembledContext {
        // Handle empty chunks — tell Claude no evidence was found.
        if chunks.is_empty() {
            let system_prompt = format!("{}{}", self.system_prompt, NO_EVIDENCE_MESSAGE);
            let token_estimate = estimate_tokens(&system_prompt);
            return AssembledContext {
                system_prompt,
                formatted_context: String::new(),
                token_estimate,
            };
        }

        // Sort by score descending so highest-relevance chunks come first.
        // We clone because the trait receives `&[ContextChunk]` (immutable).
        //
        // ## Rust Learning: `sort_by` with f32 comparison
        //
        // f32 doesn't implement `Ord` (because NaN breaks total ordering),
        // so we can't use `.sort()`. Instead, `.sort_by()` with
        // `partial_cmp().unwrap_or(Equal)` handles NaN safely by treating
        // NaN as equal to everything (it won't appear in practice — scores
        // are always valid cosine similarity values).
        let mut sorted_chunks: Vec<&ContextChunk> = chunks.iter().collect();
        sorted_chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Format chunks within the token budget.
        let mut sections: Vec<String> = Vec::new();
        let mut current_tokens = estimate_tokens(&self.system_prompt);

        for chunk in &sorted_chunks {
            let section = format_chunk(chunk);
            let section_tokens = estimate_tokens(&section);

            if current_tokens + section_tokens > max_tokens {
                // Budget exhausted — stop adding chunks.
                // The remaining (lower-scored) chunks are dropped.
                tracing::debug!(
                    dropped_chunks = sorted_chunks.len() - sections.len(),
                    "Token budget reached, dropping remaining chunks"
                );
                break;
            }

            current_tokens += section_tokens;
            sections.push(section);
        }

        let formatted_context = sections.join("\n");
        let system_prompt = self.system_prompt.clone();
        let token_estimate = estimate_tokens(&system_prompt) + estimate_tokens(&formatted_context);

        AssembledContext {
            system_prompt,
            formatted_context,
            token_estimate,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: format a single chunk
// ---------------------------------------------------------------------------

/// Format a single [`ContextChunk`] into a text section for the LLM prompt.
///
/// Produces output like:
/// ```text
/// === EVIDENCE: evidence-phillips-q74 ===
/// Title: Phillips: Emil wanted $50K returned
/// Content: Q: Did Emil ever ask for the money back? A: Yes...
/// Source: Phillips Deposition, p.42
/// Score: 0.7056
/// Related: → SUPPORTS harm-003 (Harm)
/// ```
///
/// ## Design choices
///
/// - **`=== TYPE: ID ===` delimiters**: Match the existing colossus-legal format.
///   The triple-equals are visually distinct and easy for the LLM to parse.
/// - **Score included**: Helps Claude weigh evidence — higher-scored chunks
///   are more relevant to the query.
/// - **Source line**: Combines document title + page number on one line for brevity.
/// - **Related nodes**: Shows graph relationships so Claude understands connections
///   between evidence (e.g., evidence SUPPORTS a harm claim).
pub fn format_chunk(chunk: &ContextChunk) -> String {
    let mut s = String::new();

    // Header line with node type and ID.
    s.push_str(&format!(
        "=== {}: {} ===\n",
        chunk.node_type.to_uppercase(),
        chunk.node_id
    ));

    // Title.
    s.push_str(&format!("Title: {}\n", chunk.title));

    // Content (the main text for LLM reasoning).
    if !chunk.content.is_empty() && chunk.content != chunk.title {
        s.push_str(&format!("Content: {}\n", chunk.content));
    }

    // Source reference (document + page).
    let source_line = format_source(&chunk.source);
    if !source_line.is_empty() {
        s.push_str(&format!("Source: {source_line}\n"));
    }

    // Verbatim quote if available (separate from content).
    if let Some(quote) = &chunk.source.verbatim_quote {
        s.push_str(&format!("Quote: \"{quote}\"\n"));
    }

    // Similarity score (helps Claude weigh evidence).
    s.push_str(&format!("Score: {:.4}\n", chunk.score));

    // Graph relationships (from expander).
    for rel in &chunk.relationships {
        let arrow = match rel.direction {
            crate::types::RelationDirection::Outbound => "→",
            crate::types::RelationDirection::Inbound => "←",
        };
        s.push_str(&format!(
            "Related: {} {} {} ({})\n",
            arrow, rel.relationship, rel.node_id, rel.node_type
        ));
    }

    s
}

/// Format a [`SourceReference`] into a compact "Document, p.N" string.
fn format_source(source: &crate::types::SourceReference) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(title) = &source.document_title {
        parts.push(title.clone());
    }

    if let Some(page) = source.page_number {
        parts.push(format!("p.{page}"));
    }

    parts.join(", ")
}

// ---------------------------------------------------------------------------
// Helper: token estimation
// ---------------------------------------------------------------------------

/// Estimate the token count for a text string.
///
/// Uses the standard approximation: **1 token ≈ 4 characters**.
///
/// This is intentionally simple. More accurate estimation would require
/// a tokenizer library (like `tiktoken-rs`), which adds a dependency
/// for marginal benefit. Our use cases (PipelineStats, budget truncation)
/// only need order-of-magnitude accuracy.
pub fn estimate_tokens(text: &str) -> usize {
    // Integer division rounds down, which is slightly conservative.
    // For an empty string, returns 0.
    text.len() / CHARS_PER_TOKEN
}
