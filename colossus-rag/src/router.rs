//! RuleBasedRouter — implements [`QueryRouter`] using keyword matching.
//!
//! This is v1 of the router: simple, fast, no API calls. It analyzes the
//! user's question text to decide HOW to search for relevant context.
//!
//! ## Why route at all? Why not just search everything?
//!
//! Without routing, every question does a broad vector search across the
//! entire knowledge graph. This works but has drawbacks:
//!
//! 1. **Noise**: A question about "Phillips' testimony" returns results from
//!    all witnesses, diluting the relevant context.
//! 2. **Cost**: Broader searches return more chunks → more tokens → higher
//!    Claude API costs.
//! 3. **Quality**: Focused context produces better answers. If you ask about
//!    one deposition, Claude shouldn't be distracted by unrelated documents.
//!
//! The router adds intelligence: "What did Phillips say about the check?"
//! → search only Phillips' testimony, not the entire case.
//!
//! ## Routing approaches (why rule-based for v1)
//!
//! | Approach | Pros | Cons |
//! |----------|------|------|
//! | Rule-based (this) | Fast, free, deterministic, testable | Limited flexibility |
//! | Semantic (LLM) | Understands nuance, handles ambiguity | Slow, costs money per query |
//! | Agentic (multi-step) | Can ask clarifying questions | Complex, highest latency |
//!
//! We start with rule-based because it ships fast, costs nothing, and handles
//! the common patterns in legal case analysis. If users hit edge cases where
//! rules fail, we can upgrade to semantic routing later.
//!
//! ## Rule priority
//!
//! 1. Empty question → `RagError::InvalidInput`
//! 2. Comparison signal + 2+ scopes → `Hybrid`
//! 3. Document reference → `Focused(Document)`
//! 4. Person reference → `Focused(Person)`
//! 5. Default → `Broad`

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::error::RagError;
use crate::traits::QueryRouter;
use crate::types::{RetrievalStrategy, ScopeFilter, ScopeFilterType};

// ---------------------------------------------------------------------------
// Comparison signal words
// ---------------------------------------------------------------------------

/// Words/phrases that indicate the user wants to compare multiple sources.
///
/// When these appear AND 2+ scopes are detected, we use Hybrid strategy
/// so Claude synthesizes across the scopes rather than treating them
/// independently.
const COMPARISON_SIGNALS: &[&str] = &[
    "compare",
    "comparison",
    "vs",
    "versus",
    "contradict",
    "contradiction",
    "differ",
    "difference",
    "inconsistent",
    "inconsistency",
    "conflict",
    "disagree",
    "both",
    "each claim",
];

// ---------------------------------------------------------------------------
// RuleBasedRouter struct
// ---------------------------------------------------------------------------

/// Routes questions to retrieval strategies using keyword matching.
///
/// Holds two lookup tables configured at construction:
/// - **Document aliases**: Maps short names ("phillips discovery") to
///   document IDs ("doc-phillips-discovery-response") for scoped search.
/// - **Person names**: A set of known full names ("George Phillips") split
///   into searchable parts for person-scoped search.
///
/// ## Rust Learning: `HashMap` for alias lookup
///
/// `HashMap<String, String>` gives O(1) average lookup time for alias
/// matching. We lowercase both the alias keys and the query text, so
/// matching is always case-insensitive.
///
/// ## Rust Learning: `HashSet` for O(1) membership tests
///
/// We don't use `HashSet` directly for person matching (we need to check
/// individual name parts), but the concept is the same — we precompute
/// searchable name parts at construction time to avoid recomputing on
/// every query.
pub struct RuleBasedRouter {
    /// Lowercase alias → document ID.
    /// Example: "phillips discovery" → "doc-phillips-discovery-response"
    document_aliases: HashMap<String, String>,

    /// Known person entries: (full_name, Vec<lowercase_name_parts>).
    /// Example: ("George Phillips", vec!["george", "phillips"])
    ///
    /// We store the full name for use in ScopeFilter.value and the
    /// lowercase parts for matching against query words.
    person_entries: Vec<(String, Vec<String>)>,
}

impl RuleBasedRouter {
    /// Create a router with custom document aliases and person names.
    ///
    /// ## Parameters
    /// - `document_aliases`: Maps lowercase alias strings to document IDs.
    ///   The router checks if any alias appears as a substring in the
    ///   lowercased query.
    /// - `person_names`: Full names of known persons (e.g., "George Phillips").
    ///   The router splits each name into parts and checks if any part
    ///   appears as a whole word in the query.
    pub fn new(
        document_aliases: HashMap<String, String>,
        person_names: Vec<String>,
    ) -> Self {
        // Pre-compute lowercase name parts for efficient matching.
        let person_entries = person_names
            .into_iter()
            .map(|name| {
                let parts: Vec<String> = name.to_lowercase()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
                (name, parts)
            })
            .collect();

        Self {
            document_aliases,
            person_entries,
        }
    }

    /// Create a router pre-configured with Awad v. CFS case aliases.
    ///
    /// This is a convenience constructor for the specific legal case in
    /// colossus-legal. For other cases, use `new()` with custom aliases.
    pub fn legal_defaults() -> Self {
        let mut aliases = HashMap::new();

        // Phillips documents
        aliases.insert("phillips discovery".into(), "doc-phillips-discovery-response".into());
        aliases.insert("phillips response".into(), "doc-phillips-discovery-response".into());

        // CFS documents
        aliases.insert("cfs interrogatory".into(), "doc-cfs-interrogatory-response".into());
        aliases.insert("cfs response".into(), "doc-cfs-interrogatory-response".into());

        // Complaint
        aliases.insert("complaint".into(), "doc-awad-complaint".into());
        aliases.insert("awad complaint".into(), "doc-awad-complaint".into());

        // Appeal briefs
        aliases.insert("penzien reply".into(), "doc-penzien-reply-brief-310660".into());
        aliases.insert("reply brief".into(), "doc-penzien-reply-brief-310660".into());
        aliases.insert("penzien brief".into(), "doc-penzien-coa-brief-300891".into());
        aliases.insert("penzien appeal".into(), "doc-penzien-coa-brief-300891".into());
        aliases.insert("appellant brief".into(), "doc-penzien-coa-brief-300891".into());
        aliases.insert("phillips coa".into(), "doc-phillips-coa-response-300891".into());
        aliases.insert("phillips appeal".into(), "doc-phillips-coa-response-300891".into());
        aliases.insert("appellee response".into(), "doc-phillips-coa-response-300891".into());

        let persons = vec![
            "George Phillips".into(),
            "Emil Awad".into(),
            "Marie Awad".into(),
            "Charles Penzien".into(),
            "Catholic Family Service".into(),
        ];

        Self::new(aliases, persons)
    }
}

// ---------------------------------------------------------------------------
// QueryRouter trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl QueryRouter for RuleBasedRouter {
    /// Analyze the question and choose a retrieval strategy.
    ///
    /// ## Algorithm
    /// 1. Reject empty questions
    /// 2. Lowercase the query for case-insensitive matching
    /// 3. Extract all scopes (document refs + person refs)
    /// 4. Check for comparison signals
    /// 5. Apply priority rules to choose strategy
    async fn route(&self, question: &str) -> Result<RetrievalStrategy, RagError> {
        let trimmed = question.trim();
        if trimmed.is_empty() {
            return Err(RagError::InvalidInput("Question must not be empty".into()));
        }

        // Strip apostrophes and possessives for cleaner matching.
        // "Phillips' discovery" → "phillips discovery" so it matches the alias.
        let query_lower = trimmed.to_lowercase().replace('\'', "");

        // Extract all matching scopes from the query.
        let scopes = extract_scopes(&query_lower, &self.document_aliases, &self.person_entries);
        let has_comparison = contains_comparison_signal(&query_lower);

        // Apply priority rules.
        match (has_comparison, scopes.len()) {
            // Comparison signal + 2+ scopes → Hybrid.
            (true, n) if n >= 2 => {
                tracing::debug!(
                    scopes = scopes.len(),
                    "Router: Hybrid — comparison with multiple scopes"
                );
                Ok(RetrievalStrategy::Hybrid {
                    scopes,
                    synthesize_across: true,
                })
            }
            // Scopes found (1 or more, or comparison with only 1) → Focused.
            (_, n) if n >= 1 => {
                tracing::debug!(
                    scopes = n,
                    "Router: Focused — {} scope(s) detected",
                    n
                );
                Ok(RetrievalStrategy::Focused { scope: scopes })
            }
            // No scopes, no comparison → Broad.
            _ => {
                tracing::debug!("Router: Broad — no specific scopes detected");
                Ok(RetrievalStrategy::Broad { node_types: None })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: extract all scopes from the query
// ---------------------------------------------------------------------------

/// Extract document and person scope filters from the query text.
///
/// Document aliases are checked first (higher priority — more specific).
/// Then person names are checked. All matches are returned so the caller
/// can decide between Focused (1 scope) and Hybrid (2+ scopes).
///
/// ## Rust Learning: Substring matching for documents vs word matching for persons
///
/// Documents use substring matching because aliases are multi-word phrases
/// ("phillips discovery" should match "What does Phillips' discovery response say?").
///
/// Persons use word-boundary matching because single-word names like "Phillips"
/// could false-match document aliases. We check that the name part appears as
/// a standalone word (surrounded by spaces or string boundaries).
fn extract_scopes(
    query_lower: &str,
    document_aliases: &HashMap<String, String>,
    person_entries: &[(String, Vec<String>)],
) -> Vec<ScopeFilter> {
    let mut scopes = Vec::new();
    let mut matched_doc_ids: HashSet<String> = HashSet::new();

    // Check document aliases (substring match).
    for (alias, doc_id) in document_aliases {
        if query_lower.contains(alias.as_str()) && matched_doc_ids.insert(doc_id.clone()) {
            scopes.push(ScopeFilter {
                filter_type: ScopeFilterType::Document,
                value: doc_id.clone(),
            });
        }
    }

    // Check person names (word-boundary match on any name part).
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    for (full_name, parts) in person_entries {
        // A person matches if ANY of their name parts appears as a whole word.
        // "Phillips" matches in "What did Phillips claim?"
        // but NOT in "Phillipsburg" (because we check whole words).
        let matched = parts.iter().any(|part| query_words.contains(&part.as_str()));
        if matched {
            scopes.push(ScopeFilter {
                filter_type: ScopeFilterType::Person,
                value: full_name.clone(),
            });
        }
    }

    scopes
}

// ---------------------------------------------------------------------------
// Helper: check for comparison signals
// ---------------------------------------------------------------------------

/// Check if the query contains words suggesting a comparison between sources.
///
/// Uses simple substring matching against a predefined list of signal words.
/// This is intentionally broad — false positives just trigger Hybrid mode,
/// which still works fine (it searches multiple scopes and asks Claude to
/// synthesize across them).
fn contains_comparison_signal(query_lower: &str) -> bool {
    COMPARISON_SIGNALS.iter().any(|signal| query_lower.contains(signal))
}
