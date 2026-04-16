//! Entity resolution using exact, normalized, and fuzzy string matching.
//!
//! ## Rust Learning: `#[async_trait]` on synchronous code
//!
//! The `EntityResolver` trait uses `async fn` because future implementations
//! (e.g., a `SemanticResolver` calling Qdrant for embedding similarity) will
//! need async I/O. This resolver does only in-memory string comparisons, so
//! its `async fn` completes synchronously — but we still need `#[async_trait]`
//! to satisfy the trait signature.

use async_trait::async_trait;
use strsim::jaro_winkler;

use crate::error::PipelineError;
use crate::traits::EntityResolver;
use crate::types::{ExtractedEntity, KnownEntity, ResolutionMethod, ResolvedEntity};

/// Resolves extracted entities against known entities using
/// exact, normalized, and fuzzy string matching.
///
/// ## Rust Learning: Configurable behavior via struct fields
///
/// Instead of hardcoding the fuzzy threshold, we store it as a field.
/// This lets callers tune the behavior: `NormalizedEntityResolver::new()`
/// uses a sensible default (0.85), while `with_threshold(0.90)` allows
/// stricter matching. This is the builder pattern without the full
/// builder — just one optional configuration point.
pub struct NormalizedEntityResolver {
    /// Minimum Jaro-Winkler similarity score for fuzzy matching.
    /// Range: 0.0 (no similarity) to 1.0 (identical).
    /// Default: 0.85
    fuzzy_threshold: f64,
}

impl NormalizedEntityResolver {
    pub fn new() -> Self {
        Self {
            fuzzy_threshold: 0.85,
        }
    }

    /// Set a custom fuzzy matching threshold.
    ///
    /// Higher values (e.g. 0.95) require closer matches; lower values
    /// (e.g. 0.80) are more permissive.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.fuzzy_threshold = threshold;
        self
    }
}

impl Default for NormalizedEntityResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EntityResolver for NormalizedEntityResolver {
    /// Resolve each extracted entity against the known entity list.
    ///
    /// The matching pipeline runs in order: exact → normalized → fuzzy → new.
    /// The first match wins. Non-Party entities (FactualAllegation, LegalCount,
    /// DamagesClaim) always resolve as `NewEntity` since they are
    /// document-specific and never shared across documents.
    async fn resolve(
        &self,
        entities: Vec<ExtractedEntity>,
        existing: &[KnownEntity],
    ) -> Result<Vec<ResolvedEntity>, PipelineError> {
        let mut resolved = Vec::with_capacity(entities.len());

        for extracted in &entities {
            // Non-Party entities are always new — they're document-specific
            if extracted.entity_type != "Party" {
                resolved.push(ResolvedEntity {
                    extracted: extracted.clone(),
                    matched_to: None,
                    resolution: ResolutionMethod::NewEntity,
                });
                continue;
            }

            // Filter known entities to compatible types only
            let candidates: Vec<&KnownEntity> = existing
                .iter()
                .filter(|k| compatible_type(extracted, k))
                .collect();

            if let Some(result) = self.try_resolve(extracted, &candidates) {
                resolved.push(result);
            } else {
                resolved.push(ResolvedEntity {
                    extracted: extracted.clone(),
                    matched_to: None,
                    resolution: ResolutionMethod::NewEntity,
                });
            }
        }

        Ok(resolved)
    }
}

impl NormalizedEntityResolver {
    /// Try to match an extracted entity against candidates using the
    /// three-step pipeline: exact → normalized → fuzzy.
    fn try_resolve(
        &self,
        extracted: &ExtractedEntity,
        candidates: &[&KnownEntity],
    ) -> Option<ResolvedEntity> {
        // Step 1: Exact match (case-sensitive)
        for known in candidates {
            if extracted.label == known.label {
                return Some(ResolvedEntity {
                    extracted: extracted.clone(),
                    matched_to: Some(known.id.clone()),
                    resolution: ResolutionMethod::ExactMatch,
                });
            }
        }

        let extracted_norm = normalize_name(&extracted.label);

        // Step 2: Normalized match
        for known in candidates {
            if extracted_norm == normalize_name(&known.label) {
                return Some(ResolvedEntity {
                    extracted: extracted.clone(),
                    matched_to: Some(known.id.clone()),
                    resolution: ResolutionMethod::NormalizedMatch,
                });
            }
        }

        // Step 3: Fuzzy match using Jaro-Winkler similarity
        //
        // Jaro-Winkler is a string similarity metric that gives extra weight
        // to matching prefixes. This makes it well-suited for person and
        // organization names where the first few characters are usually
        // correct (e.g. "Catholic Family Services" vs "Catholic Family Service").
        //
        // We normalize before fuzzy matching to remove noise (suffixes,
        // middle initials, casing) that would otherwise tank the similarity
        // score. For example, "Penzien & McBride, PLLC" vs "Penzien & McBride"
        // would score poorly without normalization due to the suffix.
        let mut best_score: f64 = 0.0;
        let mut best_match: Option<&KnownEntity> = None;
        let mut second_best_score: f64 = 0.0;

        for known in candidates {
            let known_norm = normalize_name(&known.label);
            let score = jaro_winkler(&extracted_norm, &known_norm);
            if score > best_score {
                second_best_score = best_score;
                best_score = score;
                best_match = Some(known);
            } else if score > second_best_score {
                second_best_score = score;
            }
        }

        if best_score >= self.fuzzy_threshold {
            // Warn on ambiguous matches — two candidates within 0.05 of each other
            if second_best_score >= self.fuzzy_threshold && (best_score - second_best_score) < 0.05
            {
                tracing::warn!(
                    entity = %extracted.label,
                    best_score = %best_score,
                    second_best_score = %second_best_score,
                    "Ambiguous fuzzy match — top two candidates within 0.05"
                );
            }

            return best_match.map(|known| ResolvedEntity {
                extracted: extracted.clone(),
                matched_to: Some(known.id.clone()),
                resolution: ResolutionMethod::FuzzyMatch,
            });
        }

        None
    }
}

/// Check whether an extracted entity and a known entity have compatible types.
///
/// ## Rust Learning: Type-level filtering
///
/// An extracted Party with `party_type: "individual"` should only match a
/// `KnownEntity` with `entity_type == "Person"`. Without this filter, a
/// person named "Phillips" could incorrectly match an organization named
/// "Phillips Corp" — the fuzzy score would be high, but the types are wrong.
fn compatible_type(extracted: &ExtractedEntity, known: &KnownEntity) -> bool {
    let party_type = extracted
        .properties
        .get("party_type")
        .and_then(|v| v.as_str())
        .unwrap_or("individual");

    match party_type {
        "individual" => known.entity_type == "Person",
        "organization" => known.entity_type == "Organization",
        _ => false,
    }
}

/// Normalize a name for comparison by removing noise that doesn't affect identity.
///
/// Steps:
/// 1. Lowercase and trim
/// 2. Strip common corporate suffixes (Inc., LLC, PLLC, Ltd., P.C.)
/// 3. Strip "The " prefix
/// 4. Remove middle initials (single letter + optional period between spaces)
/// 5. Collapse whitespace
fn normalize_name(name: &str) -> String {
    let mut s = name.to_lowercase().trim().to_string();

    // Remove common suffixes — checked in order, longest-first within
    // each suffix family to avoid partial stripping
    let suffixes = [
        ", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", pllc", " pllc", ", ltd.", " ltd.",
        ", ltd", " ltd", ", p.c.", " p.c.",
    ];
    for suffix in &suffixes {
        if s.ends_with(suffix) {
            s = s[..s.len() - suffix.len()].to_string();
            break; // only strip one suffix
        }
    }

    // Remove "the " prefix
    if s.starts_with("the ") {
        s = s[4..].to_string();
    }

    // Remove middle initials: "george r. phillips" → "george phillips"
    // Pattern: single letter followed by optional period, between spaces
    let words: Vec<&str> = s.split_whitespace().collect();
    let mut result = String::new();
    for word in &words {
        // Skip "R." style middle initials
        if word.len() == 2
            && word.ends_with('.')
            && word.chars().next().is_some_and(|c| c.is_alphabetic())
        {
            continue;
        }
        // Skip single-letter middle names like "R"
        if word.len() == 1 && word.chars().next().is_some_and(|c| c.is_alphabetic()) {
            continue;
        }
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(word);
    }

    // Collapse any remaining multiple spaces
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Helper: build an ExtractedEntity with party_type
    fn party(label: &str, party_type: &str) -> ExtractedEntity {
        ExtractedEntity {
            entity_type: "Party".to_string(),
            id: Some(format!("ext-{}", label.to_lowercase().replace(' ', "-"))),
            label: label.to_string(),
            properties: json!({ "party_type": party_type }),
            verbatim_quote: None,
            page_number: None,
            grounding_status: None,
        }
    }

    /// Helper: build a non-Party ExtractedEntity
    fn non_party(label: &str, entity_type: &str) -> ExtractedEntity {
        ExtractedEntity {
            entity_type: entity_type.to_string(),
            id: Some(format!("ext-{}", label.to_lowercase().replace(' ', "-"))),
            label: label.to_string(),
            properties: json!({}),
            verbatim_quote: None,
            page_number: None,
            grounding_status: None,
        }
    }

    /// Helper: build a KnownEntity
    fn known(id: &str, label: &str, entity_type: &str) -> KnownEntity {
        KnownEntity {
            entity_type: entity_type.to_string(),
            id: id.to_string(),
            label: label.to_string(),
            properties: json!({}),
        }
    }

    // 1. Exact match — "Marie Awad" matches "Marie Awad"
    #[tokio::test]
    async fn test_exact_match() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![party("Marie Awad", "individual")];
        let existing = vec![known("p1", "Marie Awad", "Person")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, Some("p1".to_string()));
        assert_eq!(results[0].resolution, ResolutionMethod::ExactMatch);
    }

    // 2. Normalized match — "MARIE AWAD" matches "Marie Awad"
    #[tokio::test]
    async fn test_normalized_match_case() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![party("MARIE AWAD", "individual")];
        let existing = vec![known("p1", "Marie Awad", "Person")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, Some("p1".to_string()));
        assert_eq!(results[0].resolution, ResolutionMethod::NormalizedMatch);
    }

    // 3. Normalized match — "George R. Phillips" matches "George Phillips"
    #[tokio::test]
    async fn test_normalized_match_middle_initial() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![party("George R. Phillips", "individual")];
        let existing = vec![known("p2", "George Phillips", "Person")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, Some("p2".to_string()));
        assert_eq!(results[0].resolution, ResolutionMethod::NormalizedMatch);
    }

    // 4. Fuzzy match — "Catholic Family Services" matches "Catholic Family Service"
    #[tokio::test]
    async fn test_fuzzy_match() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![party("Catholic Family Services", "organization")];
        let existing = vec![known("o1", "Catholic Family Service", "Organization")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, Some("o1".to_string()));
        assert_eq!(results[0].resolution, ResolutionMethod::FuzzyMatch);
    }

    // 5. No match — "Jeff Sharp" against empty existing list
    #[tokio::test]
    async fn test_no_match_empty_list() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![party("Jeff Sharp", "individual")];
        let existing: Vec<KnownEntity> = vec![];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, None);
        assert_eq!(results[0].resolution, ResolutionMethod::NewEntity);
    }

    // 6. Type filtering — individual Party doesn't match Organization
    #[tokio::test]
    async fn test_type_filtering() {
        let resolver = NormalizedEntityResolver::new();
        // "Phillips" as an individual should NOT match "Phillips" as an Organization
        let entities = vec![party("Phillips", "individual")];
        let existing = vec![known("o1", "Phillips", "Organization")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, None);
        assert_eq!(results[0].resolution, ResolutionMethod::NewEntity);
    }

    // 7. Non-Party entities always resolve as NewEntity
    #[tokio::test]
    async fn test_non_party_always_new() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![
            non_party("Breach of contract", "LegalCount"),
            non_party("Defendant failed to pay", "FactualAllegation"),
        ];
        // Even though we have known entities, non-Party types skip matching
        let existing = vec![known("p1", "Breach of contract", "Person")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].resolution, ResolutionMethod::NewEntity);
        assert_eq!(results[1].resolution, ResolutionMethod::NewEntity);
    }

    // 8. Multiple entities in one batch — mix of matched and new
    #[tokio::test]
    async fn test_batch_mixed() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![
            party("Marie Awad", "individual"),
            party("Unknown Person", "individual"),
            non_party("Count 1: Fraud", "LegalCount"),
        ];
        let existing = vec![
            known("p1", "Marie Awad", "Person"),
            known("p2", "George Phillips", "Person"),
        ];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].resolution, ResolutionMethod::ExactMatch);
        assert_eq!(results[0].matched_to, Some("p1".to_string()));
        assert_eq!(results[1].resolution, ResolutionMethod::NewEntity);
        assert_eq!(results[1].matched_to, None);
        assert_eq!(results[2].resolution, ResolutionMethod::NewEntity);
    }

    // 9. Ambiguous near-match — two candidates with similar scores (logs warning)
    #[tokio::test]
    async fn test_ambiguous_near_match() {
        let resolver = NormalizedEntityResolver::new();
        // Two very similar organization names — should still pick the best
        let entities = vec![party("Catholic Family Svc", "organization")];
        let existing = vec![
            known("o1", "Catholic Family Service", "Organization"),
            known("o2", "Catholic Family Services", "Organization"),
        ];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        // Should still resolve (picks the best), just logs a warning
        assert_eq!(results[0].resolution, ResolutionMethod::FuzzyMatch);
        assert!(results[0].matched_to.is_some());
    }

    // 10. Suffix stripping — "Penzien & McBride, PLLC" matches "Penzien & McBride"
    #[tokio::test]
    async fn test_suffix_stripping() {
        let resolver = NormalizedEntityResolver::new();
        let entities = vec![party("Penzien & McBride, PLLC", "organization")];
        let existing = vec![known("o1", "Penzien & McBride", "Organization")];

        let results = resolver.resolve(entities, &existing).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_to, Some("o1".to_string()));
        assert_eq!(results[0].resolution, ResolutionMethod::NormalizedMatch);
    }

    // Verify normalize_name directly
    #[test]
    fn test_normalize_name_cases() {
        assert_eq!(normalize_name("Marie Awad"), "marie awad");
        assert_eq!(normalize_name("MARIE AWAD"), "marie awad");
        assert_eq!(normalize_name("George R. Phillips"), "george phillips");
        assert_eq!(normalize_name("George R Phillips"), "george phillips");
        assert_eq!(
            normalize_name("Penzien & McBride, PLLC"),
            "penzien & mcbride"
        );
        assert_eq!(normalize_name("The Law Firm"), "law firm");
        assert_eq!(normalize_name("Acme Corp, Inc."), "acme corp");
        assert_eq!(normalize_name("Smith & Jones, Ltd."), "smith & jones");
    }
}
