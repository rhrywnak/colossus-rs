//! Tests for RuleBasedRouter — all unit tests (no infrastructure needed).
//!
//! The router uses only in-memory keyword matching, so every test runs
//! without any external services.

use colossus_rag::{
    QueryRouter, RagError, RetrievalStrategy, RuleBasedRouter, ScopeFilterType,
};

// ===========================================================================
// Test helpers
// ===========================================================================

/// Create a router with the legal defaults (Awad v. CFS aliases).
fn legal_router() -> RuleBasedRouter {
    RuleBasedRouter::legal_defaults()
}

/// Helper to run the async route() in a blocking test.
///
/// ## Rust Learning: `tokio::runtime::Runtime` for sync→async bridge
///
/// Our tests are sync (`#[test]`), but `route()` is async (required by
/// the `QueryRouter` trait). We create a tiny tokio runtime just to
/// block on the future. This avoids needing `#[tokio::test]` for what
/// is essentially synchronous logic.
fn route_sync(router: &RuleBasedRouter, question: &str) -> Result<RetrievalStrategy, RagError> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(router.route(question))
}

// ===========================================================================
// Test 1: Default → Broad
// ===========================================================================

/// A question with no document or person references should route to Broad.
#[test]
fn test_broad_default() {
    let router = legal_router();
    let strategy = route_sync(&router, "What evidence supports breach of fiduciary duty?")
        .expect("Should not error");

    match strategy {
        RetrievalStrategy::Broad { node_types } => {
            assert!(node_types.is_none(), "Default Broad should have no node_type filter");
        }
        other => panic!("Expected Broad, got {other:?}"),
    }
}

// ===========================================================================
// Test 2: Focused on document reference
// ===========================================================================

/// A question mentioning a known document alias should route to Focused
/// with a Document scope filter.
#[test]
fn test_focused_document() {
    let router = legal_router();
    let strategy = route_sync(
        &router,
        "What does Phillips' discovery response say about the check?",
    )
    .expect("Should not error");

    match strategy {
        RetrievalStrategy::Focused { scope } => {
            // Should have a Document filter.
            let doc_filter = scope.iter().find(|s| s.filter_type == ScopeFilterType::Document);
            assert!(
                doc_filter.is_some(),
                "Should have a Document scope filter. Got scopes: {scope:?}"
            );
            assert_eq!(
                doc_filter.unwrap().value,
                "doc-phillips-discovery-response"
            );
        }
        other => panic!("Expected Focused, got {other:?}"),
    }
}

// ===========================================================================
// Test 3: Focused on person reference
// ===========================================================================

/// A question mentioning a known person should route to Focused with a
/// Person scope filter (when no document alias matches).
#[test]
fn test_focused_person() {
    let router = legal_router();
    let strategy = route_sync(
        &router,
        "What did Emil claim about the $50,000?",
    )
    .expect("Should not error");

    match strategy {
        RetrievalStrategy::Focused { scope } => {
            let person_filter = scope.iter().find(|s| s.filter_type == ScopeFilterType::Person);
            assert!(
                person_filter.is_some(),
                "Should have a Person scope filter. Got scopes: {scope:?}"
            );
            assert!(
                person_filter.unwrap().value.contains("Emil"),
                "Person value should contain 'Emil'"
            );
        }
        other => panic!("Expected Focused, got {other:?}"),
    }
}

// ===========================================================================
// Test 4: Hybrid on comparison with multiple scopes
// ===========================================================================

/// A question with comparison signals AND 2+ person references should
/// route to Hybrid with synthesize_across = true.
#[test]
fn test_hybrid_comparison() {
    let router = legal_router();
    let strategy = route_sync(
        &router,
        "Compare what Phillips and Marie each claimed about costs",
    )
    .expect("Should not error");

    match strategy {
        RetrievalStrategy::Hybrid {
            scopes,
            synthesize_across,
        } => {
            assert!(
                scopes.len() >= 2,
                "Should have 2+ scopes for Hybrid. Got: {scopes:?}"
            );
            assert!(
                synthesize_across,
                "Hybrid should have synthesize_across = true"
            );
        }
        other => panic!("Expected Hybrid, got {other:?}"),
    }
}

// ===========================================================================
// Test 5: Comparison signal with single scope → Focused (not Hybrid)
// ===========================================================================

/// If a comparison signal is present but only 1 scope is found, we fall
/// back to Focused — can't compare without 2+ sources.
#[test]
fn test_comparison_signal_single_scope_becomes_focused() {
    let router = legal_router();
    let strategy = route_sync(
        &router,
        "Compare all evidence from Emil about costs",
    )
    .expect("Should not error");

    match strategy {
        RetrievalStrategy::Focused { scope } => {
            assert_eq!(scope.len(), 1, "Should have exactly 1 scope");
        }
        other => panic!("Expected Focused (comparison with 1 scope), got {other:?}"),
    }
}

// ===========================================================================
// Test 6: Empty question returns error
// ===========================================================================

/// An empty question should return RagError::InvalidInput.
#[test]
fn test_empty_question_returns_error() {
    let router = legal_router();

    let result = route_sync(&router, "");
    assert!(result.is_err(), "Empty question should error");

    let result_spaces = route_sync(&router, "   ");
    assert!(result_spaces.is_err(), "Whitespace-only question should error");
}

// ===========================================================================
// Test 7: Case-insensitive matching
// ===========================================================================

/// Matching should be case-insensitive — "PHILLIPS" should match "phillips".
#[test]
fn test_case_insensitive_matching() {
    let router = legal_router();

    // Uppercase person name.
    let strategy = route_sync(
        &router,
        "What did PHILLIPS say about the money?",
    )
    .expect("Should not error");

    match strategy {
        RetrievalStrategy::Focused { scope } => {
            assert!(
                scope.iter().any(|s| s.filter_type == ScopeFilterType::Person),
                "Should match Phillips case-insensitively"
            );
        }
        other => panic!("Expected Focused for uppercase PHILLIPS, got {other:?}"),
    }
}

// ===========================================================================
// Test 8: Partial document alias match
// ===========================================================================

/// Document aliases are substring-matched. "cfs interrogatory" should
/// match even in a longer question.
#[test]
fn test_partial_document_name() {
    let router = legal_router();
    let strategy = route_sync(
        &router,
        "In the CFS interrogatory, what did they admit?",
    )
    .expect("Should not error");

    match strategy {
        RetrievalStrategy::Focused { scope } => {
            let doc = scope.iter().find(|s| s.filter_type == ScopeFilterType::Document);
            assert!(doc.is_some(), "Should match CFS interrogatory alias");
            assert_eq!(doc.unwrap().value, "doc-cfs-interrogatory-response");
        }
        other => panic!("Expected Focused for CFS interrogatory, got {other:?}"),
    }
}

// ===========================================================================
// Test 9: Short question → Broad
// ===========================================================================

/// Very short questions without recognizable names or documents should
/// default to Broad search.
#[test]
fn test_short_question_broad() {
    let router = legal_router();
    let strategy = route_sync(&router, "the check")
        .expect("Should not error");

    match strategy {
        RetrievalStrategy::Broad { .. } => {}
        other => panic!("Expected Broad for short generic question, got {other:?}"),
    }
}

// ===========================================================================
// Test 10: legal_defaults constructor has expected aliases
// ===========================================================================

/// Verify that legal_defaults() creates a router that recognizes the
/// key documents and persons in the Awad v. CFS case.
#[test]
fn test_legal_defaults_constructor() {
    let router = legal_router();

    // Should recognize "complaint" as a document.
    let strategy = route_sync(&router, "What does the complaint allege?")
        .expect("Should not error");
    match &strategy {
        RetrievalStrategy::Focused { scope } => {
            assert!(
                scope.iter().any(|s| s.value == "doc-awad-complaint"),
                "Should map 'complaint' to doc-awad-complaint"
            );
        }
        other => panic!("Expected Focused for 'complaint', got {other:?}"),
    }

    // Should recognize "Penzien" as a person.
    let strategy = route_sync(&router, "What arguments did Penzien make?")
        .expect("Should not error");
    match &strategy {
        RetrievalStrategy::Focused { scope } => {
            assert!(
                scope.iter().any(|s| s.value.contains("Penzien")),
                "Should match Penzien as a person"
            );
        }
        other => panic!("Expected Focused for Penzien, got {other:?}"),
    }
}
