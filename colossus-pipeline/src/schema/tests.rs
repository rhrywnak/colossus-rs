//! colossus-pipeline/src/schema_tests.rs
//!
//! Tests for schema types, enums, and PipelineError.
//!
//! ## Rust Learning: #[cfg(test)] and test modules
//!
//! #[cfg(test)] tells the compiler to include this code only when running
//! tests -- it is stripped from release builds entirely. Tests live in a
//! separate file here (schema_tests.rs) rather than inline in schema.rs
//! because schema.rs is already substantive. Both patterns are idiomatic Rust.
//! The `use super::*` import brings all items from the parent module into scope.

use super::*;

// -- JobStatus / JobControl wire-format pins ---------------------------------

/// Wire-format pin for the `pipeline_jobs.status` column. These strings are
/// consumed by external dashboards/log queries with no compile-time check
/// (per CLAUDE.md "as_str() is the single source of truth" rule).
#[test]
fn job_status_serializes_to_expected_strings() {
    let cases: &[(JobStatus, &str)] = &[
        (JobStatus::Ready, "\"ready\""),
        (JobStatus::Running, "\"running\""),
        (JobStatus::Completed, "\"completed\""),
        (JobStatus::Failed, "\"failed\""),
        (JobStatus::Cancelled, "\"cancelled\""),
    ];
    for (variant, expected) in cases {
        assert_eq!(
            serde_json::to_string(variant).unwrap(),
            *expected,
            "case: {variant:?}",
        );
    }
}

/// Wire-format pin for the `pipeline_jobs.control` column.
#[test]
fn job_control_serializes_to_expected_strings() {
    let cases: &[(JobControl, &str)] = &[
        (JobControl::None, "\"none\""),
        (JobControl::CancelRequested, "\"cancel_requested\""),
        (JobControl::WaitingForInput, "\"waiting_for_input\""),
    ];
    for (variant, expected) in cases {
        assert_eq!(
            serde_json::to_string(variant).unwrap(),
            *expected,
            "case: {variant:?}",
        );
    }
}

/// Pin design contract — strict deserialization rejects unknown strings
/// (no fuzzy matching that would silently pass invalid status values).
#[test]
fn job_status_unknown_string_deserialize_fails() {
    let result: Result<JobStatus, _> = serde_json::from_str("\"processing\"");
    assert!(
        result.is_err(),
        "Unknown status string must not deserialize successfully"
    );
}

/// Pin design contract — strict deserialization for JobControl.
#[test]
fn job_control_unknown_string_deserialize_fails() {
    let result: Result<JobControl, _> = serde_json::from_str("\"pause\"");
    assert!(
        result.is_err(),
        "Unknown control string must not deserialize successfully"
    );
}

// -- JobStatus::as_str -------------------------------------------------------

#[test]
fn job_status_as_str_matches_serde_serialization() {
    // as_str() and serde serialization must always agree.
    // If they diverge, SQL binds and JSON responses show different values.
    assert_eq!(JobStatus::Ready.as_str(), "ready");
    assert_eq!(JobStatus::Running.as_str(), "running");
    assert_eq!(JobStatus::Completed.as_str(), "completed");
    assert_eq!(JobStatus::Failed.as_str(), "failed");
    assert_eq!(JobStatus::Cancelled.as_str(), "cancelled");
}

// -- JobControl::as_str ------------------------------------------------------

#[test]
fn job_control_as_str_matches_serde_serialization() {
    assert_eq!(JobControl::None.as_str(), "none");
    assert_eq!(JobControl::CancelRequested.as_str(), "cancel_requested");
    assert_eq!(JobControl::WaitingForInput.as_str(), "waiting_for_input");
}

// -- PipelineError ------------------------------------------------------------

use crate::error::PipelineError;

#[test]
fn pipeline_error_duplicate_job_displays_type_and_key() {
    let e = PipelineError::DuplicateJob {
        job_type: "document_processing".to_string(),
        job_key: "doc-001".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("document_processing"));
    assert!(msg.contains("doc-001"));
}

#[test]
fn pipeline_error_invalid_transition_displays_from_and_to() {
    let e = PipelineError::InvalidTransition {
        from: "ready".to_string(),
        to: "completed".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("ready"));
    assert!(msg.contains("completed"));
}
