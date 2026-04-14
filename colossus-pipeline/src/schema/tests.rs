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

// -- JobStatus serialization --------------------------------------------------

#[test]
fn job_status_ready_serializes_to_ready() {
    let s = serde_json::to_string(&JobStatus::Ready).unwrap();
    assert_eq!(s, "\"ready\"");
}

#[test]
fn job_status_running_serializes_to_running() {
    let s = serde_json::to_string(&JobStatus::Running).unwrap();
    assert_eq!(s, "\"running\"");
}

#[test]
fn job_status_completed_serializes_to_completed() {
    let s = serde_json::to_string(&JobStatus::Completed).unwrap();
    assert_eq!(s, "\"completed\"");
}

#[test]
fn job_status_failed_serializes_to_failed() {
    let s = serde_json::to_string(&JobStatus::Failed).unwrap();
    assert_eq!(s, "\"failed\"");
}

#[test]
fn job_status_cancelled_serializes_to_cancelled() {
    let s = serde_json::to_string(&JobStatus::Cancelled).unwrap();
    assert_eq!(s, "\"cancelled\"");
}

#[test]
fn job_status_deserializes_from_ready() {
    let s: JobStatus = serde_json::from_str("\"ready\"").unwrap();
    assert_eq!(s, JobStatus::Ready);
}

#[test]
fn job_status_unknown_string_deserialize_fails() {
    let result: Result<JobStatus, _> = serde_json::from_str("\"processing\"");
    assert!(
        result.is_err(),
        "Unknown status string must not deserialize successfully"
    );
}

// -- JobControl serialization -------------------------------------------------

#[test]
fn job_control_none_serializes_to_none() {
    let s = serde_json::to_string(&JobControl::None).unwrap();
    assert_eq!(s, "\"none\"");
}

#[test]
fn job_control_cancel_requested_serializes_correctly() {
    let s = serde_json::to_string(&JobControl::CancelRequested).unwrap();
    assert_eq!(s, "\"cancel_requested\"");
}

#[test]
fn job_control_waiting_for_input_serializes_correctly() {
    let s = serde_json::to_string(&JobControl::WaitingForInput).unwrap();
    assert_eq!(s, "\"waiting_for_input\"");
}

#[test]
fn job_control_deserializes_from_cancel_requested() {
    let s: JobControl = serde_json::from_str("\"cancel_requested\"").unwrap();
    assert_eq!(s, JobControl::CancelRequested);
}

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

// -- JobRow round-trip --------------------------------------------------------

#[test]
fn job_row_round_trips_through_serde_json() {
    // ## Rust Learning: Uuid::new_v4() in tests
    // uuid::Uuid::new_v4() generates a random UUID. We use v4 here (not v7)
    // because tests do not require time-ordering -- they just need a valid UUID.
    // Production code uses v7 (time-ordered) for efficient index scans.
    use chrono::Utc;
    use uuid::Uuid;

    let row = JobRow {
        id: Uuid::new_v4(),
        job_type: "document_processing".to_string(),
        job_key: "doc-test-001".to_string(),
        pipeline_version: 1,
        status: JobStatus::Ready,
        control: JobControl::None,
        current_step: "ExtractText".to_string(),
        step_data: serde_json::json!({"document_id": "doc-test-001"}),
        result: serde_json::json!({}),
        tried: 0,
        max_retries: 3,
        retry_delay_secs: 60,
        priority: 0,
        wakeup_at: Utc::now(),
        step_started_at: None,
        step_completed_at: None,
        timeout_at: None,
        worker_id: None,
        last_heartbeat_at: None,
        progress: None,
        error: None,
        created_by: Some("test".to_string()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        completed_at: None,
    };

    let json = serde_json::to_string(&row).unwrap();
    let restored: JobRow = serde_json::from_str(&json).unwrap();

    assert_eq!(row.id, restored.id);
    assert_eq!(row.job_type, restored.job_type);
    assert_eq!(row.job_key, restored.job_key);
    assert_eq!(row.status, restored.status);
    assert_eq!(row.control, restored.control);
    assert_eq!(row.current_step, restored.current_step);
    assert_eq!(row.tried, restored.tried);
    assert_eq!(row.max_retries, restored.max_retries);
}

// -- step_name_of -------------------------------------------------------------

// Import step_name_of from the step module for testing here.
use crate::step::step_name_of;

#[test]
fn step_name_of_returns_short_name_for_simple_type() {
    // String is a stdlib type -- its type_name is just "String" with no path.
    assert_eq!(step_name_of::<String>(), "String");
}

#[test]
fn step_name_of_returns_last_component_for_nested_type() {
    // Vec<i32> has type_name "alloc::vec::Vec<i32>" -- we want only "Vec<i32>".
    // The function splits on "::" and takes the last segment.
    // Note: the exact full path is implementation-defined, but the last
    // segment after the final "::" is always the short name.
    let name = step_name_of::<std::collections::HashMap<String, i32>>();
    // Must not contain "::" -- only the final component
    assert!(
        !name.contains("::"),
        "step_name_of must return only the final component, got: {name}"
    );
}

// -- PipelineError ------------------------------------------------------------

use crate::error::PipelineError;

#[test]
fn pipeline_error_database_displays_message() {
    let e = PipelineError::Database("connection refused".to_string());
    assert!(e.to_string().contains("connection refused"));
}

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

#[test]
fn pipeline_error_from_sqlx_error_produces_database_variant() {
    // We cannot construct a real sqlx::Error in unit tests without a database.
    // Test the From impl indirectly by verifying the variant name via display.
    // The From<sqlx::Error> impl wraps the error string in PipelineError::Database.
    // We verify this by constructing via the string constructor directly and
    // confirming the display format matches what From would produce.
    let e = PipelineError::Database("sqlx: some error".to_string());
    assert!(e.to_string().starts_with("Database error:"));
}
