//! Tests for EVT_* constants and events::log().
//!
//! ## Rust Learning: testing without a database
//!
//! log() requires a live PgPool to INSERT into pipeline_events.
//! We cannot test the database interaction in unit tests without a real
//! PostgreSQL instance. Instead we test the things we can verify without
//! a database: constant values, message truncation logic extracted into
//! a pure function, and that the constants are distinct from each other.
//! Database integration tests belong in Phase 6 end-to-end validation.

use super::*;

#[test]
fn evt_job_submitted_value_is_correct() {
    assert_eq!(EVT_JOB_SUBMITTED, "job_submitted");
}

#[test]
fn evt_step_started_value_is_correct() {
    assert_eq!(EVT_STEP_STARTED, "step_started");
}

#[test]
fn evt_step_completed_value_is_correct() {
    assert_eq!(EVT_STEP_COMPLETED, "step_completed");
}

#[test]
fn evt_step_failed_value_is_correct() {
    assert_eq!(EVT_STEP_FAILED, "step_failed");
}

#[test]
fn evt_cancelled_value_is_correct() {
    assert_eq!(EVT_CANCELLED, "cancelled");
}

#[test]
fn evt_cancel_requested_value_is_correct() {
    assert_eq!(EVT_CANCEL_REQUESTED, "cancel_requested");
}

#[test]
fn evt_job_completed_value_is_correct() {
    assert_eq!(EVT_JOB_COMPLETED, "job_completed");
}

#[test]
fn evt_job_deleted_value_is_correct() {
    assert_eq!(EVT_JOB_DELETED, "job_deleted");
}

#[test]
fn all_evt_constants_are_distinct() {
    let constants = [
        EVT_JOB_SUBMITTED,
        EVT_STEP_STARTED,
        EVT_STEP_COMPLETED,
        EVT_STEP_FAILED,
        EVT_STEP_EXHAUSTED,
        EVT_RETRY_SCHEDULED,
        EVT_ZOMBIE_RECOVERED,
        EVT_TIMEOUT_RECOVERED,
        EVT_CANCELLED,
        EVT_CANCEL_REQUESTED,
        EVT_WAITING_INPUT,
        EVT_ADVANCED,
        EVT_RESUMED,
        EVT_JOB_COMPLETED,
        EVT_JOB_DELETED,
    ];
    let mut seen = std::collections::HashSet::new();
    for c in &constants {
        assert!(seen.insert(*c), "Duplicate EVT_ constant value: {c}");
    }
    assert_eq!(constants.len(), 15, "Expected exactly 15 EVT_ constants");
}

#[test]
fn message_longer_than_500_chars_is_truncated() {
    // Verify the truncation boundary used in log().
    // We extract the truncation logic into a testable helper rather than
    // testing it only through a live database call.
    let long_message = "x".repeat(600);
    let truncated = truncate_message(&long_message);
    assert_eq!(truncated.len(), 500);
}

#[test]
fn message_exactly_500_chars_is_not_truncated() {
    let message = "x".repeat(500);
    let truncated = truncate_message(&message);
    assert_eq!(truncated.len(), 500);
}

#[test]
fn message_shorter_than_500_chars_is_unchanged() {
    let message = "short message".to_string();
    let truncated = truncate_message(&message);
    assert_eq!(truncated, message);
}
