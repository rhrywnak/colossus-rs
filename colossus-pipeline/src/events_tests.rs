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

/// Wire-format pin for `pipeline_events.event_type` — these strings are
/// consumed by external dashboards/log queries with no compile-time check
/// (per Batch 10 ruling). Includes all 15 EVT_ constants per
/// FOLLOWUP-extend-evt-coverage; matches the list in
/// `all_evt_constants_are_distinct` below.
#[test]
fn evt_constants_have_expected_values() {
    let cases: &[(&str, &str)] = &[
        (EVT_JOB_SUBMITTED, "job_submitted"),
        (EVT_STEP_STARTED, "step_started"),
        (EVT_STEP_COMPLETED, "step_completed"),
        (EVT_STEP_FAILED, "step_failed"),
        (EVT_STEP_EXHAUSTED, "step_exhausted"),
        (EVT_RETRY_SCHEDULED, "retry_scheduled"),
        (EVT_ZOMBIE_RECOVERED, "zombie_recovered"),
        (EVT_TIMEOUT_RECOVERED, "timeout_recovered"),
        (EVT_CANCELLED, "cancelled"),
        (EVT_CANCEL_REQUESTED, "cancel_requested"),
        (EVT_WAITING_INPUT, "waiting_for_input"),
        (EVT_ADVANCED, "advanced"),
        (EVT_RESUMED, "resumed"),
        (EVT_JOB_COMPLETED, "job_completed"),
        (EVT_JOB_DELETED, "job_deleted"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, expected, "EVT_ constant mismatch");
    }
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
