//! recorder_tests.rs
//!
//! Tests for the StepRecorder trait and NoopStepRecorder.
//!
//! ## Rust Learning: testing trait implementations
//!
//! We test both the no-op implementation (to verify it satisfies the
//! trait contract) and the trait's Send + Sync bounds (compile-time
//! verification that Arc<dyn StepRecorder> works across threads).

use super::*;
use uuid::Uuid;

// ── NoopStepRecorder ─────────────────────────────────────────────

#[tokio::test]
async fn noop_recorder_start_returns_zero() {
    let recorder = NoopStepRecorder;
    let handle = recorder
        .on_step_start(Uuid::new_v4(), "doc-123", "ExtractText")
        .await
        .unwrap();
    assert_eq!(handle, 0);
}

#[tokio::test]
async fn noop_recorder_success_is_ok() {
    let recorder = NoopStepRecorder;
    let result = recorder
        .on_step_success(0, 1.5, &serde_json::json!({"entities": 10}))
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn noop_recorder_failure_is_ok() {
    let recorder = NoopStepRecorder;
    let result = recorder
        .on_step_failure(0, 0.5, "something went wrong")
        .await;
    assert!(result.is_ok());
}

// ── Trait object bounds ──────────────────────────────────────────

/// Compile-time assertion that `T: Send + Sync`.
/// If the bounds fail, this function fails to type-check.
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn step_recorder_is_object_safe_and_send_sync() {
    // Arc<dyn StepRecorder> must be Send + Sync for use in tokio::spawn.
    assert_send_sync::<std::sync::Arc<dyn StepRecorder>>();
}

#[test]
fn noop_recorder_is_send_sync() {
    assert_send_sync::<NoopStepRecorder>();
}
