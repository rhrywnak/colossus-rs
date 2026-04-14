//! colossus-pipeline/src/step_tests.rs
//!
//! Tests for Step trait, Task trait, StepResult, and step_name_of.
//!
//! ## Rust Learning: testing traits with mock implementations
//!
//! Traits cannot be instantiated directly — they require a concrete type
//! that implements them. To test trait behavior we create minimal mock
//! types that implement the trait with the simplest possible behavior.
//! This is standard practice in Rust testing. The mock types live only
//! in the test module and are never compiled into production code.
//!
//! ## Rust Learning: #[allow(dead_code)] in tests
//!
//! Mock types defined in tests may have fields that are never read.
//! #[allow(dead_code)] suppresses the warning for test-only types.

use super::*;

// ── step_name_of ─────────────────────────────────────────────────────────────

#[test]
fn step_name_of_string_returns_string() {
    assert_eq!(step_name_of::<String>(), "String");
}

#[test]
fn step_name_of_u32_returns_u32() {
    assert_eq!(step_name_of::<u32>(), "u32");
}

#[test]
fn step_name_of_does_not_contain_colons() {
    // Any type in a nested module must return only the final component.
    // std::collections::HashMap has a full path — we want only the last part.
    let name = step_name_of::<std::collections::HashMap<String, u32>>();
    // The name after the last "::" must not itself contain "::"
    assert!(
        !name.contains("::"),
        "step_name_of must return only the final component, got: {name}",
    );
}

#[test]
fn step_name_of_vec_returns_short_name() {
    let name = step_name_of::<Vec<u8>>();
    assert!(!name.contains("::"), "Got: {name}");
}

// ── StepResult is Send when T is Send ────────────────────────────────────────

// This test verifies a compile-time property: if StepResult<T> is not Send
// when T is Send, this test will fail to compile rather than fail at runtime.
// ## Rust Learning: compile-time trait bound verification
// fn assert_send<T: Send>() {} is a zero-cost function that only compiles
// if T: Send. Calling it with a type verifies the bound at compile time.
fn assert_send<T: Send>() {}

#[test]
fn step_result_is_send_when_task_is_send() {
    // MockTask is Send (it contains only Send types).
    // StepResult<MockTask> must therefore also be Send.
    assert_send::<StepResult<MockTask>>();
}

// ── Mock types for trait testing ─────────────────────────────────────────────

/// Minimal Task implementation for testing Step and StepResult behavior.
/// Contains only what is needed to satisfy the trait bounds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
enum MockTask {
    /// First step in the mock pipeline.
    StepA,
    /// Second step in the mock pipeline.
    StepB,
}

#[async_trait::async_trait]
impl crate::task::Task for MockTask {
    type Context = ();

    fn current_step_name(&self) -> &'static str {
        match self {
            MockTask::StepA => "StepA",
            MockTask::StepB => "StepB",
        }
    }

    fn validate_transition(current: &str, next: &str) -> Result<(), crate::error::PipelineError> {
        match (current, next) {
            ("StepA", "StepB") => Ok(()),
            _ => Err(crate::error::PipelineError::InvalidTransition {
                from: current.to_string(),
                to: next.to_string(),
            }),
        }
    }

    async fn execute_current(
        self,
        _db: &sqlx::PgPool,
        _context: &Self::Context,
        _cancel: &crate::cancel::CancellationToken,
        _progress: &crate::progress::ProgressReporter,
    ) -> Result<StepResult<Self>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(StepResult::Done)
    }

    async fn on_cancel_current(
        self,
        _db: &sqlx::PgPool,
        _context: &Self::Context,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    async fn on_delete_current(
        self,
        _db: &sqlx::PgPool,
        _context: &Self::Context,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

// ── Task trait behavior ───────────────────────────────────────────────────────

#[test]
fn mock_task_step_a_name_is_correct() {
    assert_eq!(MockTask::StepA.current_step_name(), "StepA");
}

#[test]
fn mock_task_step_b_name_is_correct() {
    assert_eq!(MockTask::StepB.current_step_name(), "StepB");
}

#[test]
fn validate_transition_step_a_to_step_b_is_ok() {
    assert!(MockTask::validate_transition("StepA", "StepB").is_ok());
}

#[test]
fn validate_transition_step_a_to_step_a_is_invalid() {
    let result = MockTask::validate_transition("StepA", "StepA");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        crate::error::PipelineError::InvalidTransition { .. }
    ));
}

#[test]
fn validate_transition_step_b_to_step_a_is_invalid() {
    let result = MockTask::validate_transition("StepB", "StepA");
    assert!(result.is_err());
}
