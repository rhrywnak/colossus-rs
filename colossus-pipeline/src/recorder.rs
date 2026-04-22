//! recorder.rs
//!
//! The StepRecorder trait for application-specific step lifecycle recording.
//!
//! The pipeline framework calls these methods automatically around each
//! step execution in the Worker's spawned task. Applications implement
//! this trait to record step history in their own tables (e.g.,
//! colossus-legal's `pipeline_steps` table). The framework never touches
//! that table directly — that is a deliberate boundary, because
//! colossus-pipeline has zero knowledge of application-specific schemas.
//!
//! ## Rust Learning: trait objects with `Arc<dyn Trait>`
//!
//! StepRecorder is stored as `Arc<dyn StepRecorder>` on the Worker.
//! `Arc` enables shared ownership across spawned tokio tasks. `dyn` enables
//! runtime polymorphism — the framework doesn't know the concrete type.
//! The Send + Sync + 'static bounds are required because Arc<dyn StepRecorder>
//! is shared across threads (Send + Sync) and lives beyond any single
//! function scope ('static).
//!
//! ## Rust Learning: opaque handles with `i64`
//!
//! `on_step_start` returns an `i64` that the framework passes back to
//! `on_step_success` / `on_step_failure`. The framework treats it as opaque —
//! it doesn't know what it represents. In colossus-legal, it's the
//! `pipeline_steps.id` primary key. This pattern avoids the framework
//! depending on application-specific types.

use async_trait::async_trait;
use uuid::Uuid;

/// Records step lifecycle events for observability.
///
/// The framework guarantees:
/// - `on_step_start` is called exactly once before `execute_current`
/// - Exactly one of `on_step_success` or `on_step_failure` is called after
/// - Recording errors are logged but never prevent step execution or result handling
///
/// Applications that don't need step recording can use `NoopStepRecorder`.
#[async_trait]
pub trait StepRecorder: Send + Sync + 'static {
    /// Called before step execution begins.
    ///
    /// # Parameters
    /// - `job_id` — the pipeline job UUID
    /// - `job_key` — application-defined key (e.g., document_id in colossus-legal)
    /// - `step_name` — the current step name (e.g., "ExtractText", "Verify")
    ///
    /// # Returns
    /// An opaque handle (e.g., a database row ID) passed to success/failure callbacks.
    async fn on_step_start(
        &self,
        job_id: Uuid,
        job_key: &str,
        step_name: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>>;

    /// Called when a step completes successfully.
    ///
    /// # Parameters
    /// - `step_handle` — the value returned by `on_step_start`
    /// - `duration_secs` — wall-clock time the step took
    /// - `result_summary` — the accumulated result JSONB from the step
    async fn on_step_success(
        &self,
        step_handle: i64,
        duration_secs: f64,
        result_summary: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Called when a step fails, times out, or is cancelled.
    ///
    /// # Parameters
    /// - `step_handle` — the value returned by `on_step_start`
    /// - `duration_secs` — wall-clock time before the failure
    /// - `error_message` — human-readable error description
    async fn on_step_failure(
        &self,
        step_handle: i64,
        duration_secs: f64,
        error_message: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// No-op recorder for applications that don't need step recording.
///
/// Returns 0 as the step handle and succeeds on all callbacks.
/// This is the default when no recorder is configured.
///
/// ## Rust Learning: unit struct
///
/// `NoopStepRecorder` has no fields — it's a zero-size type (ZST).
/// It exists only to carry the trait implementation. ZSTs occupy no
/// memory at runtime, making this truly zero-cost.
pub struct NoopStepRecorder;

#[async_trait]
impl StepRecorder for NoopStepRecorder {
    async fn on_step_start(
        &self,
        _job_id: Uuid,
        _job_key: &str,
        _step_name: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        Ok(0)
    }

    async fn on_step_success(
        &self,
        _step_handle: i64,
        _duration_secs: f64,
        _result_summary: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    async fn on_step_failure(
        &self,
        _step_handle: i64,
        _duration_secs: f64,
        _error_message: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "recorder_tests.rs"]
mod tests;
