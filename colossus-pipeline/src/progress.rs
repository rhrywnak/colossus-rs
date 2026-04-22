//! ProgressReporter — writes progress JSONB to pipeline_jobs during step execution.
//!
//! Steps call progress.report(json!({...})) to update the progress column,
//! which the frontend polls to display real-time status to the user.
//! Errors are intentionally swallowed — a progress write failure must never
//! interrupt step execution.

use sqlx::PgPool;
use uuid::Uuid;

/// Writes progress updates to pipeline_jobs.progress during step execution.
///
/// Constructed by the executor for each step execution. The step calls
/// report() between expensive operations (e.g., after each LLM chunk call).
///
/// Also carries an in-memory `step_result` slot that steps populate via
/// `set_step_result()` just before returning. The executor reads it via
/// `take_step_result()` after `execute_current` returns and threads the
/// value into `ExecutionResult::Success` / `ExecutionResult::Done` so the
/// `StepRecorder` callback receives the step's summary data.
pub struct ProgressReporter {
    db: PgPool,
    job_id: Uuid,

    /// In-memory slot for step result summary data.
    ///
    /// Steps call `set_step_result()` before returning to store summary
    /// data (e.g., entity counts, page counts). The executor reads this
    /// after `execute_current` returns and passes it to the StepRecorder
    /// and into `ExecutionResult::Success`.
    ///
    /// ## Rust Learning: `std::sync::Mutex` vs `tokio::sync::Mutex`
    ///
    /// We use `std::sync::Mutex` (not tokio's async Mutex) because the
    /// lock is held only for the duration of a single assignment or read —
    /// nanoseconds, never across an `.await` point. std::sync::Mutex is
    /// cheaper in this case. tokio::sync::Mutex is needed only when the
    /// lock must be held across await points.
    step_result: std::sync::Mutex<serde_json::Value>,
}

impl ProgressReporter {
    /// Create a new reporter for the given job.
    pub fn new(db: PgPool, job_id: Uuid) -> Self {
        Self {
            db,
            job_id,
            step_result: std::sync::Mutex::new(serde_json::Value::Null),
        }
    }

    /// Write a progress snapshot to pipeline_jobs.progress.
    ///
    /// Errors are swallowed — a failed progress write must never fail the step.
    /// The value is arbitrary JSON; the frontend interprets the shape.
    pub async fn report(&self, value: serde_json::Value) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE pipeline_jobs SET progress = $1, updated_at = NOW() WHERE id = $2",
        )
        .bind(&value)
        .bind(self.job_id)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    /// Store step result summary data in memory.
    ///
    /// Called by step implementations just before returning `Ok(StepResult::...)`.
    /// The data is NOT written to the database — it stays in memory and is
    /// read by the executor after `execute_current` returns.
    ///
    /// Example usage in a step:
    /// ```ignore
    /// progress.set_step_result(serde_json::json!({
    ///     "entity_count": 41,
    ///     "relationship_count": 40,
    /// }));
    /// Ok(StepResult::Next(DocProcessing::Verify(Verify { doc_id })))
    /// ```
    pub fn set_step_result(&self, value: serde_json::Value) {
        if let Ok(mut guard) = self.step_result.lock() {
            *guard = value;
        }
    }

    /// Take the stored step result, replacing it with Null.
    ///
    /// Called by the executor after `execute_current` returns. Returns
    /// whatever the step stored via `set_step_result`, or `Value::Null`
    /// if nothing was stored. The slot is reset to `Null` after taking.
    ///
    /// ## Rust Learning: `std::mem::replace`
    ///
    /// `std::mem::replace(&mut *guard, Value::Null)` swaps the value in
    /// the mutex guard with `Null` and returns the old value — all in one
    /// operation without cloning. This is the idiomatic Rust "take" pattern.
    pub fn take_step_result(&self) -> serde_json::Value {
        self.step_result
            .lock()
            .map(|mut guard| std::mem::replace(&mut *guard, serde_json::Value::Null))
            .unwrap_or(serde_json::Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ## Rust Learning: testing with `connect_lazy`
    ///
    /// `PgPool::connect_lazy` creates a pool that doesn't actually connect
    /// until the first query. For tests that only exercise in-memory behavior
    /// (like set/take_step_result), this avoids needing a live database.
    fn lazy_pool() -> PgPool {
        PgPool::connect_lazy("postgres://invalid:5432/none")
            .expect("connect_lazy should not fail")
    }

    // Tests use #[tokio::test] rather than #[test] because sqlx::PgPool::connect_lazy
    // requires a tokio runtime handle to install the pool's internal maintenance tasks,
    // even though no connection is actually opened. This matches the convention in
    // colossus-pipeline/src/worker/mod_tests.rs.

    #[tokio::test]
    async fn take_step_result_returns_null_when_nothing_set() {
        let reporter = ProgressReporter::new(lazy_pool(), Uuid::new_v4());
        let result = reporter.take_step_result();
        assert_eq!(result, serde_json::Value::Null);
    }

    #[tokio::test]
    async fn set_then_take_returns_stored_value() {
        let reporter = ProgressReporter::new(lazy_pool(), Uuid::new_v4());
        let summary = serde_json::json!({"entity_count": 42});
        reporter.set_step_result(summary.clone());
        let result = reporter.take_step_result();
        assert_eq!(result, summary);
    }

    #[tokio::test]
    async fn take_resets_to_null() {
        let reporter = ProgressReporter::new(lazy_pool(), Uuid::new_v4());
        reporter.set_step_result(serde_json::json!({"x": 1}));
        let _ = reporter.take_step_result(); // consume it
        let second = reporter.take_step_result();
        assert_eq!(second, serde_json::Value::Null, "second take should be Null");
    }

    #[tokio::test]
    async fn set_overwrites_previous() {
        let reporter = ProgressReporter::new(lazy_pool(), Uuid::new_v4());
        reporter.set_step_result(serde_json::json!({"first": true}));
        reporter.set_step_result(serde_json::json!({"second": true}));
        let result = reporter.take_step_result();
        assert_eq!(result, serde_json::json!({"second": true}));
    }
}
