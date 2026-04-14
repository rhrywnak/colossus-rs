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
pub struct ProgressReporter {
    db: PgPool,
    job_id: Uuid,
}

impl ProgressReporter {
    /// Create a new reporter for the given job.
    pub fn new(db: PgPool, job_id: Uuid) -> Self {
        Self { db, job_id }
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
}
