//! worker/fetcher_api.rs
//!
//! Scheduler-facing SQL transitions for pipeline_jobs.
//!
//! These functions are called by `Scheduler` methods (cancel, resume,
//! advance_from_park). They are separated from the main fetcher module
//! (which handles Worker-facing transitions like claim, advance, fail)
//! to keep both modules under the 300-line limit.

use sqlx::PgPool;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::schema::{JobControl, JobStatus};

/// Request cancellation of a running job.
///
/// Sets control=cancel_requested. The cancel_watcher inside the executor
/// polls this column and triggers the CancellationToken when it sees this value.
/// Returns true if the job was in a cancellable state, false if not found
/// or already in a terminal state.
pub async fn request_cancel(
    db: &PgPool,
    job_id: Uuid,
) -> Result<bool, PipelineError> {
    let result = sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            control    = $1,
            updated_at = NOW()
        WHERE id = $2
          AND status NOT IN ($3, $4, $5)
        "#,
    )
    .bind(JobControl::CancelRequested.as_str())
    .bind(job_id)
    .bind(JobStatus::Completed.as_str())
    .bind(JobStatus::Failed.as_str())
    .bind(JobStatus::Cancelled.as_str())
    .execute(db)
    .await?;

    Ok(result.rows_affected() > 0)
}

/// Resume a parked or failed job.
///
/// Clears the control signal and resets status to ready so the worker
/// can re-claim it. Used after human review completes or after manual
/// retry of a failed job.
pub async fn resume(
    db: &PgPool,
    job_id: Uuid,
) -> Result<(), PipelineError> {
    let result = sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status     = $1,
            control    = $2,
            error      = NULL,
            wakeup_at  = NOW(),
            updated_at = NOW()
        WHERE id = $3
          AND (
              (status = $4)
              OR (status = $5 AND control = $6)
          )
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(JobControl::None.as_str())
    .bind(job_id)
    .bind(JobStatus::Failed.as_str())
    .bind(JobStatus::Ready.as_str())
    .bind(JobControl::WaitingForInput.as_str())
    .execute(db)
    .await?;

    if result.rows_affected() == 0 {
        return Err(PipelineError::JobNotResumable(job_id));
    }

    Ok(())
}

/// Advance a parked job to its next step after external input is provided.
///
/// Used when a WaitForInput step has received the input it was waiting for.
/// Clears waiting_for_input control, sets next step data.
pub async fn advance_from_park(
    db: &PgPool,
    job_id: Uuid,
    next_step_data: &serde_json::Value,
    next_step_name: &str,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status       = $1,
            control      = $2,
            current_step = $3,
            step_data    = $4,
            wakeup_at    = NOW(),
            updated_at   = NOW()
        WHERE id = $5
          AND status = $6
          AND control = $7
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(JobControl::None.as_str())
    .bind(next_step_name)
    .bind(next_step_data)
    .bind(job_id)
    .bind(JobStatus::Ready.as_str())
    .bind(JobControl::WaitingForInput.as_str())
    .execute(db)
    .await?;

    Ok(())
}
