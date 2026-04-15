//! colossus-pipeline/src/worker/fetcher.rs
//!
//! All SQL state transitions for pipeline_jobs.
//!
//! Each function is one atomic FSM transition. All transitions use
//! parameterized queries — no string interpolation. Status and control
//! values are bound as &str matching the sqlx::Type serialization of
//! JobStatus and JobControl — never constructed from format!().
//!
//! FOR UPDATE SKIP LOCKED on claim() is the PostgreSQL mechanism that
//! prevents two workers from claiming the same job. When worker A holds
//! a lock on a row, worker B's SKIP LOCKED silently skips that row and
//! claims the next available one instead of blocking.
//!
//! ## Rust Learning: why all transitions are separate functions
//!
//! Each transition touches different columns and enforces different
//! preconditions. Combining them into a generic "update status" function
//! would lose the FSM enforcement — any caller could set any status.
//! Separate functions make invalid transitions impossible to call by accident.
//!
//! ## Rust Learning: sqlx::query_as vs sqlx::query
//!
//! sqlx::query_as::<_, JobRow>(sql) maps the result rows into JobRow using
//! the sqlx::FromRow derive. sqlx::query(sql) returns a raw result with no
//! mapping — used for UPDATE/INSERT where we only care about rows_affected.

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::schema::{JobControl, JobRow, JobStatus};

/// Error message stored when a job is cancelled by user request.
/// Appears in pipeline_jobs.error and pipeline_events for cancelled jobs.
pub(crate) const CANCELLED_BY_USER: &str = "Cancelled by user";

/// Attempt to claim one ready job from the queue.
///
/// Uses SELECT ... FOR UPDATE SKIP LOCKED with priority DESC, wakeup_at ASC
/// ordering so high-priority jobs and overdue jobs are claimed first.
/// Sets all execution tracking fields atomically in a single UPDATE.
/// Returns None if the queue is empty or all ready jobs are locked by
/// other workers.
///
/// ## Rust Learning: FOR UPDATE SKIP LOCKED
///
/// This PostgreSQL clause is the standard pattern for job queues.
/// SELECT finds the best candidate row and locks it. SKIP LOCKED means
/// concurrent workers skip already-locked rows instead of waiting.
/// The lock is released when the transaction commits (the UPDATE).
/// Without this, two workers could claim the same job simultaneously.
pub async fn claim(
    db: &PgPool,
    worker_id: &str,
    timeout_secs: Option<u64>,
) -> Result<Option<JobRow>, PipelineError> {
    let timeout_at: Option<DateTime<Utc>> = timeout_secs.map(|secs| {
        Utc::now() + chrono::Duration::seconds(secs as i64)
    });

    let row = sqlx::query_as::<_, JobRow>(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            worker_id         = $2,
            step_started_at   = NOW(),
            step_completed_at = NULL,
            last_heartbeat_at = NOW(),
            timeout_at        = COALESCE($3, timeout_at),
            updated_at        = NOW()
        WHERE id = (
            SELECT id FROM pipeline_jobs
            WHERE status = $4
              AND control = $5
              AND wakeup_at <= NOW()
            ORDER BY priority DESC, wakeup_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *
        "#,
    )
    .bind(JobStatus::Running.as_str())
    .bind(worker_id)
    .bind(timeout_at)
    .bind(JobStatus::Ready.as_str())
    .bind(JobControl::None.as_str())
    .fetch_optional(db)
    .await?;

    Ok(row)
}

/// Advance a job to its next step after successful step completion.
///
/// Serializes the next Task variant into step_data, resets tried to 0,
/// clears error (T2 fix: error=NULL on advance so previous attempt errors
/// do not persist into the next step), and sets wakeup_at to now so the
/// job is immediately re-claimable.
///
/// ## Rust Learning: serde_json::to_value for JSONB binding
///
/// sqlx cannot bind arbitrary Rust types to JSONB directly. We serialize
/// to serde_json::Value first, then bind the Value. This works because
/// sqlx knows how to encode serde_json::Value as PostgreSQL JSONB.
pub async fn advance(
    db: &PgPool,
    job_id: Uuid,
    next_step_data: &serde_json::Value,
    next_step_name: &str,
    step_result: &serde_json::Value,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            current_step      = $2,
            step_data         = $3,
            result            = result || $4,
            tried             = 0,
            error             = NULL,
            step_completed_at = NOW(),
            wakeup_at         = NOW(),
            updated_at        = NOW()
        WHERE id = $5
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(next_step_name)
    .bind(next_step_data)
    .bind(step_result)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Advance a job to its next step with a delayed wakeup.
///
/// Identical to [`advance`] except `wakeup_at` is set to `NOW() + delay_secs`
/// instead of `NOW()`. Used by `StepResult::Delay` when a step requests a
/// backoff before re-execution (e.g., rate-limit retry detected by the step).
pub async fn advance_with_delay(
    db: &PgPool,
    job_id: Uuid,
    next_step_data: &serde_json::Value,
    next_step_name: &str,
    step_result: &serde_json::Value,
    delay_secs: i64,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            current_step      = $2,
            step_data         = $3,
            result            = result || $4,
            tried             = 0,
            error             = NULL,
            step_completed_at = NOW(),
            wakeup_at         = NOW() + make_interval(secs => $5),
            updated_at        = NOW()
        WHERE id = $6
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(next_step_name)
    .bind(next_step_data)
    .bind(step_result)
    .bind(delay_secs as f64)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Mark a job as fully completed.
///
/// Terminal state — no further transitions except deletion.
/// Merges the final step result into the accumulated result JSONB.
pub async fn complete(
    db: &PgPool,
    job_id: Uuid,
    step_result: &serde_json::Value,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            result            = result || $2,
            error             = NULL,
            step_completed_at = NOW(),
            completed_at      = NOW(),
            updated_at        = NOW()
        WHERE id = $3
        "#,
    )
    .bind(JobStatus::Completed.as_str())
    .bind(step_result)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Park a job waiting for external input (e.g. human review gate).
///
/// Sets control=waiting_for_input. The worker will not re-claim this job
/// until resume() is called. The current step data is preserved so
/// execution can continue from where it paused.
pub async fn park(
    db: &PgPool,
    job_id: Uuid,
    parked_step_data: &serde_json::Value,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status     = $1,
            control    = $2,
            step_data  = $3,
            updated_at = NOW()
        WHERE id = $4
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(JobControl::WaitingForInput.as_str())
    .bind(parked_step_data)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Mark a job as cancelled after the cancellation was processed.
///
/// Called by the executor after on_cancel_current() completes.
/// Sets status=failed with error="Cancelled by user" — cancelled jobs
/// use the failed status so they appear in the retry/resume flow if needed.
pub async fn cancel(
    db: &PgPool,
    job_id: Uuid,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            control           = $2,
            error             = $3,
            step_completed_at = NOW(),
            updated_at        = NOW()
        WHERE id = $4
        "#,
    )
    .bind(JobStatus::Failed.as_str())
    .bind(JobControl::None.as_str())
    .bind(CANCELLED_BY_USER)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Schedule a retry after a recoverable step failure.
///
/// Increments tried, sets wakeup_at in the future, resets status to ready
/// so the job is re-claimable after the delay. The error message is stored
/// for observability.
pub async fn fail_with_retry(
    db: &PgPool,
    job_id: Uuid,
    error: &str,
    delay_secs: i64,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            tried             = tried + 1,
            error             = $2,
            wakeup_at         = NOW() + ($3 * INTERVAL '1 second'),
            step_completed_at = NOW(),
            updated_at        = NOW()
        WHERE id = $4
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(error)
    .bind(delay_secs)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Mark a job as permanently failed after retries are exhausted.
///
/// Terminal state — no further automatic transitions.
/// User can call resume() to manually retry from this state.
pub async fn fail_exhausted(
    db: &PgPool,
    job_id: Uuid,
    error: &str,
) -> Result<(), PipelineError> {
    sqlx::query(
        r#"
        UPDATE pipeline_jobs
        SET
            status            = $1,
            error             = $2,
            step_completed_at = NOW(),
            updated_at        = NOW()
        WHERE id = $3
        "#,
    )
    .bind(JobStatus::Failed.as_str())
    .bind(error)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

#[cfg(test)]
#[path = "fetcher_tests.rs"]
mod tests;
