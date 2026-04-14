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
use crate::schema::JobRow;

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
            status            = 'running',
            worker_id         = $1,
            step_started_at   = NOW(),
            step_completed_at = NULL,
            last_heartbeat_at = NOW(),
            timeout_at        = $2,
            updated_at        = NOW()
        WHERE id = (
            SELECT id FROM pipeline_jobs
            WHERE status = 'ready'
              AND control = 'none'
              AND wakeup_at <= NOW()
            ORDER BY priority DESC, wakeup_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *
        "#,
    )
    .bind(worker_id)
    .bind(timeout_at)
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
            status            = 'ready',
            current_step      = $1,
            step_data         = $2,
            result            = result || $3,
            tried             = 0,
            error             = NULL,
            step_completed_at = NOW(),
            wakeup_at         = NOW(),
            updated_at        = NOW()
        WHERE id = $4
        "#,
    )
    .bind(next_step_name)
    .bind(next_step_data)
    .bind(step_result)
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
            status            = 'completed',
            result            = result || $1,
            error             = NULL,
            step_completed_at = NOW(),
            completed_at      = NOW(),
            updated_at        = NOW()
        WHERE id = $2
        "#,
    )
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
            status     = 'ready',
            control    = 'waiting_for_input',
            step_data  = $1,
            updated_at = NOW()
        WHERE id = $2
        "#,
    )
    .bind(parked_step_data)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

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
            control    = 'cancel_requested',
            updated_at = NOW()
        WHERE id = $1
          AND status NOT IN ('completed', 'failed', 'cancelled')
        "#,
    )
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(result.rows_affected() > 0)
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
            status            = 'failed',
            control           = 'none',
            error             = 'Cancelled by user',
            step_completed_at = NOW(),
            updated_at        = NOW()
        WHERE id = $1
        "#,
    )
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
            status            = 'ready',
            tried             = tried + 1,
            error             = $1,
            wakeup_at         = NOW() + ($2 * INTERVAL '1 second'),
            step_completed_at = NOW(),
            updated_at        = NOW()
        WHERE id = $3
        "#,
    )
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
            status            = 'failed',
            error             = $1,
            step_completed_at = NOW(),
            updated_at        = NOW()
        WHERE id = $2
        "#,
    )
    .bind(error)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
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
            status     = 'ready',
            control    = 'none',
            error      = NULL,
            wakeup_at  = NOW(),
            updated_at = NOW()
        WHERE id = $1
          AND (
              (status = 'failed')
              OR (status = 'ready' AND control = 'waiting_for_input')
          )
        "#,
    )
    .bind(job_id)
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
            status       = 'ready',
            control      = 'none',
            current_step = $1,
            step_data    = $2,
            wakeup_at    = NOW(),
            updated_at   = NOW()
        WHERE id = $3
          AND status = 'ready'
          AND control = 'waiting_for_input'
        "#,
    )
    .bind(next_step_name)
    .bind(next_step_data)
    .bind(job_id)
    .execute(db)
    .await?;

    Ok(())
}

/// Find zombie jobs — running jobs whose heartbeat has gone stale.
///
/// A zombie is a job stuck in 'running' state because its worker crashed
/// mid-step. Detected by last_heartbeat_at being older than threshold_secs.
/// Returns (job_id, current_step, tried, max_retries) for each zombie
/// so the recovery manager can decide whether to retry or fail each one.
///
/// ## Rust Learning: returning tuples from sqlx queries
///
/// sqlx::query_as::<_, (Uuid, String, i32, i32)> maps each result row
/// to a tuple. The types must match the SELECT column order exactly.
/// Tuples are useful for lightweight query results that don't warrant
/// a dedicated struct.
pub async fn recover_zombie(
    db: &PgPool,
    wakeup_secs: i64,
    threshold_secs: i64,
) -> Result<Vec<(Uuid, String, i32, i32)>, PipelineError> {
    let rows = sqlx::query_as::<_, (Uuid, String, i32, i32)>(
        r#"
        UPDATE pipeline_jobs
        SET
            status     = 'ready',
            wakeup_at  = NOW() + ($1 * INTERVAL '1 second'),
            updated_at = NOW()
        WHERE status = 'running'
          AND last_heartbeat_at < NOW() - ($2 * INTERVAL '1 second')
        RETURNING id, current_step, tried, max_retries
        "#,
    )
    .bind(wakeup_secs)
    .bind(threshold_secs)
    .fetch_all(db)
    .await?;

    Ok(rows)
}

/// Find timed-out jobs — running jobs whose timeout_at is in the past.
///
/// Returns (job_id, current_step, tried, max_retries) for each timed-out
/// job so the recovery manager can decide whether to retry or fail each one.
pub async fn recover_timeout(
    db: &PgPool,
    wakeup_secs: i64,
) -> Result<Vec<(Uuid, String, i32, i32)>, PipelineError> {
    let rows = sqlx::query_as::<_, (Uuid, String, i32, i32)>(
        r#"
        UPDATE pipeline_jobs
        SET
            status     = 'ready',
            wakeup_at  = NOW() + ($1 * INTERVAL '1 second'),
            updated_at = NOW()
        WHERE status = 'running'
          AND timeout_at IS NOT NULL
          AND timeout_at < NOW()
        RETURNING id, current_step, tried, max_retries
        "#,
    )
    .bind(wakeup_secs)
    .fetch_all(db)
    .await?;

    Ok(rows)
}

/// Resolved configuration for a specific step from pipeline_config.step_config JSONB.
///
/// The Worker reads this to determine retry limits and timeouts per step,
/// allowing per-document tuning without recompile.
#[derive(Debug, Clone)]
pub struct ResolvedStepConfig {
    /// Maximum retry attempts for this step. 0 means fail immediately.
    pub retry_limit: i32,
    /// Seconds to wait between retries.
    pub retry_delay_secs: i32,
    /// Step timeout in seconds. None means no timeout.
    pub timeout_secs: Option<u64>,
}

/// Read step configuration from pipeline_config.step_config JSONB.
///
/// Falls back to compiled-in defaults when no pipeline_config row exists
/// or when the step_config JSONB does not contain an entry for this step.
/// This ensures steps always have valid config even without database setup.
pub async fn resolve_step_config(
    db: &PgPool,
    job_key: &str,
    step_name: &str,
    default_retry_limit: i32,
    default_retry_delay_secs: i32,
    default_timeout_secs: Option<u64>,
) -> ResolvedStepConfig {
    let result: Option<serde_json::Value> = sqlx::query_scalar(
        r#"
        SELECT step_config -> $1
        FROM pipeline_config
        WHERE document_id = $2
        LIMIT 1
        "#,
    )
    .bind(step_name)
    .bind(job_key)
    .fetch_optional(db)
    .await
    .ok()
    .flatten();

    match result {
        Some(cfg) if !cfg.is_null() => ResolvedStepConfig {
            retry_limit: cfg.get("retry_limit")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(default_retry_limit),
            retry_delay_secs: cfg.get("retry_delay_secs")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(default_retry_delay_secs),
            timeout_secs: cfg.get("timeout_secs")
                .and_then(|v| v.as_u64())
                .or(default_timeout_secs),
        },
        _ => ResolvedStepConfig {
            retry_limit: default_retry_limit,
            retry_delay_secs: default_retry_delay_secs,
            timeout_secs: default_timeout_secs,
        },
    }
}

#[cfg(test)]
#[path = "fetcher_tests.rs"]
mod tests;
