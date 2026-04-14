//! colossus-pipeline/src/worker/fetcher_recovery.rs
//!
//! Recovery and configuration SQL transitions for pipeline_jobs.
//!
//! This module contains the zombie recovery, timeout recovery, and
//! per-step configuration resolution functions. These are separated
//! from the main fetcher module (which handles job lifecycle transitions)
//! to keep both modules under the 300-line limit.
//!
//! ## Rust Learning: module splitting for maintainability
//!
//! Rust modules are the primary unit of code organization. When a module
//! grows past a readable size, splitting it into two modules with clear
//! responsibilities is the standard approach. Each module gets its own
//! file, its own //! doc comment, and its own test module. The pub(crate)
//! or pub visibility on each function controls who can call it.

use sqlx::PgPool;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::schema::JobStatus;

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
            status     = $1,
            wakeup_at  = NOW() + ($2 * INTERVAL '1 second'),
            updated_at = NOW()
        WHERE status = $3
          AND last_heartbeat_at < NOW() - ($4 * INTERVAL '1 second')
        RETURNING id, current_step, tried, max_retries
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(wakeup_secs)
    .bind(JobStatus::Running.as_str())
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
            status     = $1,
            wakeup_at  = NOW() + ($2 * INTERVAL '1 second'),
            updated_at = NOW()
        WHERE status = $3
          AND timeout_at IS NOT NULL
          AND timeout_at < NOW()
        RETURNING id, current_step, tried, max_retries
        "#,
    )
    .bind(JobStatus::Ready.as_str())
    .bind(wakeup_secs)
    .bind(JobStatus::Running.as_str())
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
    .unwrap_or_else(|e| {
        tracing::warn!(
            job_key = %job_key,
            step_name = %step_name,
            error = %e,
            "Failed to read step_config from pipeline_config — using compiled defaults"
        );
        None
    });

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
#[path = "fetcher_recovery_tests.rs"]
mod tests;
