//! worker/recovery.rs
//!
//! Zombie and timeout recovery — detects and resets stuck jobs.
//!
//! Runs on worker startup and every `PIPELINE_RECOVERY_INTERVAL_SECS`.
//! Zombie: a Running job whose `last_heartbeat_at` is older than
//! `PIPELINE_ZOMBIE_THRESHOLD_SECS`. This indicates the worker crashed
//! mid-step. Recovery resets the job to Ready for retry.
//! Timeout: a Running job whose `timeout_at` is in the past.
//!
//! The SQL transitions (reset to Ready) are handled by `recover_zombie`
//! and `recover_timeout` in `fetcher_recovery.rs`. This module handles
//! the exhausted case (tried >= max_retries → fail) and event logging.

use sqlx::PgPool;

use crate::error::PipelineError;
use crate::events;
use crate::worker::config::WorkerConfig;
use crate::worker::fetcher::fail_exhausted;
use crate::worker::fetcher_recovery::{recover_timeout, recover_zombie};

/// Scan for zombie and timed-out jobs, reset them for retry or fail them
/// if retries are exhausted.
///
/// Called on worker startup and periodically by the recovery loop.
/// Does not execute steps or call on_cancel — only resets job status
/// and logs events.
pub(crate) async fn recover_orphans(
    db: &PgPool,
    config: &WorkerConfig,
) -> Result<(), PipelineError> {
    // 1. Zombie recovery
    let zombies = recover_zombie(
        db,
        config.recovery_wakeup_secs as i64,
        config.zombie_threshold_secs as i64,
    )
    .await?;

    for (job_id, step_name, tried, max_retries) in &zombies {
        if *tried >= *max_retries {
            fail_exhausted(
                db,
                *job_id,
                "Zombie detected: heartbeat stale, retries exhausted",
            )
            .await?;

            let details = serde_json::json!({
                "tried": tried,
                "max_retries": max_retries,
                "threshold_secs": config.zombie_threshold_secs,
            });
            events::log(
                db,
                *job_id,
                step_name,
                events::EVT_STEP_EXHAUSTED,
                "Zombie detected: heartbeat stale, retries exhausted",
                Some(&details),
            )
            .await?;
        } else {
            let details = serde_json::json!({
                "tried": tried,
                "max_retries": max_retries,
                "threshold_secs": config.zombie_threshold_secs,
            });
            events::log(
                db,
                *job_id,
                step_name,
                events::EVT_ZOMBIE_RECOVERED,
                "Zombie detected: heartbeat stale, reset to ready for retry",
                Some(&details),
            )
            .await?;
        }
    }

    if !zombies.is_empty() {
        tracing::info!(count = zombies.len(), "Recovery: zombies processed");
    }

    // 2. Timeout recovery
    let timeouts = recover_timeout(db, config.recovery_wakeup_secs as i64).await?;

    for (job_id, step_name, tried, max_retries) in &timeouts {
        if *tried >= *max_retries {
            fail_exhausted(
                db,
                *job_id,
                "Timeout detected: deadline exceeded, retries exhausted",
            )
            .await?;

            let details = serde_json::json!({
                "tried": tried,
                "max_retries": max_retries,
            });
            events::log(
                db,
                *job_id,
                step_name,
                events::EVT_STEP_EXHAUSTED,
                "Timeout detected: deadline exceeded, retries exhausted",
                Some(&details),
            )
            .await?;
        } else {
            let details = serde_json::json!({
                "tried": tried,
                "max_retries": max_retries,
            });
            events::log(
                db,
                *job_id,
                step_name,
                events::EVT_TIMEOUT_RECOVERED,
                "Timeout detected: deadline exceeded, reset to ready for retry",
                Some(&details),
            )
            .await?;
        }
    }

    if !timeouts.is_empty() {
        tracing::info!(count = timeouts.len(), "Recovery: timeouts processed");
    }

    Ok(())
}

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod tests;
