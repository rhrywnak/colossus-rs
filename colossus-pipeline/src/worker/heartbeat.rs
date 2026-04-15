//! worker/heartbeat.rs
//!
//! Periodic heartbeat task for running pipeline jobs.
//!
//! Updates `pipeline_jobs.last_heartbeat_at` on a fixed interval so the
//! recovery manager can distinguish live jobs from zombies. The heartbeat
//! runs as a **separate `tokio::spawn` task**, independent of step execution,
//! because LLM calls and other long-running steps may block for minutes
//! without yielding — if the heartbeat shared the step's task, a slow step
//! would starve the heartbeat and trigger false zombie detection.
//!
//! ## How the heartbeat stops
//!
//! 1. **Self-stop:** when `rows_affected == 0` the job is no longer in
//!    `Running` status (completed, failed, or cancelled), so the loop breaks.
//! 2. **External abort:** the executor calls `JoinHandle::abort()` after the
//!    job finishes, which cancels the spawned task immediately.

use std::time::Duration;

use crate::schema::JobStatus;

/// Spawn a background task that periodically touches `last_heartbeat_at`
/// for the given job.
///
/// # Parameters
///
/// - `db` — shared connection pool; cloned into the spawned task.
/// - `job_id` — the `pipeline_jobs.id` to heartbeat.
/// - `interval_secs` — seconds between heartbeat writes.
///
/// # Returns
///
/// A `JoinHandle<()>` the caller can `.abort()` when the job finishes.
pub fn spawn_heartbeat(
    db: sqlx::PgPool,
    job_id: uuid::Uuid,
    interval_secs: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;

            let result = sqlx::query(
                "UPDATE pipeline_jobs \
                 SET last_heartbeat_at = NOW(), updated_at = NOW() \
                 WHERE id = $1 AND status = $2",
            )
            .bind(job_id)
            .bind(JobStatus::Running.as_str())
            .execute(&db)
            .await;

            match result {
                Ok(pg_result) => {
                    if pg_result.rows_affected() == 0 {
                        tracing::debug!(
                            %job_id,
                            "Heartbeat: job no longer running, stopping heartbeat task"
                        );
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        %job_id,
                        error = %e,
                        "Heartbeat: database error, will retry next interval"
                    );
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requires DATABASE_URL env var pointing to a live PostgreSQL instance
    #[tokio::test]
    #[ignore]
    async fn test_spawn_heartbeat_updates_last_heartbeat_at() {
        let db_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set for this test");
        let db = sqlx::PgPool::connect(&db_url).await.unwrap();
        let job_id = uuid::Uuid::new_v4();

        // Insert a Running job
        sqlx::query(
            "INSERT INTO pipeline_jobs (id, job_type, job_key, pipeline_version, \
             status, control, current_step, step_data, result, tried, max_retries, \
             retry_delay_secs, priority, wakeup_at, created_at, updated_at) \
             VALUES ($1, 'test', 'hb-test', 1, $2, 'none', 'Init', '{}', '{}', \
             0, 3, 60, 0, NOW(), NOW(), NOW())",
        )
        .bind(job_id)
        .bind(JobStatus::Running.as_str())
        .execute(&db)
        .await
        .unwrap();

        let handle = spawn_heartbeat(db.clone(), job_id, 1);

        tokio::time::sleep(Duration::from_secs(3)).await;

        let row: (Option<chrono::DateTime<chrono::Utc>>,) = sqlx::query_as(
            "SELECT last_heartbeat_at FROM pipeline_jobs WHERE id = $1",
        )
        .bind(job_id)
        .fetch_one(&db)
        .await
        .unwrap();

        assert!(row.0.is_some(), "last_heartbeat_at should have been set");

        handle.abort();

        // Cleanup
        sqlx::query("DELETE FROM pipeline_jobs WHERE id = $1")
            .bind(job_id)
            .execute(&db)
            .await
            .unwrap();
    }

    // Requires DATABASE_URL env var pointing to a live PostgreSQL instance
    #[tokio::test]
    #[ignore]
    async fn test_spawn_heartbeat_stops_when_job_completed() {
        let db_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set for this test");
        let db = sqlx::PgPool::connect(&db_url).await.unwrap();
        let job_id = uuid::Uuid::new_v4();

        // Insert a Running job
        sqlx::query(
            "INSERT INTO pipeline_jobs (id, job_type, job_key, pipeline_version, \
             status, control, current_step, step_data, result, tried, max_retries, \
             retry_delay_secs, priority, wakeup_at, created_at, updated_at) \
             VALUES ($1, 'test', 'hb-stop', 1, $2, 'none', 'Init', '{}', '{}', \
             0, 3, 60, 0, NOW(), NOW(), NOW())",
        )
        .bind(job_id)
        .bind(JobStatus::Running.as_str())
        .execute(&db)
        .await
        .unwrap();

        let handle = spawn_heartbeat(db.clone(), job_id, 1);

        // Transition to completed
        sqlx::query("UPDATE pipeline_jobs SET status = $1 WHERE id = $2")
            .bind(JobStatus::Completed.as_str())
            .bind(job_id)
            .execute(&db)
            .await
            .unwrap();

        // Wait for heartbeat to detect rows_affected == 0
        tokio::time::sleep(Duration::from_secs(3)).await;

        assert!(
            handle.is_finished(),
            "heartbeat should have stopped after job completed"
        );

        // Cleanup
        sqlx::query("DELETE FROM pipeline_jobs WHERE id = $1")
            .bind(job_id)
            .execute(&db)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_spawn_heartbeat_abort_stops_immediately() {
        // Does not need a real database — abort fires before the first sleep completes.
        // We use a pool that will never connect, but that is fine because
        // the task sleeps first and we abort before the sleep finishes.
        let db = sqlx::PgPool::connect_lazy("postgres://invalid:5432/none")
            .expect("connect_lazy should not fail");

        let job_id = uuid::Uuid::new_v4();
        let handle = spawn_heartbeat(db, job_id, 3600);

        handle.abort();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(
            handle.is_finished(),
            "heartbeat should be finished after abort"
        );
    }
}
