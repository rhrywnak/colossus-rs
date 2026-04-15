//! scheduler.rs
//!
//! Public API for submitting and managing pipeline jobs.
//!
//! The Scheduler is the only entry point for job submission. It enforces
//! the deduplication invariant (one active job per job_type+job_key),
//! generates UUID v7 job IDs (time-ordered for efficient index scans),
//! and provides status query, cancel, resume, and delete operations.
//!
//! Application code calls Scheduler methods directly. The Worker is
//! internal — application code does not call Worker methods.
//!
//! ## Rust Learning: method-level generics vs struct-level generics
//!
//! The Scheduler struct holds only a `&PgPool` — it is not generic over
//! `Task`. Instead, methods that need the Task type (submit, delete) are
//! generic on the method itself: `fn submit<T: Task>(...)`. This keeps
//! the Scheduler concrete and shareable while allowing different Task
//! types in the same application.

use sqlx::PgPool;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::events;
use crate::schema::{JobControl, JobRow, JobStatus, PipelineEvent};
use crate::task::Task;
use crate::worker::fetcher;
use crate::worker::fetcher_api;

/// Pipeline version written to every new job row.
/// Increment when the step graph or step_data schema changes in a
/// backwards-incompatible way. The Worker can use this to reject jobs
/// from an older pipeline version after a deployment.
const PIPELINE_VERSION: i32 = 1;

/// SQL for inserting a new pipeline job with all required columns.
const INSERT_JOB_SQL: &str = r#"
    INSERT INTO pipeline_jobs (
        id, job_type, job_key, pipeline_version,
        status, control, current_step, step_data,
        result, tried, max_retries, retry_delay_secs,
        priority, wakeup_at, created_by,
        created_at, updated_at
    ) VALUES (
        $1, $2, $3, $4,
        $5, $6, $7, $8,
        '{}', 0, 0, 0,
        $9, NOW(), $10,
        NOW(), NOW()
    )
"#;

/// PostgreSQL error code for unique constraint violation.
const PG_UNIQUE_VIOLATION: &str = "23505";

/// Public API for submitting and managing pipeline jobs.
///
/// Holds a reference to the database pool. Constructed once at application
/// startup and shared across request handlers.
pub struct Scheduler<'a> {
    db: &'a PgPool,
}

impl<'a> Scheduler<'a> {
    /// Create a new Scheduler backed by the given connection pool.
    pub fn new(db: &'a PgPool) -> Self {
        Self { db }
    }

    /// Submit a new job to the pipeline queue.
    ///
    /// Generates a UUID v7 (time-ordered), serializes the initial task to
    /// JSONB, and inserts a Ready job row. Returns the job ID on success.
    /// Returns `DuplicateJob` if an active job already exists for this
    /// (job_type, job_key) combination.
    pub async fn submit<T: Task>(
        &self,
        job_type: &str,
        job_key: &str,
        initial_task: T,
        priority: i32,
        created_by: Option<&str>,
    ) -> Result<Uuid, PipelineError> {
        let job_id = Uuid::now_v7();
        let step_data = serde_json::to_value(&initial_task)?;
        let step_name = initial_task.current_step_name();

        let result = sqlx::query(INSERT_JOB_SQL)
            .bind(job_id)
            .bind(job_type)
            .bind(job_key)
            .bind(PIPELINE_VERSION)
            .bind(JobStatus::Ready.as_str())
            .bind(JobControl::None.as_str())
            .bind(step_name)
            .bind(&step_data)
            .bind(priority)
            .bind(created_by)
            .execute(self.db)
            .await;

        match result {
            Ok(_) => {}
            Err(e) => {
                if is_unique_violation(&e) {
                    return Err(PipelineError::DuplicateJob {
                        job_type: job_type.to_string(),
                        job_key: job_key.to_string(),
                    });
                }
                return Err(PipelineError::Database(e.to_string()));
            }
        }

        events::log(
            self.db,
            job_id,
            step_name,
            events::EVT_JOB_SUBMITTED,
            "Job submitted",
            None,
        )
        .await?;

        Ok(job_id)
    }

    /// Fetch the current state of a job by ID.
    ///
    /// Returns the full `JobRow` or `NotFound` if the job does not exist.
    pub async fn status(&self, job_id: Uuid) -> Result<JobRow, PipelineError> {
        let row = sqlx::query_as::<_, JobRow>(
            "SELECT * FROM pipeline_jobs WHERE id = $1",
        )
        .bind(job_id)
        .fetch_optional(self.db)
        .await?;

        row.ok_or(PipelineError::NotFound(job_id))
    }

    /// Find the most recent active job for a (job_type, job_key) pair.
    ///
    /// Returns `None` if no active job exists. Active means status is not
    /// in a terminal state (completed, failed, cancelled).
    pub async fn status_by_key(
        &self,
        job_type: &str,
        job_key: &str,
    ) -> Result<Option<JobRow>, PipelineError> {
        let row = sqlx::query_as::<_, JobRow>(
            "SELECT * FROM pipeline_jobs \
             WHERE job_type = $1 AND job_key = $2 \
               AND status NOT IN ($3, $4, $5) \
             ORDER BY created_at DESC LIMIT 1",
        )
        .bind(job_type)
        .bind(job_key)
        .bind(JobStatus::Completed.as_str())
        .bind(JobStatus::Failed.as_str())
        .bind(JobStatus::Cancelled.as_str())
        .fetch_optional(self.db)
        .await?;

        Ok(row)
    }

    /// Cancel a job.
    ///
    /// If the job is Ready (not yet claimed by a worker), cancels it
    /// immediately by setting status=Failed. If the job is Running,
    /// sets `control = cancel_requested` so the executor's cancel_watcher
    /// detects it on its next poll. Returns `JobNotCancellable` if the
    /// job is already in a terminal state.
    pub async fn cancel(&self, job_id: Uuid) -> Result<(), PipelineError> {
        let job = self.status(job_id).await?;

        match job.status {
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled => {
                return Err(PipelineError::JobNotCancellable(job_id));
            }
            JobStatus::Ready => {
                // Ready jobs have no cancel_watcher — cancel directly.
                fetcher::cancel(self.db, job_id).await?;
                events::log(
                    self.db,
                    job_id,
                    &job.current_step,
                    events::EVT_CANCELLED,
                    "Job cancelled (was not yet running)",
                    None,
                )
                .await?;
            }
            JobStatus::Running => {
                // Running jobs are cancelled via the cancel_watcher.
                fetcher_api::request_cancel(self.db, job_id).await?;
                events::log(
                    self.db,
                    job_id,
                    &job.current_step,
                    events::EVT_CANCEL_REQUESTED,
                    "Cancellation requested",
                    None,
                )
                .await?;
            }
        }

        Ok(())
    }

    /// Resume a failed or parked job.
    ///
    /// Delegates to `fetcher_api::resume` which accepts both Failed jobs and
    /// Ready+WaitingForInput jobs. Returns `JobNotResumable` if the job
    /// is not in a resumable state.
    pub async fn resume(&self, job_id: Uuid) -> Result<(), PipelineError> {
        // fetcher_api::resume handles the precondition check and returns
        // JobNotResumable if rows_affected == 0.
        fetcher_api::resume(self.db, job_id).await?;

        let job = self.status(job_id).await?;
        events::log(
            self.db,
            job_id,
            &job.current_step,
            events::EVT_RESUMED,
            "Job resumed",
            None,
        )
        .await?;

        Ok(())
    }

    /// Delete a job and its associated events.
    ///
    /// Calls `on_delete_current` on the task to clean up external side
    /// effects before removing the row. Returns `JobRunning` if the job
    /// is currently executing, `NotFound` if it does not exist.
    ///
    /// Pipeline_events rows are cascade-deleted by the FK constraint,
    /// so the event log is not preserved after deletion. A tracing log
    /// records the deletion for server-side audit.
    pub async fn delete<T: Task>(
        &self,
        job_id: Uuid,
        context: &T::Context,
    ) -> Result<(), PipelineError> {
        let job = self.status(job_id).await?;

        if job.status == JobStatus::Running {
            return Err(PipelineError::JobRunning(job_id));
        }

        let task: T = serde_json::from_value(job.step_data.clone())
            .map_err(|e| PipelineError::Serialization(e.to_string()))?;

        task.on_delete_current(self.db, context)
            .await
            .map_err(|e| PipelineError::Cleanup(e.to_string()))?;

        tracing::info!(
            %job_id,
            job_type = %job.job_type,
            job_key = %job.job_key,
            step = %job.current_step,
            "Job deleted"
        );

        sqlx::query("DELETE FROM pipeline_jobs WHERE id = $1")
            .bind(job_id)
            .execute(self.db)
            .await?;

        Ok(())
    }

    /// Advance a parked job to a specific next step.
    ///
    /// Validates the transition via `T::validate_transition`, then calls
    /// `advance_from_park` to clear WaitingForInput and set the next step.
    /// Used when external input has been received and the job should
    /// continue to a specific next step.
    pub async fn advance<T: Task>(
        &self,
        job_id: Uuid,
        next_task: T,
    ) -> Result<(), PipelineError> {
        let job = self.status(job_id).await?;
        let next_step_name = next_task.current_step_name();

        T::validate_transition(&job.current_step, next_step_name)?;

        let next_step_data = serde_json::to_value(&next_task)?;

        fetcher_api::advance_from_park(
            self.db,
            job_id,
            &next_step_data,
            next_step_name,
        )
        .await?;

        events::log(
            self.db,
            job_id,
            next_step_name,
            events::EVT_ADVANCED,
            "Job advanced to next step",
            None,
        )
        .await?;

        Ok(())
    }

    /// Fetch the event log for a job, ordered chronologically.
    ///
    /// Returns an empty Vec if the job has no events (not an error).
    pub async fn events(
        &self,
        job_id: Uuid,
    ) -> Result<Vec<PipelineEvent>, PipelineError> {
        let rows = sqlx::query_as::<_, PipelineEvent>(
            "SELECT * FROM pipeline_events WHERE job_id = $1 ORDER BY created_at ASC",
        )
        .bind(job_id)
        .fetch_all(self.db)
        .await?;

        Ok(rows)
    }

    /// List jobs with optional filters, ordered by creation time descending.
    ///
    /// Both `job_type` and `status` are optional filters. When `None`, the
    /// filter is not applied. Results are paginated via `limit` and `offset`.
    pub async fn list(
        &self,
        job_type: Option<&str>,
        status: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<JobRow>, PipelineError> {
        let rows = match (job_type, status) {
            (Some(jt), Some(st)) => {
                sqlx::query_as::<_, JobRow>(
                    "SELECT * FROM pipeline_jobs \
                     WHERE job_type = $1 AND status = $2 \
                     ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                )
                .bind(jt)
                .bind(st)
                .bind(limit)
                .bind(offset)
                .fetch_all(self.db)
                .await?
            }
            (Some(jt), None) => {
                sqlx::query_as::<_, JobRow>(
                    "SELECT * FROM pipeline_jobs \
                     WHERE job_type = $1 \
                     ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                )
                .bind(jt)
                .bind(limit)
                .bind(offset)
                .fetch_all(self.db)
                .await?
            }
            (None, Some(st)) => {
                sqlx::query_as::<_, JobRow>(
                    "SELECT * FROM pipeline_jobs \
                     WHERE status = $1 \
                     ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                )
                .bind(st)
                .bind(limit)
                .bind(offset)
                .fetch_all(self.db)
                .await?
            }
            (None, None) => {
                sqlx::query_as::<_, JobRow>(
                    "SELECT * FROM pipeline_jobs \
                     ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                )
                .bind(limit)
                .bind(offset)
                .fetch_all(self.db)
                .await?
            }
        };

        Ok(rows)
    }
}

/// Check whether a sqlx error is a PostgreSQL unique constraint violation.
///
/// Matches on the PostgreSQL error code "23505" which is the standard code
/// for unique_violation. This is more reliable than string-matching on
/// the error message text.
fn is_unique_violation(e: &sqlx::Error) -> bool {
    if let sqlx::Error::Database(ref db_err) = e {
        if let Some(code) = db_err.code() {
            return code == PG_UNIQUE_VIOLATION;
        }
    }
    false
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
