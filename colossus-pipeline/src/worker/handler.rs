//! worker/handler.rs
//!
//! Maps `ExecutionResult` to SQL transitions and event logs.
//!
//! After the executor finishes a step (via `tokio::select!`), the Worker
//! calls `handle_execution_result` to advance, complete, retry, fail,
//! or cancel the job based on the outcome. All SQL transitions are
//! delegated to fetcher functions — this module contains no raw SQL.

use sqlx::PgPool;

use crate::error::PipelineError;
use crate::events;
use crate::schema::JobRow;
use crate::step::{ExecutionResult, StepResult};
use crate::task::Task;
use crate::worker::config::WorkerConfig;
use crate::worker::fetcher;
use crate::worker::retry::compute_delay;

/// Error message used when a step times out.
const STEP_TIMED_OUT: &str = "Step timed out";

/// Map an `ExecutionResult` to the appropriate SQL transition and event log.
///
/// Called by `try_claim_and_run` after `execute_step` returns. Each variant
/// drives a different FSM transition via the corresponding fetcher function.
pub(crate) async fn handle_execution_result<T: Task>(
    db: &PgPool,
    job: &JobRow,
    result: ExecutionResult<T>,
    _config: &WorkerConfig,
) -> Result<(), PipelineError> {
    match result {
        ExecutionResult::Success(step_result, accumulated_result) => {
            handle_success(db, job, step_result, &accumulated_result).await
        }
        ExecutionResult::Done(accumulated_result) => {
            handle_done(db, job, &accumulated_result).await
        }
        ExecutionResult::WaitForInput(parked_task) => {
            handle_wait_for_input(db, job, parked_task).await
        }
        ExecutionResult::Failed(error_message) => {
            handle_failed(db, job, &error_message).await
        }
        ExecutionResult::Timeout => {
            handle_failed(db, job, STEP_TIMED_OUT).await
        }
        ExecutionResult::Cancelled => {
            fetcher::cancel(db, job.id).await?;
            events::log(
                db,
                job.id,
                &job.current_step,
                events::EVT_CANCELLED,
                "Job cancelled",
                None,
            )
            .await?;
            Ok(())
        }
    }
}

/// Handle a successful step that returned `StepResult`.
async fn handle_success<T: Task>(
    db: &PgPool,
    job: &JobRow,
    step_result: StepResult<T>,
    accumulated_result: &serde_json::Value,
) -> Result<(), PipelineError> {
    match step_result {
        StepResult::Next(next_task) => {
            let next_step_name = next_task.current_step_name();
            let next_step_data = serde_json::to_value(&next_task)
                .map_err(|e| PipelineError::Serialization(e.to_string()))?;

            events::log(
                db,
                job.id,
                &job.current_step,
                events::EVT_STEP_COMPLETED,
                "Step completed, advancing",
                None,
            )
            .await?;

            fetcher::advance(
                db,
                job.id,
                &next_step_data,
                next_step_name,
                accumulated_result,
            )
            .await?;

            Ok(())
        }

        StepResult::Delay(next_task, duration) => {
            let next_step_name = next_task.current_step_name();
            let next_step_data = serde_json::to_value(&next_task)
                .map_err(|e| PipelineError::Serialization(e.to_string()))?;
            let delay_secs = duration.as_secs() as i64;

            let details = serde_json::json!({"delay_secs": delay_secs});
            events::log(
                db,
                job.id,
                &job.current_step,
                events::EVT_STEP_COMPLETED,
                "Step completed, delaying before next execution",
                Some(&details),
            )
            .await?;

            fetcher::advance_with_delay(
                db,
                job.id,
                &next_step_data,
                next_step_name,
                accumulated_result,
                delay_secs,
            )
            .await?;

            Ok(())
        }

        StepResult::Done => {
            handle_done(db, job, accumulated_result).await
        }

        StepResult::WaitForInput(parked_task) => {
            handle_wait_for_input(db, job, parked_task).await
        }
    }
}

/// Handle job completion — all steps done.
async fn handle_done(
    db: &PgPool,
    job: &JobRow,
    accumulated_result: &serde_json::Value,
) -> Result<(), PipelineError> {
    events::log(
        db,
        job.id,
        &job.current_step,
        events::EVT_STEP_COMPLETED,
        "Step completed",
        None,
    )
    .await?;

    fetcher::complete(db, job.id, accumulated_result).await?;

    events::log(
        db,
        job.id,
        &job.current_step,
        events::EVT_JOB_COMPLETED,
        "All steps complete",
        None,
    )
    .await?;

    Ok(())
}

/// Handle a step parking for external input.
async fn handle_wait_for_input<T: Task>(
    db: &PgPool,
    job: &JobRow,
    parked_task: T,
) -> Result<(), PipelineError> {
    let parked_data = serde_json::to_value(&parked_task)
        .map_err(|e| PipelineError::Serialization(e.to_string()))?;

    fetcher::park(db, job.id, &parked_data).await?;

    events::log(
        db,
        job.id,
        &job.current_step,
        events::EVT_WAITING_INPUT,
        "Step parked waiting for external input",
        None,
    )
    .await?;

    Ok(())
}

/// Handle a failed or timed-out step — retry or exhaust.
async fn handle_failed(
    db: &PgPool,
    job: &JobRow,
    error_message: &str,
) -> Result<(), PipelineError> {
    events::log(
        db,
        job.id,
        &job.current_step,
        events::EVT_STEP_FAILED,
        error_message,
        None,
    )
    .await?;

    if job.tried < job.max_retries {
        let delay = compute_delay(job.retry_delay_secs, job.tried + 1);

        fetcher::fail_with_retry(db, job.id, error_message, delay).await?;

        let details = serde_json::json!({
            "attempt": job.tried + 1,
            "max_retries": job.max_retries,
            "delay_secs": delay,
        });
        events::log(
            db,
            job.id,
            &job.current_step,
            events::EVT_RETRY_SCHEDULED,
            "Retry scheduled",
            Some(&details),
        )
        .await?;
    } else {
        fetcher::fail_exhausted(db, job.id, error_message).await?;

        events::log(
            db,
            job.id,
            &job.current_step,
            events::EVT_STEP_EXHAUSTED,
            error_message,
            None,
        )
        .await?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "handler_tests.rs"]
mod tests;
