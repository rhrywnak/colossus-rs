//! worker/executor.rs
//!
//! Runs a single pipeline step inside `tokio::select!` with timeout and
//! cancel_watcher racing against the step future.
//!
//! The executor is the only place in the framework that calls `tokio::select!`.
//! Steps themselves must never call `tokio::spawn` or `tokio::select!`
//! internally (hard guarantee G3).
//!
//! ## Rust Learning: `tokio::select!` with `biased`
//!
//! `tokio::select!` polls multiple futures concurrently and executes the
//! branch of the first one that completes. The `biased;` directive makes
//! polling order deterministic — branches are checked top-to-bottom. This
//! ensures that if both execute and timeout complete simultaneously, the
//! execute result wins (it is listed first).
//!
//! ## Rust Learning: `OptionFuture`
//!
//! `futures::future::OptionFuture` wraps an `Option<Future>`. When the
//! inner Option is `None`, the future never resolves — the `select!` branch
//! using `Some(_) = &mut option_fut` is permanently disabled. This gives
//! us optional timeout without any branching logic outside `select!`.

use futures::future::OptionFuture;
use sqlx::PgPool;

use crate::cancel::CancellationToken;
use crate::progress::ProgressReporter;
use crate::schema::JobRow;
use crate::step::{ExecutionResult, StepResult};
use crate::task::Task;
use crate::worker::config::WorkerConfig;
use crate::worker::heartbeat::spawn_heartbeat;

/// Cancel-watcher polling query — reads the raw control column text.
const CANCEL_POLL_SQL: &str = "SELECT control FROM pipeline_jobs WHERE id = $1";

/// Execute a single step of a pipeline job.
///
/// Deserializes `T` from `job.step_data`, constructs cancellation and
/// progress infrastructure, then races the step execution against an
/// optional timeout and a cancel-watcher that polls the database.
///
/// Returns an `ExecutionResult` that the Worker uses to drive the FSM.
/// The heartbeat task is always aborted before returning.
pub(crate) async fn execute_step<T: Task>(
    job: &JobRow,
    db: &PgPool,
    context: &T::Context,
    config: &WorkerConfig,
) -> ExecutionResult<T> {
    // 1. Deserialize the task from step_data JSONB.
    let task: T = match serde_json::from_value(job.step_data.clone()) {
        Ok(t) => t,
        Err(e) => return ExecutionResult::Failed(format!("Failed to deserialize step_data: {e}")),
    };

    // 2. Clone for cancel/timeout branches — execute_current consumes self.
    let task_for_cancel = task.clone();

    // 3. Construct cancellation token and progress reporter.
    let cancel_token = CancellationToken::new();
    let progress = ProgressReporter::new(db.clone(), job.id);

    // 4. Spawn heartbeat background task.
    let heartbeat_handle = spawn_heartbeat(db.clone(), job.id, config.heartbeat_interval_secs);

    // 5. Build the execute future (consumes task by value).
    let execute_future = task.execute_current(db, context, &cancel_token, &progress);

    // 6. Build the cancel_watcher — polls DB control column.
    let cancel_db = db.clone();
    let cancel_job_id = job.id;
    let cancel_interval = config.cancel_poll_interval_secs;
    let cancel_watcher = async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(cancel_interval)).await;
            let result: Result<Option<String>, _> = sqlx::query_scalar(CANCEL_POLL_SQL)
                .bind(cancel_job_id)
                .fetch_optional(&cancel_db)
                .await;
            match result {
                Err(e) => {
                    tracing::trace!(
                        %cancel_job_id,
                        error = %e,
                        "Cancel watcher: DB poll error, will retry next interval"
                    );
                    continue;
                }
                Ok(None) => break,
                Ok(Some(control)) if control == "cancel_requested" => break,
                Ok(Some(_)) => continue,
            }
        }
    };

    // 7. Build the optional timeout future via OptionFuture.
    let timeout_duration = job.timeout_at.map(|t| {
        let remaining = t - chrono::Utc::now();
        remaining.to_std().unwrap_or(std::time::Duration::ZERO)
    });
    let timeout_fut: OptionFuture<_> = timeout_duration.map(tokio::time::sleep).into();
    tokio::pin!(timeout_fut);

    // 8. Race: execute vs timeout vs cancel_watcher.
    let result = tokio::select! {
        biased;

        exec_result = execute_future => {
            match exec_result {
                Ok(step_result) => match step_result {
                    StepResult::Next(next_task) => {
                        ExecutionResult::Success(
                            StepResult::Next(next_task),
                            serde_json::Value::Null,
                        )
                    }
                    StepResult::Delay(next_task, duration) => {
                        ExecutionResult::Success(
                            StepResult::Delay(next_task, duration),
                            serde_json::Value::Null,
                        )
                    }
                    StepResult::Done => ExecutionResult::Done(serde_json::Value::Null),
                    StepResult::WaitForInput(parked_task) => {
                        ExecutionResult::WaitForInput(parked_task)
                    }
                },
                Err(e) => ExecutionResult::Failed(e.to_string()),
            }
        }

        Some(_) = &mut timeout_fut => {
            cancel_token.trigger();
            let _ = task_for_cancel.on_cancel_current(db, context).await;
            ExecutionResult::Timeout
        }

        _ = cancel_watcher => {
            let _ = task_for_cancel.on_cancel_current(db, context).await;
            ExecutionResult::Cancelled
        }
    };

    // 10. Always abort heartbeat before returning.
    heartbeat_handle.abort();
    result
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
