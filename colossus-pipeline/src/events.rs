//! Event type constants and the log() function for pipeline_events.
//!
//! Every state transition, retry, recovery, and cancellation is recorded
//! as a pipeline_event row. This provides a complete audit trail for any
//! job and makes debugging production issues possible without log scraping.
//!
//! ## Rust Learning: &'static str constants
//!
//! `pub const EVT_JOB_SUBMITTED: &'static str = "job_submitted"` defines
//! a string constant with 'static lifetime — it lives for the entire program.
//! Using constants instead of string literals means a typo is a compile error.
//! The &'static str type is the correct choice for string constants in Rust.

use sqlx::PgPool;
use uuid::Uuid;

use crate::error::PipelineError;

/// Maximum length of a pipeline event message stored in pipeline_events.
/// Messages longer than this are truncated before INSERT to prevent
/// runaway error messages from filling the events table.
pub(crate) const MAX_EVENT_MESSAGE_LEN: usize = 500;

/// Fired when a new job is submitted to the queue via Scheduler::submit().
pub const EVT_JOB_SUBMITTED:     &str = "job_submitted";

/// Fired when the Worker begins executing a step.
pub const EVT_STEP_STARTED:      &str = "step_started";

/// Fired when a step completes successfully and the job advances.
pub const EVT_STEP_COMPLETED:    &str = "step_completed";

/// Fired when a step returns an error and a retry will be attempted.
pub const EVT_STEP_FAILED:       &str = "step_failed";

/// Fired when a step fails and retries are exhausted — job transitions to Failed.
pub const EVT_STEP_EXHAUSTED:    &str = "step_exhausted";

/// Fired when a retry is scheduled — records the delay before next attempt.
pub const EVT_RETRY_SCHEDULED:   &str = "retry_scheduled";

/// Fired when the recovery manager detects and resets a zombie job.
pub const EVT_ZOMBIE_RECOVERED:  &str = "zombie_recovered";

/// Fired when the recovery manager detects and resets a timed-out job.
pub const EVT_TIMEOUT_RECOVERED: &str = "timeout_recovered";

/// Fired when cancellation completes and the job transitions to Failed.
pub const EVT_CANCELLED:         &str = "cancelled";

/// Fired when a cancel request is received and control=cancel_requested is set.
pub const EVT_CANCEL_REQUESTED:  &str = "cancel_requested";

/// Fired when a job parks itself waiting for external input.
pub const EVT_WAITING_INPUT:     &str = "waiting_for_input";

/// Fired when a job advances from one step to the next successfully.
pub const EVT_ADVANCED:          &str = "advanced";

/// Fired when a parked or failed job is resumed by the user.
pub const EVT_RESUMED:           &str = "resumed";

/// Fired when all steps complete and the job reaches the Completed state.
pub const EVT_JOB_COMPLETED:     &str = "job_completed";

/// Fired when a job and all its events are deleted from the database.
pub const EVT_JOB_DELETED:       &str = "job_deleted";

/// Insert a row into pipeline_events.
///
/// Called by the Worker at every state transition. The step name comes
/// from step_name_of::<T>() — never a manually typed string.
/// The message is human-readable. Details is optional structured context.
///
/// Truncates message to 500 characters if longer — pipeline_events is
/// append-only and high-volume; runaway error messages must not fill the table.
pub async fn log(
    db: &PgPool,
    job_id: Uuid,
    step: &str,
    event_type: &str,
    message: &str,
    details: Option<&serde_json::Value>,
) -> Result<(), PipelineError> {
    let message = truncate_message(message);

    sqlx::query(
        "INSERT INTO pipeline_events (job_id, step, event_type, message, details)
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(job_id)
    .bind(step)
    .bind(event_type)
    .bind(message)
    .bind(details)
    .execute(db)
    .await?;

    Ok(())
}

/// Truncate a message to 500 characters if it exceeds that length.
///
/// Used by [`log()`] to prevent runaway error messages from filling the
/// pipeline_events table. Returns the original string unchanged if it is
/// 500 characters or shorter.
fn truncate_message(message: &str) -> &str {
    if message.len() > MAX_EVENT_MESSAGE_LEN {
        &message[..MAX_EVENT_MESSAGE_LEN]
    } else {
        message
    }
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
