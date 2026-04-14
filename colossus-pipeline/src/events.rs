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

pub const EVT_JOB_SUBMITTED:     &str = "job_submitted";
pub const EVT_STEP_STARTED:      &str = "step_started";
pub const EVT_STEP_COMPLETED:    &str = "step_completed";
pub const EVT_STEP_FAILED:       &str = "step_failed";
pub const EVT_STEP_EXHAUSTED:    &str = "step_exhausted";
pub const EVT_RETRY_SCHEDULED:   &str = "retry_scheduled";
pub const EVT_ZOMBIE_RECOVERED:  &str = "zombie_recovered";
pub const EVT_TIMEOUT_RECOVERED: &str = "timeout_recovered";
pub const EVT_CANCELLED:         &str = "cancelled";
pub const EVT_CANCEL_REQUESTED:  &str = "cancel_requested";
pub const EVT_WAITING_INPUT:     &str = "waiting_for_input";
pub const EVT_ADVANCED:          &str = "advanced";
pub const EVT_RESUMED:           &str = "resumed";
pub const EVT_JOB_COMPLETED:     &str = "job_completed";
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
    let message = if message.len() > 500 {
        &message[..500]
    } else {
        message
    };

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
