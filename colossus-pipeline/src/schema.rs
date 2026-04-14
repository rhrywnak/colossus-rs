//! Database row types and status enums for the pipeline_jobs and
//! pipeline_events tables.
//!
//! ## Rust Learning: sqlx::Type derive
//!
//! #[derive(sqlx::Type)] with #[sqlx(type_name = "text", rename_all = "snake_case")]
//! teaches sqlx how to map a Rust enum to a PostgreSQL TEXT column.
//! rename_all = "snake_case" converts variant names automatically:
//! CancelRequested → "cancel_requested", Ready → "ready".
//! We never write status strings in SQL — we bind enum values.
//! A typo in a status string becomes a compile error, not a silent runtime bug.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Job lifecycle status. Managed exclusively by the Worker.
/// Application code never sets this directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum JobStatus {
    /// Waiting to be claimed by a worker.
    Ready,
    /// Currently executing on a worker.
    Running,
    /// All steps completed successfully.
    Completed,
    /// Terminal failure — retries exhausted or unrecoverable error.
    Failed,
    /// Cancelled after user request reached a terminal state.
    Cancelled,
}

impl JobStatus {
    /// Return the database string representation of this status value.
    ///
    /// This is the single source of truth for status strings —
    /// used when binding to SQL parameters so the string is never
    /// manually typed in query code.
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Ready => "ready",
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
}

/// Control signals set by external actors (API handlers), read by the Worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum JobControl {
    /// No pending control signal. Normal execution continues.
    None,
    /// User requested cancellation. Executor stops after current step.
    CancelRequested,
    /// Job is parked waiting for external input (e.g., human review).
    WaitingForInput,
}

impl JobControl {
    /// Return the database string representation of this control value.
    ///
    /// Used when binding to SQL parameters — never type control strings manually.
    pub fn as_str(&self) -> &'static str {
        match self {
            JobControl::None => "none",
            JobControl::CancelRequested => "cancel_requested",
            JobControl::WaitingForInput => "waiting_for_input",
        }
    }
}

/// A complete row from the pipeline_jobs table.
///
/// All columns present — no partial structs. sqlx::FromRow maps every
/// column by name. Optional columns use Option<T>.
///
/// ## Rust Learning: sqlx::FromRow
///
/// #[derive(sqlx::FromRow)] generates code that reads a database row
/// and maps each column to a struct field by matching column names to
/// field names. The field type must implement sqlx::Decode for the
/// corresponding SQL type. DateTime<Utc> maps to TIMESTAMPTZ.
/// serde_json::Value maps to JSONB.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct JobRow {
    pub id: Uuid,
    pub job_type: String,
    pub job_key: String,
    pub pipeline_version: i32,
    pub status: JobStatus,
    pub control: JobControl,
    pub current_step: String,
    pub step_data: serde_json::Value,
    pub result: serde_json::Value,
    pub tried: i32,
    pub max_retries: i32,
    pub retry_delay_secs: i32,
    pub priority: i32,
    pub wakeup_at: DateTime<Utc>,
    pub step_started_at: Option<DateTime<Utc>>,
    pub step_completed_at: Option<DateTime<Utc>>,
    pub timeout_at: Option<DateTime<Utc>>,
    pub worker_id: Option<String>,
    pub last_heartbeat_at: Option<DateTime<Utc>>,
    pub progress: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_by: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests;

/// A complete row from the pipeline_events table.
/// Every state transition and notable occurrence is recorded here.
/// Provides a full audit trail for any job.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PipelineEvent {
    pub id: i64,
    pub job_id: Uuid,
    pub step: String,
    pub event_type: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}
