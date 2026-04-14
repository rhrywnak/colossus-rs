//! Pipeline error type — covers all failure modes in the framework.
//!
//! ## Rust Learning: thiserror
//!
//! `thiserror` is a derive macro that generates std::error::Error and
//! std::fmt::Display from annotated enum variants. #[error("...")] sets
//! the Display message. {0} inserts the inner value. This eliminates
//! hundreds of lines of boilerplate error impl code.

use uuid::Uuid;

/// All error conditions the pipeline framework can produce.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A sqlx database operation failed.
    #[error("Database error: {0}")]
    Database(String),

    /// A job was requested by ID but does not exist.
    #[error("Job not found: {0}")]
    NotFound(Uuid),

    /// Job cannot be cancelled (already completed, failed, or cancelled).
    #[error("Job {0} is not in a cancellable state")]
    JobNotCancellable(Uuid),

    /// Job cannot be resumed (not parked or failed).
    #[error("Job {0} is not in a resumable state")]
    JobNotResumable(Uuid),

    /// Job is currently running and cannot be deleted. Cancel first.
    #[error("Job {0} is currently running — cancel before deleting")]
    JobRunning(Uuid),

    /// An active job already exists for this (job_type, job_key) combination.
    #[error("Active job already exists for type '{job_type}' key '{job_key}'")]
    DuplicateJob { job_type: String, job_key: String },

    /// A state transition was attempted that the FSM does not permit.
    #[error("Invalid transition from '{from}' to '{to}'")]
    InvalidTransition { from: String, to: String },

    /// Serializing or deserializing job state (step_data, result JSONB) failed.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// A cleanup operation (on_delete, on_cancel) failed.
    #[error("Cleanup error: {0}")]
    Cleanup(String),

    /// An LLM provider call failed. Lives here so step implementations
    /// can return PipelineError without depending on colossus-extract.
    #[error("LLM provider error: {0}")]
    LlmProvider(String),
}

impl From<sqlx::Error> for PipelineError {
    fn from(e: sqlx::Error) -> Self {
        PipelineError::Database(e.to_string())
    }
}

impl From<serde_json::Error> for PipelineError {
    fn from(e: serde_json::Error) -> Self {
        PipelineError::Serialization(e.to_string())
    }
}
