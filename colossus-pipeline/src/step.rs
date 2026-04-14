//! The Step trait and associated result types.
//!
//! A Step is one unit of work in a pipeline job. It receives the database
//! pool, application context, a cancellation token, and a progress reporter.
//! It returns a StepResult indicating what happens next.
//!
//! Steps MUST be idempotent — the Worker may execute the same step more than
//! once on retry. Every step implementation must check whether its work has
//! already been done before doing it again.
//!
//! ## Rust Learning: associated constants in traits
//!
//! Traits can define associated constants with default values:
//!   const DEFAULT_RETRY_LIMIT: i32 = 0;
//! Implementors can override these per step:
//!   const DEFAULT_RETRY_LIMIT: i32 = 3;
//! The Worker reads these at job submission time to set max_retries and
//! retry_delay_secs on the pipeline_jobs row. This is how per-step retry
//! configuration works without hardcoding values anywhere.
//!
//! ## Rust Learning: generic traits and type parameters
//!
//! Step<T: Task> is generic over T — the Task type that owns this step.
//! This allows StepResult<T> to carry the next step variant as a value,
//! giving the compiler full type safety over step transitions.
//! A step that returns StepResult::Next(T::SomeVariant) is verified at
//! compile time to be a valid variant of that Task's enum.

use std::error::Error;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::progress::ProgressReporter;
use crate::task::Task;

/// What happens after a step finishes executing.
///
/// The Worker reads this value and drives the FSM transition accordingly.
pub enum StepResult<T: Task> {
    /// Advance to the next step. T is the next Task variant to execute.
    Next(T),

    /// Delay before retrying this same step. Used for rate limit backoff
    /// that the step itself detects (separate from the Worker retry mechanism).
    Delay(T, Duration),

    /// All steps complete. Job transitions to Completed.
    Done,

    /// Park the job waiting for external input. Job transitions to
    /// WaitingForInput control state. Execution resumes when resumed via API.
    WaitForInput(T),
}

/// Internal result returned by executor::execute_step.
/// Not part of the public API — the Worker uses this to decide next action.
pub(crate) enum ExecutionResult<T: Task> {
    /// Step completed. Carry the StepResult and any accumulated result JSONB.
    Success(StepResult<T>, serde_json::Value),
    /// Step signalled job completion.
    Done(serde_json::Value),
    /// Step returned WaitForInput.
    WaitForInput(T),
    /// Step returned an error. String is the error message for pipeline_events.
    Failed(String),
    /// Step exceeded its timeout. Worker will call on_cancel and retry or fail.
    Timeout,
    /// Cancel signal was detected. Worker will call on_cancel and mark failed.
    Cancelled,
}

/// A single step in a pipeline job.
///
/// ## Implementation requirements
///
/// 1. execute() MUST be idempotent — check before doing work.
/// 2. execute() MUST NOT call tokio::spawn internally.
/// 3. execute() MUST check cancel.is_cancelled() between expensive operations.
/// 4. on_cancel() and on_delete() MUST clean up all external side effects.
/// 5. All DB writes in execute() that affect multiple tables MUST use transactions.
#[async_trait]
pub trait Step<T: Task>: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static {
    /// Default retry limit when not overridden by step_config JSONB.
    /// 0 means no retries — fail immediately on error.
    const DEFAULT_RETRY_LIMIT: i32 = 0;

    /// Default delay between retries in seconds when not overridden by step_config.
    const DEFAULT_RETRY_DELAY_SECS: u64 = 0;

    /// Default step timeout in seconds. None means no timeout.
    /// Override per step: const DEFAULT_TIMEOUT_SECS: Option<u64> = Some(900);
    const DEFAULT_TIMEOUT_SECS: Option<u64> = None;

    /// Maximum concurrent instances of this step type across all jobs.
    /// None means use the global PIPELINE_MAX_CONCURRENT limit.
    const MAX_CONCURRENCY: Option<u32> = None;

    /// Execute the step's work. The primary method every step implements.
    ///
    /// # Arguments
    /// - db: Database pool for all persistence operations
    /// - context: Application context (Arc<dyn LlmProvider> etc.) — generic parameter
    ///   kept as a type parameter on Task so colossus-pipeline stays domain-agnostic
    /// - cancel: Token to check for cancellation between operations
    /// - progress: Reporter to update pipeline_jobs.progress JSONB
    async fn execute(
        self,
        db: &sqlx::PgPool,
        context: &T::Context,
        cancel: &CancellationToken,
        progress: &ProgressReporter,
    ) -> Result<StepResult<T>, Box<dyn Error + Send + Sync>>;

    /// Called when cancellation is detected mid-step.
    /// Must reverse any partial external side effects (Neo4j writes, Qdrant upserts).
    /// Default implementation does nothing — override when side effects exist.
    async fn on_cancel(
        self,
        _db: &sqlx::PgPool,
        _context: &T::Context,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Called when a job is deleted while this step is current.
    /// Must clean up all external state for this document.
    /// Default implementation does nothing — override when side effects exist.
    async fn on_delete(
        self,
        _db: &sqlx::PgPool,
        _context: &T::Context,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

/// Returns the short name of a type — the last component after "::".
///
/// Used to derive step names from Rust type names without manual string maintenance.
/// step_name_of::<ExtractText>() returns "ExtractText", not the full module path.
///
/// ## Rust Learning: std::any::type_name
///
/// std::any::type_name::<T>() returns the full path of type T as a &'static str,
/// e.g. "colossus_legal::pipeline::steps::extract_text::ExtractText".
/// We split on "::" and take the last segment to get just "ExtractText".
/// This is stable in release builds. The result is used as the current_step
/// column value in pipeline_jobs — it must be consistent across restarts.
pub fn step_name_of<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    // Split on "::" and return the last segment.
    // type_name guarantees at least one segment exists.
    match full.rsplit("::").next() {
        Some(name) => name,
        None => full,
    }
}
