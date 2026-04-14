//! The Task trait — the application-defined enum that drives the pipeline FSM.
//!
//! Each Colossus application defines its own Task enum. For colossus-legal
//! this will be DocProcessing with variants ExtractText, LlmExtract, Ingest,
//! Index, Completeness. For colossus-ai it will be something different.
//! The colossus-pipeline crate knows nothing about either — it only requires
//! that the application type implements this trait.
//!
//! ## Rust Learning: associated types in traits
//!
//! `type Context: Send + Sync + 'static` is an associated type — a placeholder
//! that each implementor fills in with a concrete type. This lets the Worker
//! be generic over both the Task and its Context without colossus-pipeline
//! knowing what AppContext contains. The bounds Send + Sync + 'static are
//! required because the context is shared across tokio tasks.

use std::error::Error;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::error::PipelineError;
use crate::progress::ProgressReporter;
use crate::step::StepResult;

/// Application-defined pipeline task enum.
///
/// Each variant represents one step in the pipeline. The Task impl
/// provides dispatch — the Worker calls execute_current() and the
/// Task routes to the correct Step implementation.
#[async_trait]
pub trait Task: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static {
    /// Application context passed to every step execution.
    /// For colossus-legal this is AppContext (holds Arc<dyn LlmProvider> etc.).
    /// The colossus-pipeline crate places no constraints on its contents
    /// beyond thread-safety.
    type Context: Send + Sync + 'static;

    /// Returns the short name of the current step variant.
    /// Must use step_name_of::<ConcreteStepType>() — never a hand-typed string.
    /// This value is written to pipeline_jobs.current_step.
    fn current_step_name(&self) -> &'static str;

    /// Validate that transitioning from current_step to next_step is permitted.
    /// Returns Ok(()) if valid, Err(InvalidTransition) if not.
    /// Called by the Scheduler before advancing a job.
    fn validate_transition(current: &str, next: &str) -> Result<(), PipelineError>;

    /// Execute the current step variant, routing to the concrete Step impl.
    async fn execute_current(
        self,
        db: &sqlx::PgPool,
        context: &Self::Context,
        cancel: &CancellationToken,
        progress: &ProgressReporter,
    ) -> Result<StepResult<Self>, Box<dyn Error + Send + Sync>>;

    /// Call on_cancel on the current step variant.
    async fn on_cancel_current(
        self,
        db: &sqlx::PgPool,
        context: &Self::Context,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Call on_delete on the current step variant.
    async fn on_delete_current(
        self,
        db: &sqlx::PgPool,
        context: &Self::Context,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}
