//! # colossus-pipeline
//!
//! Domain-agnostic async job pipeline framework for Colossus applications.
//!
//! Provides persistent job execution with FSM-enforced state transitions,
//! heartbeat-based zombie detection, cancellation, retry with backoff,
//! and crash recovery — all backed by PostgreSQL with no external broker.
//!
//! ## Design goals
//!
//! - Zero knowledge of LLMs, embeddings, or document processing
//! - Any Colossus application (colossus-legal, colossus-ai) uses this
//!   crate unchanged by supplying its own Task and Step implementations
//! - All operational parameters configurable via env vars at runtime
//! - No silent failures anywhere in the framework
//!
//! ## Rust Learning: workspace crate structure
//!
//! This crate lives in the colossus-rs workspace. It declares its public
//! API through `pub mod` and `pub use` re-exports in this file. Consumers
//! import from the crate root: `use colossus_pipeline::JobStatus` rather
//! than reaching into internal modules.

pub mod cancel;
pub mod error;
pub mod events;
pub mod progress;
pub mod recorder;
pub mod schema;
pub mod scheduler;
pub mod step;
pub mod task;
pub mod worker;

pub use error::PipelineError;
pub use recorder::{NoopStepRecorder, StepRecorder};
pub use schema::{JobControl, JobRow, JobStatus, PipelineEvent};
pub use step::{Step, StepResult};
pub use task::Task;
pub use events::EVT_JOB_SUBMITTED;
pub use scheduler::Scheduler;
pub use worker::Worker;
