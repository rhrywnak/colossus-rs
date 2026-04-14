//! Worker — claims jobs from the queue and executes their steps.
//!
//! The Worker runs as a long-lived tokio task spawned at application startup.
//! It polls pipeline_jobs for ready work, claims jobs with FOR UPDATE SKIP LOCKED
//! (preventing double-execution), spawns a task per job up to max_concurrent,
//! and drives each job through its steps until completion, failure, or cancellation.
//!
//! Internal submodules handle distinct concerns:
//! - config.rs: WorkerConfig and WorkerConfig::from_env()
//! - fetcher.rs: All SQL state transitions (claim, advance, complete, fail, etc.)
//! - executor.rs: tokio::select! across execute/timeout/cancel_watcher
//! - heartbeat.rs: Separate tokio::spawn for last_heartbeat_at updates
//! - retry.rs: Backoff delay computation with deterministic jitter
//! - recovery.rs: Zombie and timeout detection on startup and interval

pub mod config;
pub mod executor;
pub mod fetcher;
pub mod heartbeat;
pub mod recovery;
pub mod retry;
