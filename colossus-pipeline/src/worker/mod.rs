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
//! - handler.rs: Maps ExecutionResult to SQL transitions and event logs
//! - heartbeat.rs: Separate tokio::spawn for last_heartbeat_at updates
//! - retry.rs: Backoff delay computation with deterministic jitter
//! - recovery.rs: Zombie and timeout detection on startup and interval
//!
//! ## Rust Learning: `tokio::sync::Semaphore` for concurrency limiting
//!
//! A Semaphore with N permits limits concurrent jobs to N. Each spawned
//! task acquires a permit (via `try_acquire_owned`) before starting and
//! releases it when the task completes (the `OwnedSemaphorePermit` is
//! dropped). `try_acquire_owned` is non-blocking — if no permits are
//! available, the Worker skips claiming and waits for the next poll tick.

pub mod config;
pub mod executor;
pub mod fetcher;
pub mod fetcher_api;
pub mod fetcher_recovery;
pub mod handler;
pub mod heartbeat;
pub mod recovery;
pub mod retry;

use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;
use tokio::sync::Semaphore;

use crate::error::PipelineError;
use crate::events;
use crate::task::Task;
use crate::worker::config::WorkerConfig;
use crate::worker::executor::execute_step;
use crate::worker::handler::handle_execution_result;
use crate::worker::recovery::recover_orphans;

/// Pipeline worker — claims and executes jobs from the queue.
///
/// Generic over `T: Task` — each application provides its own Task enum.
/// The Worker spawns up to `max_concurrent` tasks, each driving one job
/// through its steps. A `watch::Receiver<bool>` signals graceful shutdown.
///
/// ## Rust Learning: PhantomData
///
/// `Worker<T>` does not store a value of type `T` directly — the Task is
/// deserialized from the database on each claim. `PhantomData<T>` tells the
/// compiler that Worker is logically parameterized by `T` even though no
/// field uses `T`. Without it, the compiler would reject `Worker<T>` as
/// having an unused type parameter.
pub struct Worker<T: Task> {
    db: PgPool,
    context: Arc<T::Context>,
    config: WorkerConfig,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Task> Worker<T> {
    /// Create a new Worker.
    ///
    /// # Parameters
    ///
    /// - `db` — connection pool, cloned into each spawned job task.
    /// - `context` — application context (e.g., LlmProvider), shared via Arc.
    /// - `config` — operational parameters from environment variables.
    /// - `shutdown_rx` — watch channel; send `true` to trigger graceful shutdown.
    pub fn new(
        db: PgPool,
        context: Arc<T::Context>,
        config: WorkerConfig,
        shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) -> Self {
        Self {
            db,
            context,
            config,
            shutdown_rx,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Run the worker main loop until shutdown is signalled.
    ///
    /// Startup sequence: recover orphans, create PgListener, create semaphore.
    /// Main loop: select! across shutdown, recovery tick, PG notification, poll tick.
    /// Shutdown: drain in-flight jobs up to drain_timeout_secs.
    pub async fn run(mut self) -> Result<(), PipelineError> {
        tracing::info!(
            worker_id = %self.config.worker_id,
            max_concurrent = self.config.max_concurrent,
            poll_interval_secs = self.config.poll_interval_secs,
            notify_channel = %self.config.notify_channel,
            "Worker starting"
        );

        // Startup recovery sweep
        if let Err(e) = recover_orphans(&self.db, &self.config).await {
            tracing::error!(error = %e, "Startup recovery sweep failed");
        }

        // PgListener for NOTIFY wakeups
        let mut listener = sqlx::postgres::PgListener::connect_with(&self.db)
            .await
            .map_err(|e| PipelineError::Database(e.to_string()))?;
        listener
            .listen(&self.config.notify_channel)
            .await
            .map_err(|e| PipelineError::Database(e.to_string()))?;

        // Concurrency limiter
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));

        // Tick intervals
        let mut recovery_tick = tokio::time::interval(
            Duration::from_secs(self.config.recovery_interval_secs),
        );
        let mut poll_tick = tokio::time::interval(
            Duration::from_secs(self.config.poll_interval_secs),
        );

        loop {
            tokio::select! {
                biased;

                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("Shutdown signal received");
                        break;
                    }
                }

                _ = recovery_tick.tick() => {
                    if let Err(e) = recover_orphans(&self.db, &self.config).await {
                        tracing::error!(error = %e, "Recovery sweep failed");
                    }
                }

                notification = listener.recv() => {
                    match notification {
                        Ok(_) => {
                            self.try_claim_and_run(&semaphore).await;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "PgListener error, will reconnect");
                        }
                    }
                }

                _ = poll_tick.tick() => {
                    self.try_claim_and_run(&semaphore).await;
                }
            }
        }

        // Graceful drain
        tracing::info!("Draining in-flight jobs...");
        let drain_result = tokio::time::timeout(
            Duration::from_secs(self.config.drain_timeout_secs),
            semaphore.acquire_many(self.config.max_concurrent as u32),
        )
        .await;

        match drain_result {
            Ok(Ok(_)) => tracing::info!("All jobs drained cleanly"),
            Ok(Err(_)) => tracing::warn!("Semaphore closed during drain"),
            Err(_) => tracing::warn!("Drain timeout reached, exiting with jobs still running"),
        }

        Ok(())
    }

    /// Claim all available jobs up to max concurrency and spawn tasks.
    ///
    /// Loops until either no semaphore permits are available (at max
    /// concurrency) or no jobs remain in the queue. This ensures a single
    /// NOTIFY representing multiple inserts claims all available jobs
    /// without waiting for the next poll tick.
    async fn try_claim_and_run(&self, semaphore: &Arc<Semaphore>) {
        loop {
            let permit = match semaphore.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(_) => return, // At max concurrency
            };

            let job = match fetcher::claim(
                &self.db,
                &self.config.worker_id,
                None,
            )
            .await
            {
                Ok(Some(job)) => job,
                Ok(None) => {
                    drop(permit);
                    return; // No more jobs
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to claim job");
                    drop(permit);
                    return;
                }
            };

            // TODO(Phase 2): Set timeout_at from Step::DEFAULT_TIMEOUT_SECS or
            // pipeline_config.step_config after claiming. Currently steps run
            // without timeout unless timeout_at is set externally. The executor
            // handles timeout_at correctly when present — the gap is setting it.

            let db = self.db.clone();
            let context = Arc::clone(&self.context);
            let config = self.config.clone();
            let job_id = job.id;
            let step_name = job.current_step.clone();

            tokio::spawn(async move {
                // Log step started
                if let Err(e) = events::log(
                    &db,
                    job_id,
                    &step_name,
                    events::EVT_STEP_STARTED,
                    "Step execution started",
                    None,
                )
                .await
                {
                    tracing::warn!(
                        %job_id,
                        error = %e,
                        "Failed to log step_started event"
                    );
                }

                // Execute the step
                let result = execute_step::<T>(&job, &db, &context, &config).await;

                // Handle the result
                if let Err(e) =
                    handle_execution_result::<T>(&db, &job, result, &config).await
                {
                    tracing::error!(
                        %job_id,
                        error = %e,
                        "Failed to handle execution result"
                    );
                }

                // Permit dropped here — releases the semaphore slot
                drop(permit);
            });
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
