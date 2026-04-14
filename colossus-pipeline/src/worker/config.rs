//! colossus-pipeline/src/worker/config.rs
//!
//! WorkerConfig — all operational parameters for the pipeline worker.
//!
//! Every parameter reads from an environment variable at startup with a
//! compiled-in default. No parameter requires a recompile to change —
//! this is hard guarantee G9 from the design spec.
//!
//! ## Rust Learning: environment variable parsing pattern
//!
//! std::env::var() returns Result<String, VarError>. The pattern
//! .ok().and_then(|v| v.parse().ok()).unwrap_or(default) reads the var,
//! converts the error to None with .ok(), parses the string value (also
//! converting parse errors to None), and falls back to the default if
//! either step fails. This is the correct pattern for optional env vars
//! with typed defaults — it never panics and silently uses the default
//! for missing or malformed values.

use uuid::Uuid;

/// All operational parameters for the pipeline worker.
///
/// Constructed once at application startup via WorkerConfig::from_env().
/// All fields are immutable after construction — runtime tuning requires
/// a container restart, which is intentional (forces deliberate changes).
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Maximum number of jobs executing concurrently on this worker.
    /// Env var: PIPELINE_MAX_CONCURRENT, default: 4
    pub max_concurrent: usize,

    /// How often the worker polls for new jobs when no LISTEN notification arrives.
    /// Env var: PIPELINE_POLL_INTERVAL_SECS, default: 2
    pub poll_interval_secs: u64,

    /// How long to wait for in-flight jobs to complete on graceful shutdown.
    /// After this timeout, the worker exits regardless of job state.
    /// Env var: PIPELINE_DRAIN_TIMEOUT_SECS, default: 300
    pub drain_timeout_secs: u64,

    /// How often the heartbeat task updates last_heartbeat_at.
    /// Must be less than zombie_threshold_secs / 2 to avoid false positives.
    /// Env var: PIPELINE_HEARTBEAT_INTERVAL_SECS, default: 30
    pub heartbeat_interval_secs: u64,

    /// Jobs running with last_heartbeat_at older than this are considered zombies.
    /// Env var: PIPELINE_ZOMBIE_THRESHOLD_SECS, default: 120
    pub zombie_threshold_secs: u64,

    /// How often the recovery task scans for zombies and timed-out jobs.
    /// Env var: PIPELINE_RECOVERY_INTERVAL_SECS, default: 60
    pub recovery_interval_secs: u64,

    /// How far in the future to schedule wakeup_at when recovering a zombie.
    /// Small value so recovered jobs run quickly after recovery.
    /// Env var: PIPELINE_RECOVERY_WAKEUP_SECS, default: 5
    pub recovery_wakeup_secs: u64,

    /// How often the cancel_watcher polls the database control column.
    /// Env var: PIPELINE_CANCEL_POLL_INTERVAL_SECS, default: 5
    pub cancel_poll_interval_secs: u64,

    /// PostgreSQL LISTEN/NOTIFY channel name for job wake events.
    /// Env var: PIPELINE_NOTIFY_CHANNEL, default: "pipeline_jobs_changed"
    pub notify_channel: String,

    /// Unique identifier for this worker instance.
    /// Generated fresh at startup — not process ID (which is ambiguous after restart).
    /// Env var: PIPELINE_WORKER_ID, default: new UUID v4 generated at startup
    pub worker_id: String,
}

impl WorkerConfig {
    /// Construct WorkerConfig from environment variables.
    ///
    /// All parameters have defaults — this function never fails.
    /// Missing or malformed env vars silently use the default value.
    /// Two calls with no env vars set produce different worker_ids
    /// because UUID v4 is randomly generated each call.
    pub fn from_env() -> Self {
        Self {
            max_concurrent: env_usize("PIPELINE_MAX_CONCURRENT", 4),
            poll_interval_secs: env_u64("PIPELINE_POLL_INTERVAL_SECS", 2),
            drain_timeout_secs: env_u64("PIPELINE_DRAIN_TIMEOUT_SECS", 300),
            heartbeat_interval_secs: env_u64("PIPELINE_HEARTBEAT_INTERVAL_SECS", 30),
            zombie_threshold_secs: env_u64("PIPELINE_ZOMBIE_THRESHOLD_SECS", 120),
            recovery_interval_secs: env_u64("PIPELINE_RECOVERY_INTERVAL_SECS", 60),
            recovery_wakeup_secs: env_u64("PIPELINE_RECOVERY_WAKEUP_SECS", 5),
            cancel_poll_interval_secs: env_u64("PIPELINE_CANCEL_POLL_INTERVAL_SECS", 5),
            notify_channel: std::env::var("PIPELINE_NOTIFY_CHANNEL")
                .unwrap_or_else(|_| "pipeline_jobs_changed".to_string()),
            worker_id: std::env::var("PIPELINE_WORKER_ID")
                .unwrap_or_else(|_| Uuid::new_v4().to_string()),
        }
    }
}

/// Read a u64 environment variable with a default fallback.
///
/// Returns the parsed value if the env var exists and parses as u64,
/// otherwise returns the provided default.
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Read a usize environment variable with a default fallback.
///
/// Returns the parsed value if the env var exists and parses as usize,
/// otherwise returns the provided default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
