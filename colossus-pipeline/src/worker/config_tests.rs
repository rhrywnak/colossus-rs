//! colossus-pipeline/src/worker/config_tests.rs
//!
//! Tests for WorkerConfig::from_env().
//!
//! ## Rust Learning: testing environment variables
//!
//! Environment variables are process-global state. Tests that set env vars
//! can interfere with each other when run in parallel (Rust runs tests
//! concurrently by default). The safe pattern is to use std::env::set_var()
//! for the specific var under test and restore or remove it after, or to
//! run env-var tests with -- --test-threads=1. Here we test each var in
//! isolation by setting only the var being tested and using unique values
//! unlikely to conflict with real environment state.
//!
//! Note: std::env::set_var is unsafe in Rust 1.80+ in multithreaded contexts.
//! We use it here only in tests that run single-threaded by design.

use super::*;

#[test]
fn from_env_defaults_when_no_env_vars_set() {
    // Remove all PIPELINE_ vars that might be set in the environment.
    // This test verifies compiled-in defaults, not env var parsing.
    let vars = [
        "PIPELINE_MAX_CONCURRENT",
        "PIPELINE_POLL_INTERVAL_SECS",
        "PIPELINE_DRAIN_TIMEOUT_SECS",
        "PIPELINE_HEARTBEAT_INTERVAL_SECS",
        "PIPELINE_ZOMBIE_THRESHOLD_SECS",
        "PIPELINE_RECOVERY_INTERVAL_SECS",
        "PIPELINE_RECOVERY_WAKEUP_SECS",
        "PIPELINE_CANCEL_POLL_INTERVAL_SECS",
        "PIPELINE_NOTIFY_CHANNEL",
        "PIPELINE_WORKER_ID",
    ];
    for var in &vars {
        unsafe { std::env::remove_var(var) };
    }

    let config = WorkerConfig::from_env();

    assert_eq!(config.max_concurrent, 4);
    assert_eq!(config.poll_interval_secs, 2);
    assert_eq!(config.drain_timeout_secs, 300);
    assert_eq!(config.heartbeat_interval_secs, 30);
    assert_eq!(config.zombie_threshold_secs, 120);
    assert_eq!(config.recovery_interval_secs, 60);
    assert_eq!(config.recovery_wakeup_secs, 5);
    assert_eq!(config.cancel_poll_interval_secs, 5);
    assert_eq!(config.notify_channel, "pipeline_jobs_changed");
}

#[test]
fn from_env_reads_max_concurrent_override() {
    unsafe { std::env::set_var("PIPELINE_MAX_CONCURRENT", "8") };
    let config = WorkerConfig::from_env();
    unsafe { std::env::remove_var("PIPELINE_MAX_CONCURRENT") };
    assert_eq!(config.max_concurrent, 8);
}

#[test]
fn from_env_reads_notify_channel_override() {
    unsafe { std::env::set_var("PIPELINE_NOTIFY_CHANNEL", "my_custom_channel") };
    let config = WorkerConfig::from_env();
    unsafe { std::env::remove_var("PIPELINE_NOTIFY_CHANNEL") };
    assert_eq!(config.notify_channel, "my_custom_channel");
}

#[test]
fn from_env_worker_id_is_non_empty_without_env_var() {
    unsafe { std::env::remove_var("PIPELINE_WORKER_ID") };
    let config = WorkerConfig::from_env();
    assert!(!config.worker_id.is_empty());
}

#[test]
fn from_env_two_calls_without_worker_id_env_produce_different_ids() {
    unsafe { std::env::remove_var("PIPELINE_WORKER_ID") };
    let config1 = WorkerConfig::from_env();
    let config2 = WorkerConfig::from_env();
    assert_ne!(
        config1.worker_id, config2.worker_id,
        "Two calls without PIPELINE_WORKER_ID must produce different worker IDs"
    );
}

#[test]
fn from_env_reads_worker_id_override() {
    unsafe { std::env::set_var("PIPELINE_WORKER_ID", "test-worker-123") };
    let config = WorkerConfig::from_env();
    unsafe { std::env::remove_var("PIPELINE_WORKER_ID") };
    assert_eq!(config.worker_id, "test-worker-123");
}

#[test]
fn malformed_env_var_uses_default() {
    unsafe { std::env::set_var("PIPELINE_MAX_CONCURRENT", "not-a-number") };
    let config = WorkerConfig::from_env();
    unsafe { std::env::remove_var("PIPELINE_MAX_CONCURRENT") };
    assert_eq!(config.max_concurrent, 4, "Malformed value must fall back to default");
}
