//! Zombie and timeout recovery — detects and resets stuck jobs.
//!
//! Runs on worker startup and every PIPELINE_RECOVERY_INTERVAL_SECS.
//! Zombie: a Running job whose last_heartbeat_at is older than
//! PIPELINE_ZOMBIE_THRESHOLD_SECS. This indicates the worker crashed
//! mid-step. Recovery resets the job to Ready for retry.
//! Timeout: a Running job whose timeout_at is in the past.

// Stub — full implementation in P1-9.
