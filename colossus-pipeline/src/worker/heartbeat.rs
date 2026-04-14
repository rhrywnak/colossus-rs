//! Heartbeat task — updates pipeline_jobs.last_heartbeat_at independently
//! of step execution.
//!
//! The heartbeat runs as a separate tokio::spawn task so that even a
//! blocking LLM call (which may not yield for minutes) cannot prevent
//! the heartbeat from updating. The recovery manager uses this timestamp
//! to detect zombie jobs. This is hard guarantee G8.

// Stub — full implementation in P1-9.
