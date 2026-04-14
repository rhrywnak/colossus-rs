//! Scheduler — public API for submitting and managing pipeline jobs.
//!
//! The Scheduler is the only entry point for job submission. It enforces
//! the deduplication invariant (one active job per job_type+job_key),
//! generates UUID v7 job IDs (time-ordered for efficient index scans),
//! and provides status query, cancel, resume, and delete operations.
//!
//! Application code calls Scheduler methods directly. The Worker is
//! internal — application code does not call Worker methods.

// Stub — full implementation in P1-5 through P1-7.
// This file exists so lib.rs compiles cleanly as a module declaration.
