//! Step executor — runs a single step inside tokio::select! with timeout
//! and cancel_watcher racing against the step future.
//!
//! The executor is the only place in the framework that calls tokio::select!.
//! Steps themselves must never call tokio::spawn or tokio::select! internally
//! (hard guarantee G3).

// Stub — full implementation in P1-10.
