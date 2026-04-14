//! CancellationToken — allows steps to detect cancellation requests.
//!
//! The executor constructs a token per step execution. The cancel_watcher
//! inside tokio::select! polls the database control column and calls
//! trigger() when CancelRequested is detected. Steps call is_cancelled()
//! between expensive operations to exit cleanly.
//!
//! ## Rust Learning: Arc<AtomicBool> for cross-task shared state
//!
//! An AtomicBool wrapped in Arc can be safely shared between the step
//! future and the cancel_watcher future running concurrently inside
//! tokio::select!. Atomic operations are lock-free and correct under
//! concurrent access. The Arc provides shared ownership so both futures
//! can hold a clone pointing to the same underlying bool.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared cancellation signal between the executor and a running step.
///
/// The executor creates one token per step execution. The cancel_watcher
/// calls trigger() when it detects a cancel request in the database.
/// The step calls is_cancelled() periodically and returns early if true.
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new token in the non-cancelled state.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation. Called by the cancel_watcher when it detects
    /// CancelRequested in the database control column.
    pub fn trigger(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check whether cancellation has been signalled.
    /// Steps call this between expensive operations and return early if true.
    pub async fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}
