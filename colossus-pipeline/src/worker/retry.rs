//! worker/retry.rs
//!
//! Retry delay computation with deterministic jitter.
//!
//! Computes exponential-linear backoff for failed pipeline jobs.
//! The delay grows linearly with the attempt number (base = delay * attempt)
//! and is perturbed by ±15% jitter derived from the system clock's
//! nanosecond component — no `rand` crate needed.
//!
//! ## Rust Learning: nanosecond-based jitter
//!
//! `SystemTime::now().duration_since(UNIX_EPOCH).subsec_nanos()` gives the
//! sub-second nanosecond component (0–999_999_999). Modular arithmetic maps
//! this into a small signed range for jitter without pulling in a PRNG crate.

use std::cmp;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum jitter percentage applied to the base delay (±15%).
const JITTER_RANGE_PCT: i64 = 15;

/// Compute the retry delay in seconds for a given attempt.
///
/// Returns `retry_delay_secs * attempt` adjusted by ±15% jitter,
/// with a minimum of 1 second. The jitter seed is the nanosecond
/// component of the current wall-clock time.
pub fn compute_delay(retry_delay_secs: i32, attempt: i32) -> i64 {
    let base = retry_delay_secs as i64 * attempt as i64;

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as i64;

    let jitter_range = JITTER_RANGE_PCT * 2 + 1; // 31 — covers -15..=+15
    let jitter_pct = (seed % jitter_range) - JITTER_RANGE_PCT;
    let jitter = base * jitter_pct / 100;

    cmp::max(1, base + jitter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_delay_base_60_attempt_1_in_range() {
        let delay = compute_delay(60, 1);
        assert!(
            (51..=69).contains(&delay),
            "expected 51..=69, got {delay}"
        );
    }

    #[test]
    fn test_compute_delay_base_60_attempt_3_in_range() {
        let delay = compute_delay(60, 3);
        assert!(
            (153..=207).contains(&delay),
            "expected 153..=207, got {delay}"
        );
    }

    #[test]
    fn test_compute_delay_zero_base_returns_minimum() {
        let delay = compute_delay(0, 5);
        assert_eq!(delay, 1, "zero base must return minimum of 1");
    }

    #[test]
    fn test_compute_delay_jitter_is_not_constant() {
        let results: Vec<i64> = (0..20).map(|_| compute_delay(60, 1)).collect();
        let distinct = results.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            distinct >= 2,
            "expected at least 2 distinct values in 20 calls, got {distinct}: {results:?}"
        );
    }
}
