//! colossus-pipeline/src/worker/fetcher_recovery_tests.rs
//!
//! Tests for recovery SQL transitions and ResolvedStepConfig.
//!
//! ## Rust Learning: testing database code without a live database
//!
//! The recovery functions require a live PostgreSQL database with the
//! pipeline_jobs table present. Instead we test the non-SQL logic:
//! ResolvedStepConfig construction, Clone/Debug derives, and value ranges.
//! Integration tests against a real database belong in Phase 6.

use super::*;

#[test]
fn resolved_step_config_holds_correct_defaults() {
    let cfg = ResolvedStepConfig {
        retry_limit: 3,
        retry_delay_secs: 60,
        timeout_secs: Some(900),
    };
    assert_eq!(cfg.retry_limit, 3);
    assert_eq!(cfg.retry_delay_secs, 60);
    assert_eq!(cfg.timeout_secs, Some(900));
}

#[test]
fn resolved_step_config_none_timeout_is_valid() {
    let cfg = ResolvedStepConfig {
        retry_limit: 0,
        retry_delay_secs: 0,
        timeout_secs: None,
    };
    assert!(cfg.timeout_secs.is_none());
    assert_eq!(cfg.retry_limit, 0);
}

#[test]
fn resolved_step_config_clone_is_independent() {
    // Verify Clone derive works correctly — changing the clone
    // does not affect the original. This also exercises Debug derive
    // via the format! call.
    let cfg = ResolvedStepConfig {
        retry_limit: 5,
        retry_delay_secs: 30,
        timeout_secs: Some(600),
    };
    let mut cloned = cfg.clone();
    cloned.retry_limit = 10;
    assert_eq!(cfg.retry_limit, 5);
    assert_eq!(cloned.retry_limit, 10);
    // Exercise Debug derive
    let debug_str = format!("{:?}", cfg);
    assert!(debug_str.contains("ResolvedStepConfig"));
}

#[test]
fn resolved_step_config_large_timeout_value() {
    // Verify u64 timeout handles large values (e.g. 24 hours in seconds).
    let cfg = ResolvedStepConfig {
        retry_limit: 1,
        retry_delay_secs: 120,
        timeout_secs: Some(86400),
    };
    assert_eq!(cfg.timeout_secs, Some(86400));
}
