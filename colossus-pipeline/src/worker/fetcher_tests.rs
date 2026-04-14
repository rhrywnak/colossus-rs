//! colossus-pipeline/src/worker/fetcher_tests.rs
//!
//! Tests for fetcher job-lifecycle SQL transitions.
//!
//! ## Rust Learning: testing database code without a live database
//!
//! The fetcher functions all require a live PostgreSQL database with the
//! pipeline_jobs table present. True unit tests of SQL are not possible
//! without that infrastructure. Instead we test the things we can verify
//! without a database: constants and compile-time type verification.
//! Integration tests against a real database belong in Phase 6
//! end-to-end validation.

use super::*;

#[test]
fn cancelled_by_user_constant_value() {
    assert_eq!(CANCELLED_BY_USER, "Cancelled by user");
}

#[test]
fn cancelled_by_user_is_not_empty() {
    assert!(!CANCELLED_BY_USER.is_empty());
}
