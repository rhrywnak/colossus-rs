//! recorder_tests.rs
//!
//! All NoopStepRecorder unit tests deleted in the test audit (Batch 10):
//! the type is a literal no-op (returns Ok(()) / Ok(0)) and the Send+Sync
//! trait-object bounds are compile-only checks the compiler enforces
//! wherever the trait is consumed. NoopStepRecorder usage is exercised
//! end-to-end by the integration tests in tests/integration.rs.
