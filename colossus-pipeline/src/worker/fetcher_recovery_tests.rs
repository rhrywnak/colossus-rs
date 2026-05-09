//! colossus-pipeline/src/worker/fetcher_recovery_tests.rs
//!
//! All ResolvedStepConfig tests deleted in the test audit (Batch 10):
//! they are pure struct-construct-and-assert-fields-just-set tests on a
//! POD struct (rule 1) plus a Clone/Debug derive test (rule 5).
//! Real recovery SQL behavior is exercised by recovery_tests.rs and the
//! integration suite.
