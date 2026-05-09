//! colossus-pipeline/src/step_tests.rs
//!
//! Tests for Step trait, Task trait, StepResult, and step_name_of.
//!
//! ## Rust Learning: testing traits with mock implementations
//!
//! Traits cannot be instantiated directly — they require a concrete type
//! that implements them. To test trait behavior we create minimal mock
//! types that implement the trait with the simplest possible behavior.
//! This is standard practice in Rust testing. The mock types live only
//! in the test module and are never compiled into production code.
//!
//! ## Rust Learning: #[allow(dead_code)] in tests
//!
//! Mock types defined in tests may have fields that are never read.
//! #[allow(dead_code)] suppresses the warning for test-only types.

use super::*;

// ── step_name_of ─────────────────────────────────────────────────────────────

/// Pin the contract that `step_name_of::<T>()` returns only the final
/// `::`-separated component of `std::any::type_name::<T>()`. Cases include
/// stdlib top-level types (no `::`) and nested types (must strip path).
#[test]
fn step_name_of_returns_final_component() {
    assert_eq!(step_name_of::<String>(), "String");
    assert_eq!(step_name_of::<u32>(), "u32");

    // Nested types — assert the result contains no "::" path separators.
    let hashmap_name = step_name_of::<std::collections::HashMap<String, u32>>();
    assert!(
        !hashmap_name.contains("::"),
        "step_name_of must strip path for nested types, got: {hashmap_name}",
    );

    let vec_name = step_name_of::<Vec<u8>>();
    assert!(
        !vec_name.contains("::"),
        "step_name_of must strip path for Vec, got: {vec_name}",
    );
}
