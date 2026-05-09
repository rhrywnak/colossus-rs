//! Behavioral tests for the `ConfigAccess` extension trait.
//!
//! Every test verifies a specific contract — happy path, missing key,
//! wrong type, edge cases (negative→u64 default, integer→f64, empty
//! string preserved, fully empty map). Asserts specific values, never
//! "didn't crash."

use colossus_extract::ConfigAccess;
use serde_json::json;
use std::collections::HashMap;

// Helper: build a config map with known values for testing
fn sample_config() -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    map.insert("mode".to_string(), json!("structured"));
    map.insert("units_per_chunk".to_string(), json!(25));
    map.insert("timeout_secs".to_string(), json!(1800));
    map.insert("enabled".to_string(), json!(true));
    map.insert("threshold".to_string(), json!(0.85));
    map.insert("negative_number".to_string(), json!(-5));
    map.insert("float_as_int".to_string(), json!(3.7));
    map.insert("string_as_number".to_string(), json!("not_a_number"));
    map.insert("empty_string".to_string(), json!(""));
    map
}

// --- get_* tests (consolidated routing-table arms per ConfigAccess method) ---
//
// Each test exercises one method across (key, default, expected) tuples
// covering: present-and-typed, missing, wrong-type, and any
// method-specific edge cases (empty string preserved, negative-number
// safety for u64, integer→f64 conversion).

#[test]
fn test_get_str_cases() {
    let config = sample_config();
    let cases: &[(&str, &str, &str)] = &[
        ("mode", "fallback", "structured"),                // present
        ("nonexistent_key", "fallback", "fallback"),       // missing → default
        ("units_per_chunk", "default_str", "default_str"), // wrong type → default
        ("empty_string", "should_not_see_this", ""),       // empty string preserved
    ];
    for (key, default, expected) in cases {
        assert_eq!(
            config.get_str(key, default),
            *expected,
            "case: get_str({key:?}, {default:?})",
        );
    }
}

#[test]
fn test_get_i64_cases() {
    let config = sample_config();
    let cases: &[(&str, i64, i64)] = &[
        ("units_per_chunk", 10, 25), // present
        ("nonexistent_key", 42, 42), // missing → default
        ("mode", 99, 99),            // wrong type (string) → default
        ("negative_number", 0, -5),  // negative numbers preserved
    ];
    for (key, default, expected) in cases {
        assert_eq!(
            config.get_i64(key, *default),
            *expected,
            "case: get_i64({key:?}, {default})",
        );
    }
}

#[test]
fn test_get_u64_cases() {
    // Safety case: serde_json::Value::as_u64() returns None for negative
    // numbers, so -5 in YAML must NOT wrap to a huge positive u64.
    let config = sample_config();
    let cases: &[(&str, u64, u64)] = &[
        ("timeout_secs", 600, 1800),   // present
        ("nonexistent_key", 600, 600), // missing → default
        ("negative_number", 100, 100), // negative → default (no wrap)
    ];
    for (key, default, expected) in cases {
        assert_eq!(
            config.get_u64(key, *default),
            *expected,
            "case: get_u64({key:?}, {default})",
        );
    }
}

#[test]
fn test_get_bool_cases() {
    let config = sample_config();
    let cases: &[(&str, bool, bool)] = &[
        ("enabled", false, true),        // present
        ("nonexistent_key", true, true), // missing → default
        ("mode", false, false),          // wrong type (string) → default (no truthy coercion)
    ];
    for (key, default, expected) in cases {
        assert_eq!(
            config.get_bool(key, *default),
            *expected,
            "case: get_bool({key:?}, {default})",
        );
    }
}

#[test]
fn test_get_f64_cases() {
    // JSON integer 25 should be readable as f64 25.0 — Value::as_f64()
    // handles this conversion.
    let config = sample_config();
    let cases: &[(&str, f64, f64)] = &[
        ("threshold", 0.5, 0.85),       // present (float)
        ("nonexistent_key", 0.5, 0.5),  // missing → default
        ("mode", 1.0, 1.0),             // wrong type (string) → default
        ("units_per_chunk", 0.0, 25.0), // integer readable as f64
    ];
    for (key, default, expected) in cases {
        let actual = config.get_f64(key, *default);
        assert!(
            (actual - expected).abs() < f64::EPSILON,
            "case: get_f64({key:?}, {default}) — expected {expected}, got {actual}",
        );
    }
}

// --- empty map tests ---

#[test]
fn test_empty_config_returns_all_defaults() {
    let config: HashMap<String, serde_json::Value> = HashMap::new();
    assert_eq!(config.get_str("anything", "default"), "default");
    assert_eq!(config.get_i64("anything", 42), 42);
    assert_eq!(config.get_u64("anything", 600), 600);
    assert!(config.get_bool("anything", true));
    assert!((config.get_f64("anything", std::f64::consts::PI) - std::f64::consts::PI).abs() < f64::EPSILON);
}
