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

// --- get_str tests ---

#[test]
fn test_get_str_returns_value_when_present() {
    let config = sample_config();
    let result = config.get_str("mode", "full");
    assert_eq!(
        result, "structured",
        "Must return the actual value, not the default"
    );
}

#[test]
fn test_get_str_returns_default_when_missing() {
    let config = sample_config();
    let result = config.get_str("nonexistent_key", "fallback");
    assert_eq!(result, "fallback", "Missing key must return default");
}

#[test]
fn test_get_str_returns_default_when_wrong_type() {
    // "units_per_chunk" is a number (25), not a string
    let config = sample_config();
    let result = config.get_str("units_per_chunk", "default_str");
    assert_eq!(
        result, "default_str",
        "Non-string value must return default, not a stringified number"
    );
}

#[test]
fn test_get_str_returns_empty_string_when_value_is_empty() {
    let config = sample_config();
    let result = config.get_str("empty_string", "should_not_see_this");
    assert_eq!(
        result, "",
        "Empty string IS a valid string — must not fall through to default"
    );
}

// --- get_i64 tests ---

#[test]
fn test_get_i64_returns_value_when_present() {
    let config = sample_config();
    let result = config.get_i64("units_per_chunk", 10);
    assert_eq!(result, 25, "Must return the actual value, not the default");
}

#[test]
fn test_get_i64_returns_default_when_missing() {
    let config = sample_config();
    let result = config.get_i64("nonexistent_key", 42);
    assert_eq!(result, 42, "Missing key must return default");
}

#[test]
fn test_get_i64_returns_default_when_wrong_type() {
    // "mode" is a string ("structured"), not a number
    let config = sample_config();
    let result = config.get_i64("mode", 99);
    assert_eq!(
        result, 99,
        "String value must return default, not panic or 0"
    );
}

#[test]
fn test_get_i64_handles_negative_numbers() {
    let config = sample_config();
    let result = config.get_i64("negative_number", 0);
    assert_eq!(result, -5, "Negative numbers must be returned correctly");
}

// --- get_u64 tests ---

#[test]
fn test_get_u64_returns_value_when_present() {
    let config = sample_config();
    let result = config.get_u64("timeout_secs", 600);
    assert_eq!(
        result, 1800,
        "Must return the actual value, not the default"
    );
}

#[test]
fn test_get_u64_returns_default_when_missing() {
    let config = sample_config();
    let result = config.get_u64("nonexistent_key", 600);
    assert_eq!(result, 600, "Missing key must return default");
}

#[test]
fn test_get_u64_returns_default_for_negative_number() {
    // This is the safety case: -5 in YAML should NOT become a huge u64.
    // serde_json::Value::as_u64() returns None for negative numbers.
    let config = sample_config();
    let result = config.get_u64("negative_number", 100);
    assert_eq!(
        result, 100,
        "Negative number must return default, not wrap to a huge positive value"
    );
}

// --- get_bool tests ---

#[test]
fn test_get_bool_returns_value_when_present() {
    let config = sample_config();
    let result = config.get_bool("enabled", false);
    assert!(result, "Must return true, not the false default");
}

#[test]
fn test_get_bool_returns_default_when_missing() {
    let config = sample_config();
    let result = config.get_bool("nonexistent_key", true);
    assert!(result, "Missing key must return default");
}

#[test]
fn test_get_bool_returns_default_when_wrong_type() {
    // "mode" is a string, not a boolean
    let config = sample_config();
    let result = config.get_bool("mode", false);
    assert!(
        !result,
        "String value must return default, not truthy coercion"
    );
}

// --- get_f64 tests ---

#[test]
fn test_get_f64_returns_value_when_present() {
    let config = sample_config();
    let result = config.get_f64("threshold", 0.5);
    assert!(
        (result - 0.85).abs() < f64::EPSILON,
        "Must return 0.85, not the 0.5 default. Got: {result}"
    );
}

#[test]
fn test_get_f64_returns_default_when_missing() {
    let config = sample_config();
    let result = config.get_f64("nonexistent_key", 0.5);
    assert!(
        (result - 0.5).abs() < f64::EPSILON,
        "Missing key must return default"
    );
}

#[test]
fn test_get_f64_returns_default_when_wrong_type() {
    // "mode" is a string, not a number
    let config = sample_config();
    let result = config.get_f64("mode", 1.0);
    assert!(
        (result - 1.0).abs() < f64::EPSILON,
        "String value must return default, not panic or 0.0"
    );
}

#[test]
fn test_get_f64_reads_integer_as_f64() {
    // JSON integer 25 should be readable as f64 25.0
    // serde_json::Value::as_f64() handles this conversion
    let config = sample_config();
    let result = config.get_f64("units_per_chunk", 0.0);
    assert!(
        (result - 25.0).abs() < f64::EPSILON,
        "Integer value must be readable as f64. Got: {result}"
    );
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
