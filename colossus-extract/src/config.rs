//! Typed configuration access for flexible key-value maps.
//!
//! ## Rust Learning: Extension traits
//!
//! `ConfigAccess` is an extension trait — it adds methods to a type we
//! don't own (`HashMap<String, serde_json::Value>`). This pattern lets
//! any code that imports the trait call `.get_str()`, `.get_i64()`, etc.
//! on a HashMap without wrapping it in a newtype.
//!
//! The key design principle: configuration maps are flexible (any key can
//! exist), but reads are typed and safe (you always get the right type or
//! the default). This means adding new configuration parameters requires
//! only a YAML change and a `get_xxx()` call — no struct changes, no
//! migrations, no recompilation.

use serde_json::Value;
use std::collections::HashMap;

/// Extension trait for reading typed values from `HashMap<String, Value>`
/// with compile-time-safe defaults.
///
/// Every method follows the same contract:
/// 1. Look up the key in the map
/// 2. Attempt to convert the Value to the requested Rust type
/// 3. If the key is missing OR the value is the wrong JSON type, return the default
///
/// This means callers never need to handle None or type mismatches —
/// the default is always a valid fallback. Wrong-type values are treated
/// the same as missing keys (return default, no error). This is intentional:
/// configuration should be forgiving of unexpected values, not crash.
pub trait ConfigAccess {
    /// Read a string value, returning `default` if the key is missing
    /// or the value is not a JSON string.
    fn get_str(&self, key: &str, default: &str) -> String;

    /// Read an i64 value, returning `default` if the key is missing
    /// or the value is not a JSON number that fits in i64.
    fn get_i64(&self, key: &str, default: i64) -> i64;

    /// Read a u64 value, returning `default` if the key is missing
    /// or the value is not a JSON number that fits in u64.
    ///
    /// Unsigned reads use `serde_json::Value::as_u64()` directly, which
    /// returns None for negative numbers — preventing the silent
    /// negative→huge-positive footgun of casting `get_i64` to `u64`.
    fn get_u64(&self, key: &str, default: u64) -> u64;

    /// Read a boolean value, returning `default` if the key is missing
    /// or the value is not a JSON boolean.
    fn get_bool(&self, key: &str, default: bool) -> bool;

    /// Read an f64 value, returning `default` if the key is missing
    /// or the value is not a JSON number.
    fn get_f64(&self, key: &str, default: f64) -> f64;
}

impl ConfigAccess for HashMap<String, Value> {
    fn get_str(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(default)
            .to_string()
    }

    fn get_i64(&self, key: &str, default: i64) -> i64 {
        self.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
    }

    fn get_u64(&self, key: &str, default: u64) -> u64 {
        self.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
    }

    fn get_bool(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }
}
