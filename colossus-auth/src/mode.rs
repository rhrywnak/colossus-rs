//! Authentication mode configuration via environment variable.
//!
//! ## Rust Learning: `std::env::var()` with pattern matching
//!
//! `std::env::var("NAME")` returns `Result<String, VarError>`. We use `ok()`
//! to convert it to `Option<String>`, then pattern match. This is a common
//! Rust idiom for reading optional configuration with safe defaults.

use std::env;

/// Controls whether authentication is enforced.
///
/// - `Required` (default) — requests without auth headers get 401
/// - `Optional` — requests without auth headers get an anonymous admin user
///   (useful for local development without Authentik/Traefik)
#[derive(Debug, Clone, PartialEq)]
pub enum AuthMode {
    Required,
    Optional,
}

impl AuthMode {
    /// Reads the `AUTH_MODE` environment variable.
    ///
    /// Returns `Optional` only if AUTH_MODE is exactly `"optional"`.
    /// Any other value (including unset) returns `Required` — the safe default.
    pub fn from_env() -> Self {
        match env::var("AUTH_MODE").ok().as_deref() {
            Some("optional") => AuthMode::Optional,
            _ => AuthMode::Required,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Note: these tests modify the process environment, which is global state.
    /// They run sequentially within this module since `cargo test` runs tests
    /// in the same test binary. For safety, each test sets and then removes
    /// the variable.

    #[test]
    fn test_auth_mode_optional() {
        env::set_var("AUTH_MODE", "optional");
        assert_eq!(AuthMode::from_env(), AuthMode::Optional);
        env::remove_var("AUTH_MODE");
    }

    #[test]
    fn test_auth_mode_required() {
        env::set_var("AUTH_MODE", "required");
        assert_eq!(AuthMode::from_env(), AuthMode::Required);
        env::remove_var("AUTH_MODE");
    }

    #[test]
    fn test_auth_mode_unset_defaults_to_required() {
        env::remove_var("AUTH_MODE");
        assert_eq!(AuthMode::from_env(), AuthMode::Required);
    }

    #[test]
    fn test_auth_mode_unknown_value_defaults_to_required() {
        env::set_var("AUTH_MODE", "something_else");
        assert_eq!(AuthMode::from_env(), AuthMode::Required);
        env::remove_var("AUTH_MODE");
    }
}
