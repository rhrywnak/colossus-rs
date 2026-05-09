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

    /// Note: this test modifies the process environment, which is global state.
    /// All AUTH_MODE cases are exercised in a single test that sets/unsets
    /// the variable sequentially — serializing the env-var manipulation
    /// removes the parallel-test risk on process-global state.
    #[test]
    fn from_env_cases() {
        // (env_value, expected) — None means var unset
        let cases: &[(Option<&str>, AuthMode)] = &[
            (Some("optional"), AuthMode::Optional),
            (Some("required"), AuthMode::Required),
            (None, AuthMode::Required), // unset → safe default
            (Some("something_else"), AuthMode::Required), // unknown → safe default
        ];

        for (value, expected) in cases {
            match value {
                Some(v) => env::set_var("AUTH_MODE", v),
                None => env::remove_var("AUTH_MODE"),
            }
            assert_eq!(AuthMode::from_env(), *expected, "case: AUTH_MODE={value:?}",);
            env::remove_var("AUTH_MODE");
        }
    }
}
