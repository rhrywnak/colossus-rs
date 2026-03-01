//! Axum extractor that builds an `AuthUser` from Authentik headers.
//!
//! ## Rust Learning: `FromRequestParts` vs `FromRequest`
//!
//! Axum provides two extractor traits:
//! - `FromRequestParts<S>` — accesses headers, query params, path params, extensions
//!   (everything except the body). Multiple extractors can run since they share `&mut Parts`.
//! - `FromRequest<S>` — consumes the entire request including body. Only one per handler.
//!
//! Since we only read headers, `FromRequestParts` is the right choice. This also means
//! `AuthUser` can be combined with `Json<T>` or other body extractors in the same handler.
//!
//! ## Rust Learning: Generic type parameters and trait bounds
//!
//! The `impl<S> FromRequestParts<S> for AuthUser where S: Send + Sync` syntax means:
//! - `<S>` — this implementation works for ANY type S (not just one specific AppState)
//! - `where S: Send + Sync` — S must be thread-safe (required by Axum's async runtime)
//! - This makes the crate reusable: colossus-legal, colossus-ai, etc. each have their own
//!   AppState, but they can all use this same `AuthUser` extractor.

use async_trait::async_trait;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use serde::Serialize;
use tracing::{debug, warn};

use crate::error::AuthError;
use crate::mode::AuthMode;

// --- Header constants (private to this module) ---

const HEADER_USERNAME: &str = "x-authentik-username";
const HEADER_EMAIL: &str = "x-authentik-email";
const HEADER_GROUPS: &str = "x-authentik-groups";
const HEADER_NAME: &str = "x-authentik-name";
const GROUPS_SEPARATOR: char = '|';

/// Authenticated user extracted from Authentik/Traefik headers.
///
/// Axum extracts this automatically in handler parameters:
/// ```rust,ignore
/// async fn my_handler(user: AuthUser) -> impl IntoResponse {
///     format!("Hello, {}!", user.username)
/// }
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct AuthUser {
    pub username: String,
    pub email: String,
    pub display_name: String,
    pub groups: Vec<String>,
}

impl AuthUser {
    /// Creates an anonymous user with admin privileges.
    ///
    /// Used when `AuthMode::Optional` is active and no auth headers are present.
    /// Grants full admin access for local development without Authentik/Traefik.
    pub(crate) fn anonymous() -> Self {
        Self {
            username: "anonymous".to_string(),
            email: String::new(),
            display_name: "Anonymous".to_string(),
            groups: vec!["admin".to_string()],
        }
    }
}

/// ## Rust Learning: `impl<S> FromRequestParts<S> for AuthUser`
///
/// This is where the generic state parameter matters. By using `S` instead of a
/// concrete type like `AppState`, any Colossus application can use `AuthUser` as
/// an extractor regardless of what state type it passes to `Router::with_state()`.
///
/// The `Send + Sync` bounds are required because Axum handlers run on a multi-threaded
/// async runtime (Tokio), so the state must be safe to share across threads.
///
/// ## Rust Learning: `#[async_trait]`
///
/// Axum 0.7 defines `FromRequestParts` using the `#[async_trait]` macro, which
/// desugars `async fn` into a boxed future with specific lifetime parameters.
/// Our impl must also use `#[async_trait]` so the signatures match.
#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        // Try to read the username header — this is our primary signal
        // that Authentik/Traefik has authenticated the request.
        let username = extract_header(&parts.headers, HEADER_USERNAME);

        match username {
            Some(username) => {
                let email = extract_header(&parts.headers, HEADER_EMAIL)
                    .unwrap_or_default();
                let display_name = extract_header(&parts.headers, HEADER_NAME)
                    .unwrap_or_else(|| username.clone());
                let groups = parse_groups(&parts.headers);

                debug!(
                    username = %username,
                    groups = ?groups,
                    "Authenticated user from headers"
                );

                Ok(AuthUser {
                    username,
                    email,
                    display_name,
                    groups,
                })
            }
            None => {
                // No auth headers — check if auth is optional
                match AuthMode::from_env() {
                    AuthMode::Optional => {
                        debug!("No auth headers, AUTH_MODE=optional → anonymous user");
                        Ok(AuthUser::anonymous())
                    }
                    AuthMode::Required => {
                        warn!("No auth headers and AUTH_MODE=required → 401");
                        Err(AuthError {
                            error: "unauthorized".to_string(),
                            message: "Authentication required".to_string(),
                            user: None,
                            groups: None,
                        })
                    }
                }
            }
        }
    }
}

/// Extracts a single header value as a String.
fn extract_header(
    headers: &axum::http::HeaderMap,
    name: &str,
) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// ## Rust Learning: Iterator chains
///
/// This function demonstrates a common Rust pattern — transforming data through
/// a chain of iterator methods:
///
/// ```text
/// "admin|editor|viewer"
///   .split('|')           → ["admin", "editor", "viewer"]  (split by separator)
///   .map(|s| s.trim())    → ["admin", "editor", "viewer"]  (trim whitespace)
///   .filter(|s| !s.is_empty()) → skip any empty segments
///   .map(|s| s.to_string()) → convert &str to owned String
///   .collect::<Vec<_>>()  → gather into a Vec<String>
/// ```
///
/// Each step is lazy — nothing happens until `.collect()` drives the chain.
fn parse_groups(headers: &axum::http::HeaderMap) -> Vec<String> {
    extract_header(headers, HEADER_GROUPS)
        .map(|groups_str| {
            groups_str
                .split(GROUPS_SEPARATOR)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};

    /// Helper: build a HeaderMap with the given authentik headers.
    fn make_headers(
        username: Option<&str>,
        email: Option<&str>,
        name: Option<&str>,
        groups: Option<&str>,
    ) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(v) = username {
            headers.insert(HEADER_USERNAME, HeaderValue::from_str(v).unwrap());
        }
        if let Some(v) = email {
            headers.insert(HEADER_EMAIL, HeaderValue::from_str(v).unwrap());
        }
        if let Some(v) = name {
            headers.insert(HEADER_NAME, HeaderValue::from_str(v).unwrap());
        }
        if let Some(v) = groups {
            headers.insert(HEADER_GROUPS, HeaderValue::from_str(v).unwrap());
        }
        headers
    }

    #[test]
    fn test_extract_header_present() {
        let headers = make_headers(Some("roman"), None, None, None);
        assert_eq!(
            extract_header(&headers, HEADER_USERNAME),
            Some("roman".to_string())
        );
    }

    #[test]
    fn test_extract_header_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_header(&headers, HEADER_USERNAME), None);
    }

    #[test]
    fn test_parse_groups_pipe_separated() {
        let headers = make_headers(None, None, None, Some("admin|editor|viewer"));
        let groups = parse_groups(&headers);
        assert_eq!(groups, vec!["admin", "editor", "viewer"]);
    }

    #[test]
    fn test_parse_groups_single() {
        let headers = make_headers(None, None, None, Some("admin"));
        let groups = parse_groups(&headers);
        assert_eq!(groups, vec!["admin"]);
    }

    #[test]
    fn test_parse_groups_empty_string() {
        let headers = make_headers(None, None, None, Some(""));
        let groups = parse_groups(&headers);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_parse_groups_missing_header() {
        let headers = HeaderMap::new();
        let groups = parse_groups(&headers);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_parse_groups_with_whitespace() {
        let headers = make_headers(None, None, None, Some("admin | editor | viewer"));
        let groups = parse_groups(&headers);
        assert_eq!(groups, vec!["admin", "editor", "viewer"]);
    }

    // Integration tests using the full FromRequestParts flow require an async
    // runtime and constructing axum request Parts. These are tested via the
    // helper functions above which cover the core logic.

    #[test]
    fn test_anonymous_user() {
        let user = AuthUser::anonymous();
        assert_eq!(user.username, "anonymous");
        assert_eq!(user.groups, vec!["admin"]);
    }

    #[test]
    fn test_display_name_falls_back_to_username() {
        let headers = make_headers(Some("roman"), None, None, Some("admin"));
        let username = extract_header(&headers, HEADER_USERNAME).unwrap();
        let email = extract_header(&headers, HEADER_EMAIL).unwrap_or_default();
        let display_name = extract_header(&headers, HEADER_NAME)
            .unwrap_or_else(|| username.clone());
        assert_eq!(email, "");
        assert_eq!(display_name, "roman");
    }
}
