//! Authentication error types and HTTP response conversion.
//!
//! ## Rust Learning: `#[serde(skip_serializing_if)]`
//!
//! The `skip_serializing_if` attribute tells serde to omit a field from JSON output
//! when a condition is true. For `Option<T>` fields, `Option::is_none` omits the field
//! entirely when it's `None`, producing cleaner JSON responses:
//!
//! ```text
//! // 401 (no user):  {"error": "unauthorized", "message": "..."}
//! // 403 (has user): {"error": "forbidden", "message": "...", "user": "roman", "groups": [...]}
//! ```

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// Authentication/authorization error returned as a JSON response.
///
/// - If `user` is `None` → 401 Unauthorized (not authenticated)
/// - If `user` is `Some` → 403 Forbidden (authenticated but lacking permissions)
#[derive(Debug, Clone, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,

    /// Present only for 403 errors — shows who was denied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Present only for 403 errors — shows the user's groups for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub groups: Option<Vec<String>>,
}

/// ## Rust Learning: `IntoResponse` trait
///
/// Axum requires error types used as `Rejection` in extractors to implement
/// `IntoResponse`. This converts our `AuthError` into an HTTP response with
/// the appropriate status code and a JSON body.
///
/// The status code is chosen based on whether we know who the user is:
/// - No user info → 401 (you need to authenticate)
/// - Has user info → 403 (you're authenticated but not allowed)
impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let status = if self.user.is_none() {
            StatusCode::UNAUTHORIZED
        } else {
            StatusCode::FORBIDDEN
        };

        let body = axum::Json(&self);
        (status, body).into_response()
    }
}
