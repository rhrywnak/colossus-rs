//! Generic `/api/me` handler for returning current user info.
//!
//! This handler can be mounted in any Colossus application's router:
//! ```rust,ignore
//! use colossus_auth::me_handler;
//!
//! let app = Router::new()
//!     .route("/api/me", get(me_handler));
//! ```

use axum::Json;
use serde::Serialize;

use crate::extractor::AuthUser;
use crate::permissions::Permissions;

/// Response body for the `/api/me` endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct MeResponse {
    pub username: String,
    pub email: String,
    pub display_name: String,
    pub groups: Vec<String>,
    pub permissions: Permissions,
}

/// Handler that returns the current authenticated user's info and permissions.
///
/// Axum automatically extracts `AuthUser` from request headers before calling
/// this function. If extraction fails, the error response is returned instead.
pub async fn me_handler(user: AuthUser) -> Json<MeResponse> {
    let permissions = user.permissions();

    Json(MeResponse {
        username: user.username,
        email: user.email,
        display_name: user.display_name,
        groups: user.groups,
        permissions,
    })
}
