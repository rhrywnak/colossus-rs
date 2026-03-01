//! Permission checking methods and guard functions.
//!
//! ## Rust Learning: The `?` operator with guard functions
//!
//! The guard functions (`require_edit`, `require_ai`, `require_admin`) return
//! `Result<(), AuthError>`. In a handler, you use the `?` operator to short-circuit:
//!
//! ```rust,ignore
//! async fn edit_document(user: AuthUser, body: Json<EditRequest>) -> Result<..., AuthError> {
//!     require_edit(&user)?;  // Returns 403 early if user lacks permission
//!     // ... rest of handler only runs if permission check passed
//! }
//! ```
//!
//! The `?` operator unwraps `Ok(())` (continuing execution) or returns `Err(AuthError)`
//! (which Axum converts to a 403 JSON response via our `IntoResponse` impl).

use serde::Serialize;

use crate::error::AuthError;
use crate::extractor::AuthUser;
use crate::{GROUP_ADMIN, GROUP_AI_USER, GROUP_LEGAL_EDITOR, GROUP_LEGAL_VIEWER};

/// Summary of a user's permissions, suitable for sending to frontends.
#[derive(Debug, Clone, Serialize)]
pub struct Permissions {
    pub can_read: bool,
    pub can_edit: bool,
    pub can_use_ai: bool,
    pub is_admin: bool,
}

// --- Permission methods on AuthUser ---

impl AuthUser {
    /// Returns true if the user is in the admin group.
    pub fn is_admin(&self) -> bool {
        self.has_group(GROUP_ADMIN)
    }

    /// Returns true if the user can read data (admin, editor, or viewer).
    pub fn can_read(&self) -> bool {
        self.is_admin()
            || self.has_group(GROUP_LEGAL_EDITOR)
            || self.has_group(GROUP_LEGAL_VIEWER)
    }

    /// Returns true if the user can edit data (admin or editor).
    pub fn can_edit(&self) -> bool {
        self.is_admin() || self.has_group(GROUP_LEGAL_EDITOR)
    }

    /// Returns true if the user can use AI features (admin or ai_user).
    pub fn can_use_ai(&self) -> bool {
        self.is_admin() || self.has_group(GROUP_AI_USER)
    }

    /// Checks whether the user belongs to a specific group.
    pub fn has_group(&self, group: &str) -> bool {
        self.groups.iter().any(|g| g == group)
    }

    /// Builds a `Permissions` summary for this user.
    pub fn permissions(&self) -> Permissions {
        Permissions {
            can_read: self.can_read(),
            can_edit: self.can_edit(),
            can_use_ai: self.can_use_ai(),
            is_admin: self.is_admin(),
        }
    }
}

// --- Guard functions ---

/// Returns `Err(AuthError)` with 403 if the user cannot edit.
pub fn require_edit(user: &AuthUser) -> Result<(), AuthError> {
    if user.can_edit() {
        Ok(())
    } else {
        Err(forbidden(user, "Edit permission required"))
    }
}

/// Returns `Err(AuthError)` with 403 if the user cannot use AI features.
pub fn require_ai(user: &AuthUser) -> Result<(), AuthError> {
    if user.can_use_ai() {
        Ok(())
    } else {
        Err(forbidden(user, "AI permission required"))
    }
}

/// Returns `Err(AuthError)` with 403 if the user is not an admin.
pub fn require_admin(user: &AuthUser) -> Result<(), AuthError> {
    if user.is_admin() {
        Ok(())
    } else {
        Err(forbidden(user, "Admin permission required"))
    }
}

/// Helper: builds a 403 AuthError with the user's identity for debugging.
fn forbidden(user: &AuthUser, message: &str) -> AuthError {
    AuthError {
        error: "forbidden".to_string(),
        message: message.to_string(),
        user: Some(user.username.clone()),
        groups: Some(user.groups.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a test user with the given groups.
    fn user_with_groups(groups: &[&str]) -> AuthUser {
        AuthUser {
            username: "testuser".to_string(),
            email: "test@example.com".to_string(),
            display_name: "Test User".to_string(),
            groups: groups.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_admin_has_all_permissions() {
        let user = user_with_groups(&["admin"]);
        assert!(user.is_admin());
        assert!(user.can_read());
        assert!(user.can_edit());
        assert!(user.can_use_ai());
    }

    #[test]
    fn test_editor_can_read_and_edit() {
        let user = user_with_groups(&["legal_editor"]);
        assert!(!user.is_admin());
        assert!(user.can_read());
        assert!(user.can_edit());
        assert!(!user.can_use_ai());
    }

    #[test]
    fn test_viewer_can_only_read() {
        let user = user_with_groups(&["legal_viewer"]);
        assert!(!user.is_admin());
        assert!(user.can_read());
        assert!(!user.can_edit());
        assert!(!user.can_use_ai());
    }

    #[test]
    fn test_ai_user_can_use_ai() {
        let user = user_with_groups(&["ai_user"]);
        assert!(user.can_use_ai());
        assert!(!user.can_edit());
        assert!(!user.is_admin());
    }

    #[test]
    fn test_require_edit_allows_editor() {
        let user = user_with_groups(&["legal_editor"]);
        assert!(require_edit(&user).is_ok());
    }

    #[test]
    fn test_require_edit_rejects_viewer() {
        let user = user_with_groups(&["legal_viewer"]);
        let err = require_edit(&user).unwrap_err();
        assert_eq!(err.error, "forbidden");
        assert_eq!(err.user, Some("testuser".to_string()));
    }

    #[test]
    fn test_require_ai_rejects_editor() {
        let user = user_with_groups(&["legal_editor"]);
        let err = require_ai(&user).unwrap_err();
        assert_eq!(err.error, "forbidden");
    }

    #[test]
    fn test_require_admin_rejects_non_admin() {
        let user = user_with_groups(&["legal_editor", "ai_user"]);
        let err = require_admin(&user).unwrap_err();
        assert_eq!(err.error, "forbidden");
    }

    #[test]
    fn test_require_admin_allows_admin() {
        let user = user_with_groups(&["admin"]);
        assert!(require_admin(&user).is_ok());
    }

    #[test]
    fn test_permissions_struct() {
        let user = user_with_groups(&["legal_editor", "ai_user"]);
        let perms = user.permissions();
        assert!(perms.can_read);
        assert!(perms.can_edit);
        assert!(perms.can_use_ai);
        assert!(!perms.is_admin);
    }

    #[test]
    fn test_no_groups_no_permissions() {
        let user = user_with_groups(&[]);
        assert!(!user.can_read());
        assert!(!user.can_edit());
        assert!(!user.can_use_ai());
        assert!(!user.is_admin());
    }
}
