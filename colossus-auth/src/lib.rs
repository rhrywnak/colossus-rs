//! # colossus-auth
//!
//! Authentik + Axum authentication integration for Colossus applications.
//!
//! Reads `X-authentik-*` headers injected by Traefik ForwardAuth, constructs
//! an [`AuthUser`], and provides permission guard functions.
//!
//! ## Rust Learning: `pub use` re-exports
//!
//! A library crate's `lib.rs` defines its public API. By using `pub use`, we
//! re-export items from internal modules so consumers can write:
//! ```rust,ignore
//! use colossus_auth::{AuthUser, AuthError, require_edit};
//! ```
//! instead of reaching into internal modules:
//! ```rust,ignore
//! use colossus_auth::extractor::AuthUser;  // works but less ergonomic
//! ```
//!
//! ## Rust Learning: Cargo workspaces
//!
//! This crate lives in a Cargo workspace (`colossus-rs`). The workspace root
//! `Cargo.toml` uses `[workspace.dependencies]` to define shared dependency
//! versions. Each member crate references them with `{ workspace = true }`,
//! ensuring all crates use the same versions. This prevents version conflicts
//! and simplifies upgrades.

mod error;
mod extractor;
mod handler;
mod mode;
mod permissions;

// --- Public API re-exports ---

pub use error::AuthError;
pub use extractor::AuthUser;
pub use handler::{me_handler, MeResponse};
pub use mode::AuthMode;
pub use permissions::{require_admin, require_ai, require_edit, Permissions};

// --- Group name constants ---

/// Authentik group name for administrators (full access).
pub const GROUP_ADMIN: &str = "admin";

/// Authentik group name for legal document editors.
pub const GROUP_LEGAL_EDITOR: &str = "legal_editor";

/// Authentik group name for AI feature users.
pub const GROUP_AI_USER: &str = "ai_user";

/// Authentik group name for read-only legal viewers.
pub const GROUP_LEGAL_VIEWER: &str = "legal_viewer";
