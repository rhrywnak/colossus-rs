//! # spike — Rig Framework Compatibility Validation
//!
//! This is a throwaway crate used to prove that the Rig framework
//! (rig-core, rig-qdrant, rig-fastembed) works with our infrastructure:
//! - Qdrant vector database (colossus_evidence collection)
//! - fastembed (nomic-embed-text v1.5, 768 dimensions)
//! - Claude API via Rig's Anthropic provider
//!
//! **DO NOT** build production features on this crate. It will be deleted
//! once the spike results are recorded.

// This crate is intentionally empty — all validation happens in tests/rig_spike.rs.
