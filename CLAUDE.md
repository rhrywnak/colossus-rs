# CLAUDE.md — colossus-rs

> **Read this FIRST before any task.**

## Project

**colossus-rs** — Shared Rust library workspace for Colossus applications.
This is a library workspace. It has no binary, no HTTP server, no frontend,
no Ansible deployment, and no containers. It is consumed as a git dependency
by colossus-legal and colossus-ai.

### Crates in this workspace

| Crate | Description |
|-------|-------------|
| colossus-auth | Authentik + Axum authentication integration |
| colossus-extract | Document extraction types, traits, providers, schema loader |
| colossus-rag | RAG pipeline (retriever, expander, synthesizer, decomposer) |
| colossus-pdf | PDF text extraction |
| colossus-graph | Neo4j query functions (domain-agnostic) |
| colossus-pipeline | Async job pipeline framework (domain-agnostic) ← ACTIVE PHASE |

### Current phase

**Phase PV — colossus-pipeline crate build (P1-1 through P1-15)**
Design doc: COLOSSUS_PIPELINE_DESIGN_v5_2.md
Task tracker: COLOSSUS_PIPELINE_TASK_TRACKER_v1_1.md
Branch: main

---

## Human Context

**Developer:** Roman — 45 years IT, CS degree, retired, learning Rust.
- Explain every Rust pattern you use with a `## Rust Learning:` doc comment
- Reference patterns in doc comments, not in chat
- Clear explanations over terse code
- Working code over perfect code

---

## The Golden Rules

```
1. cargo check after EVERY change
2. Never accumulate more than 10 errors
3. No module over 300 lines (code lines, excluding doc comments)
4. No function over 50 lines
5. Tests MUST pass before cargo build — a clean compile is NOT verification
6. Never bump version numbers — Roman does that
7. Every module, struct, trait, and public function MUST have a doc comment
   explaining what it does AND why it exists in this system
8. No magic strings or numbers — use constants
9. No .unwrap() or .expect() in library code — use ? or explicit error handling
10. Single repo only — never reference files in colossus-legal or any other repo
```

---

## Doc Comment Requirement (MANDATORY)

Every file must have a `//!` module doc comment at the top:

```rust
//! colossus-pipeline/src/worker/heartbeat.rs
//!
//! Heartbeat task for pipeline job liveness tracking.
//!
//! One-sentence description of what this module does.
//! One-sentence description of WHY it exists (what problem it solves).
//!
//! ## Rust Learning: [pattern name]
//!
//! Explanation of the key Rust pattern used in this module.
```

Every public struct, enum, trait, and function must have a `///` doc comment:

```rust
/// Short description of what this is.
///
/// Longer explanation of why it exists and how it fits in the system.
/// Include Rust Learning notes when a non-obvious pattern is used.
pub struct MyStruct { ... }
```

---

## Pre-Coding Process

For every task, report these before writing any code:

```
### Files to read (report contents before modifying)
### Files to modify (exact paths)
### Files to create (exact paths)
### Tests to write (names and what they verify)
### Potential issues
```

Then proceed — no explicit "Proceed" gate required for colossus-rs tasks
unless the task instruction specifies a STOP gate.

---

## Post-Coding Requirements

```bash
cargo test -p <crate-name>     # Tests pass FIRST
cargo build -p <crate-name>    # Then build
cargo build --workspace        # Confirm no workspace breakage
```

Provide completion report: commit hash, test count, error/warning count.

---

## Rust Quick Reference

```rust
// ✅ Required derives for pipeline types
#[derive(Debug, Clone, Serialize, Deserialize)]

// ✅ sqlx enum mapping to TEXT column
#[derive(sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "snake_case")]

// ✅ serde snake_case for enums
#[serde(rename_all = "snake_case")]

// ✅ Error handling with thiserror
#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("descriptive message: {0}")]
    Variant(String),
}

// ✅ async trait (required for trait objects)
#[async_trait::async_trait]
pub trait MyTrait: Send + Sync + 'static {
    async fn my_method(&self) -> Result<(), MyError>;
}

// ✅ Arc for shared ownership across tasks
pub field: Arc<dyn MyTrait>

// ❌ NEVER
option.unwrap()          // Use ? or match
todo!()                  // Stubs use // Stub comment, not todo!()
version bump             // Roman bumps versions only
```

---

## What NOT To Do

❌ Reference colossus-legal, colossus-ansible, or colossus-homelab paths
❌ Write code before reading the files the task specifies
❌ Bump version numbers in any Cargo.toml
❌ Use unwrap() or expect() in library code
❌ Skip writing tests — tests before cargo build, always
❌ Leave stubs without a `// Stub — full implementation in P1-N` comment
❌ Write a function without a doc comment
❌ Use string literals where a constant should be used
❌ Combine changes to multiple crates in one commit

---

## If Something Goes Wrong

**STOP all edits.** Report the exact compiler error or test failure.
Read-only operations only until the issue is understood.
Never fix a test to make it pass — fix the code.

---

## Commands

```bash
# Build and test a single crate
cargo build -p colossus-pipeline
cargo test -p colossus-pipeline

# Build entire workspace (run after every crate change)
cargo build --workspace
cargo test --workspace

# Check current branch
git branch --show-current   # Must return: main

# Lint
cargo clippy -p colossus-pipeline
```

---

## Architecture Context

```
colossus-rs (this repo — library only)
  └── colossus-pipeline    ← current work
        Domain-agnostic job queue backed by PostgreSQL.
        No knowledge of LLMs, legal documents, or any application domain.
        colossus-legal and colossus-ai both use this crate unchanged.

colossus-legal (separate repo — do not touch)
  └── Uses colossus-pipeline via workspace path dependency
  └── Defines DocProcessing Task enum and Step implementations
  └── Branch: feature/pipeline-v5

colossus-ai (future repo — do not touch)
  └── Will use colossus-pipeline unchanged
```

---

# End of CLAUDE.md
