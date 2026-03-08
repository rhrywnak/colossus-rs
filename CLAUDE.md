# CLAUDE.md — Colossus-Legal

> **Read this FIRST.** Then read `docs/CLAUDE_CODE_INSTRUCTIONS.md` for full standards.

## Project

**Colossus-Legal** — Legal document analysis and case management system (Awad v. CFS/Phillips)
- **Backend:** Rust + Axum (port 3403) → `backend/`
- **Frontend:** React + Vite + TS (port 5473) → `frontend/`
- **Database:** Neo4j 5.x (DEV: `bolt://10.10.100.200:7687`)
- **Vector DB:** Qdrant (DEV: REST `http://10.10.100.200:6333`, gRPC port 6334)
- **RAG Pipeline:** colossus-rag crate (Rig framework + Claude API)
- **Auth:** Authentik SSO → Traefik ForwardAuth → X-authentik-* headers
- **Shared Libraries:** colossus-rs workspace (colossus-auth, colossus-rag)

**Current Phase:** Deployment Stabilization (post-audit)
**Repos:** colossus-legal, colossus-rs, colossus-ansible, colossus-homelab

---

## Human Context

**Developer:** Roman — 45 years IT, CS degree, retired, learning Rust.
- Explain patterns when you use them
- Reference `docs/RUST-PATTERNS.md` for pattern examples
- Clear explanations over terse code
- Working code over perfect code

---

## The Golden Rules

```
 1. cargo check after EVERY change
 2. Never accumulate more than 10 errors
 3. No module over 300 lines (code lines, excluding doc comments)
 4. No function over 50 lines
 5. Pre-Coding Analysis BEFORE any code
 6. Wait for "Proceed" before implementing
 7. Every HTTP call MUST have a timeout
 8. No .unwrap() or .expect() in production handlers
 9. No plaintext secrets in code, config, or Butane files
10. No :latest tags on container images
11. Audit before deploying — verify the full path, not just the component
```

---

## Deployment & Configuration Rules

These rules exist because of real failures found in the March 2026 cross-repo audit.

### Secrets
- **NEVER** hardcode passwords, API keys, or tokens in source files, Butane configs, or scripts
- Secrets belong in Ansible Vault (`vault.yml`) or `.env` files that are gitignored
- `.env` files MUST be in `.gitignore` — verify before first commit
- If a secret is accidentally committed, it MUST be rotated immediately

### Environment Variables
- All config values that differ between DEV and PROD must be env vars
- Every env var the backend reads must exist in the Ansible template (`colossus-legal-backend.env.j2`)
- When adding a new env var: update `config.rs` + Ansible template + group_vars + vault (if secret)
- Use sensible defaults for local dev, but log a warning when defaults are used

### Timeouts (MANDATORY)
- **Frontend:** Every `authFetch` call must use `AbortController` with a timeout signal
  - Normal endpoints: 30 seconds
  - `/ask` (RAG synthesis): 90 seconds
- **Backend:** Every `reqwest::Client` must be built with `.timeout()` and `.connect_timeout()`
- **Backend:** Share one `reqwest::Client` via AppState — do not create per-request
- **Backend:** qdrant-client must have timeout configured

### Container Images
- Always pin to specific version tags (e.g., `v0.5.4`), never `:latest`
- Update Butane files when container versions change
- Version in `/api/status` must use `env!("CARGO_PKG_VERSION")`, not a hardcoded string

### Route Patterns
- Current state: mixed (`/ask` vs `/api/me`) — documented tech debt
- When adding new routes: follow the existing pattern for that area
- Do not add `/api/` prefix to existing non-prefixed routes without migration plan

### Docker Builds
- Never suppress build errors (`2>/dev/null || true` is forbidden)
- `.fastembed_cache/` must be in `.gitignore` (ONNX models are 500MB+)
- The `build-release.sh` in colossus-ansible is the canonical build script

---

## Mandatory Pre-Coding Process

**For EVERY task, provide Pre-Coding Analysis first:**

```markdown
## Pre-Coding Analysis for [Task ID]

### Task Understanding
[What will be implemented]

### Branch Verification
- Current: `feature/xxx`
- Clean: YES/NO

### Files to Modify
| File | Changes |
|------|---------|

### Files to Create
| File | Purpose | Est. Lines |
|------|---------|------------|

### Env Vars / Config Changes
| Variable | Where to Add | Default |
|----------|-------------|---------|
(Leave empty if none)

### Rust Patterns to Implement
| Pattern | Example |
|---------|---------|

### Tests to Write
| Test Name | Description |
|-----------|-------------|

### Deployment Impact
[Does this change require: new env vars? Ansible template update? Container rebuild? Traefik config change? If none, say "None — code-only change"]

### Potential Issues
[Any concerns]
```

**STOP. Wait for "Proceed" before writing code.**

---

## Post-Coding Requirements

```bash
git diff --name-only    # Only approved files?
cargo build             # Compiles?
cargo test              # Tests pass?
cargo clippy            # No warnings?
```

Provide completion report with build/test results.

**Before marking any task DONE:**
- If new env vars were added → confirm Ansible template updated
- If new endpoints were added → confirm frontend calls use timeout
- If new HTTP clients were created → confirm timeout configured

---

## Key Documents

| Document | When to Read |
|----------|--------------|
| `docs/CLAUDE_CODE_INSTRUCTIONS.md` | Before ANY coding task |
| `docs/TASK_TRACKER.md` | Check task status |
| `docs/DATA_MODEL_v3.md` | Working on Neo4j models/queries |
| `docs/RUST-PATTERNS.md` | Writing Rust code |
| `AUDIT_REPORT_COLOSSUS_LEGAL.md` | Before fixing audit items |
| `TRANSITION_DOC_2026-03-06_SESSION3_EOD.md` | Session continuity |

---

## Rust Quick Reference

```rust
// ✅ Required derives
#[derive(Debug, Clone, Serialize, Deserialize)]

// ✅ Enums with snake_case
#[serde(rename_all = "snake_case")]

// ✅ Error handling
#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("message: {0}")]
    Variant(String),
}

// ✅ Optional fields
#[serde(skip_serializing_if = "Option::is_none")]
pub field: Option<String>,

// ✅ HTTP client with timeout (MANDATORY)
let client = reqwest::Client::builder()
    .timeout(Duration::from_secs(30))
    .connect_timeout(Duration::from_secs(5))
    .build()?;

// ✅ Version from Cargo.toml
version: env!("CARGO_PKG_VERSION"),

// ❌ NEVER use in production handlers
option.unwrap()           // Use ? or match
"error".into()            // Use typed errors
reqwest::Client::new()    // Use builder with timeout
```

---

## Commands

```bash
# Backend
cd backend && cargo check    # Quick check
cd backend && cargo test     # Run tests
cd backend && cargo clippy   # Lint

# Git
git branch --show-current
git status
git diff --name-only

# Module size check (before committing)
find src -name "*.rs" -exec sh -c \
  'lines=$(grep -v "^\s*$" "$1" | grep -v "^\s*//" | wc -l); \
   if [ $lines -gt 300 ]; then echo "OVER: $lines $1"; fi' _ {} \;
```

---

## What NOT To Do

❌ Write code before Pre-Coding Analysis approved
❌ Modify files not in approved list
❌ Add features not in task spec
❌ Use `unwrap()` or `expect()` in production handlers
❌ Create modules over 300 lines
❌ Skip layers (must do L0 before L1)
❌ Create HTTP clients without timeouts
❌ Create fetch calls without AbortController
❌ Hardcode secrets, passwords, or API keys
❌ Use `:latest` tags on container images
❌ Suppress build errors with `2>/dev/null || true`
❌ Commit `.env` files or `.fastembed_cache/` to git
❌ Add env vars to backend without updating Ansible template
❌ Deploy without testing the full path (browser → Traefik → auth → backend → response)

---

## If Something Goes Wrong

**STOP all edits.** Report the issue. Read-only operations only until resolved.

---

## Layer System

| Layer | Description |
|-------|-------------|
| L0 | Skeleton — compiles, structure in place |
| L1 | Real Data — happy path works |
| L2 | Validation — error handling complete |
| L3 | Integration — advanced features |

Never skip layers.

---

## Architecture Quick Reference

```
Browser → Traefik (TLS) → Authentik ForwardAuth (frontend only)
                        → Backend (API routes: no ForwardAuth, backend checks X-authentik-* headers)

RAG Pipeline: Question → Router → QdrantRetriever → Neo4jExpander → LegalAssembler → RigSynthesizer → Answer

Repos:
  colossus-legal     — Application (Rust backend + React frontend)
  colossus-rs        — Shared Rust libraries (colossus-auth, colossus-rag)
  colossus-ansible   — Deployment automation (Ansible + Semaphore)
  colossus-homelab   — Infrastructure docs, Butane configs, scripts
```

---

# End of CLAUDE.md
