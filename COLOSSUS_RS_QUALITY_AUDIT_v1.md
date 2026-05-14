# colossus-rs Library Quality Audit — v1

**Date:** 2026-05-13
**Auditor:** Claude Code (Opus 4.7, 1M context)
**Repository:** colossus-rs
**Commit:** `751bdd3d950b4e2325196c4d7389972cf725647b`
**Branch at audit time:** `feature/strict-schema-validation` (the audit instruction specified `main`; the branch was not switched. The current branch is one commit ahead of `main` — commit `751bdd3 feat!: enforce required and min_count in schema YAML` — but otherwise audit findings reflect the same code that will reach `main`.)
**Scope:** All Rust source files in `colossus-auth`, `colossus-extract`, `colossus-graph`, `colossus-pdf`, `colossus-pipeline`, `colossus-rag`, `spike` (workspace members), plus every `Cargo.toml`.

---

## Executive Summary

This audit is a complete inventory of every error-handling, silent-failure, hardcoded-value, and code-quality problem in colossus-rs. Findings are listed per file with line numbers. This is a READ-ONLY audit; no code was changed.

- **Total issues found: ~140**
- **Critical (causes wrong data to reach consumer silently, or breaks a documented invariant): 4**
- **High (causes consumer errors with poor diagnostics, or significant data quality loss): 18**
- **Medium (limits flexibility, hides operator-debuggable info, or risks subtle bugs): 56**
- **Low (code quality, documentation, minor robustness): ~60**

**Critical findings at a glance:**

1. `colossus-pipeline/src/worker/executor.rs:93` — Hardcoded SQL control-string literal `"cancel_requested"` instead of `JobControl::CancelRequested.as_str()`. Direct violation of the `CLAUDE.md` enum-binding invariant ("A typo in a literal is a silent runtime failure"). If the enum's serialized value ever drifts, cancellation will silently never fire.
2. `colossus-pdf/src/docx_extractor.rs:55, 104` — All DOCX content is mapped to page 1 (`SINGLE_PAGE_NUMBER: i32 = 1`); multi-page DOCX cannot be page-grounded. Library is silently lossy for any DOCX with page breaks.
3. `colossus-extract/src/structure_splitter.rs:92-94` — Invalid boundary regex is silently swallowed (`Err(_) => return Vec::new()`), and the caller's "no_boundary_matches" fallback path produces the *same* output as a successful zero-match. An operator misconfiguring the schema cannot tell from logs that the regex is malformed.
4. `colossus-graph/src/queries.rs:35-54, 64, 69, 138-142` — `extract_node_properties` and `row_to_graph_node` silently coerce/empty-default critical fields (unknown property type → `Null`; missing `labels` column → empty vec; missing `id` → empty string). Combined with `get_node_neighbors`'s empty-ID deduplication (line 138-142), nodes with no `id` are silently dropped, and unknown-type properties surface as `null` indistinguishable from explicit nulls. This is the worst-case shape for a library: bad data flows to the consumer with no signal.

**Notable systemic patterns:**

- **Error type strategy is good across the workspace** — every consumer crate uses `thiserror`-derived enums and the variants carry context. The weakest is `colossus-graph::GraphAccessError` which flattens all neo4rs errors into `QueryFailed(String)`, losing the timeout/auth/network distinction.
- **Domain-agnostic discipline holds.** `colossus-pipeline` has no LLM/document knowledge; `colossus-rag` has no PDF knowledge; `colossus-graph` has no legal-schema knowledge. The CLAUDE.md "no application domain knowledge" rule is observed.
- **`colossus-pdf` is the lowest-tested crate** — no `tests/` directory at all; only inline `#[cfg(test)]` modules. Given that this crate sits upstream of every extraction in colossus-legal, this is a coverage gap that warrants investment.
- **`spike/` crate should be deleted.** Its own docstring says so. It also pulls dual rig-core versions (0.31 transitively, 0.33 directly) into the dependency tree.

---

## Statistics

- **Files audited:** 86 (`*.rs`) + 8 (`Cargo.toml`)
- **Public API items audited:** ~125 across 7 crates (re-exports, structs, enums, traits, functions, consts)
- **Error types audited:** 6 (one per consumer crate: `AuthError`, `PipelineError` [extract], `GraphAccessError`, `PdfError`, `PipelineError` [pipeline — namespace collision with extract], `RagError`)
- **Silent failures found:** ~25 (counting only patterns that lose information the consumer cannot recover; intentional "graceful degradation" with logging is reported but not counted as a defect)
- **Hardcoded values found:** ~55 (constants, magic numbers, Cypher LIMIT clauses, default timeouts, threshold scores)
- **Undocumented public items:** ~7 (mostly in colossus-pdf: `PdfError` enum, `PageText`, `GroundingResult`, `MatchType`, `SearchHit`, `SearchConfig`; one in colossus-extract: `PromptBuilder` struct)
- **Test coverage gaps:** ~14 modules have zero or near-zero unit tests (notably `colossus-pdf` has no `tests/` dir at all; `colossus-graph` has zero tests; `colossus-rag::graph_retriever` has zero dedicated tests)
- **Dead code instances:** 1 entire crate (`spike`, by design — flagged in source), 1 duplicated helper (`map_neo4j_err` in two expander files), 2 documented TODOs

---

## Section 1: Error Propagation

### 1a. Error Types

#### colossus-auth

- **`colossus-auth/src/error.rs:22-34`** — TYPE: `pub struct AuthError`
  - VARIANTS: 4 public fields — `error: String`, `message: String`, `user: Option<String>`, `groups: Option<Vec<String>>`
  - IMPLEMENTS: `Debug` (derived), `Clone` (derived), `Serialize` (derived), `IntoResponse` (custom impl at `error.rs:45-56`)
  - CONTEXT PRESERVED: Yes — user and groups fields preserved for 403 errors.
  - QUALITY: **GOOD**
  - PROBLEM: None; struct shape leaks publicly (see 7c).

#### colossus-extract

- **`colossus-extract/src/error.rs:5-74`** — TYPE: `pub enum PipelineError`
  - VARIANTS:
    - `Schema(String)` (line 7) — schema validation/parsing failures
    - `Template(String)` (line 10) — template file load/render failures
    - `LlmProvider(String)` (line 13) — all HTTP/auth/API failures
    - `Extraction(String)` (line 16) — LLM extraction logic failures
    - `Verification(String)` (line 19) — post-extraction verification failures
    - `EntityResolution(String)` (line 22) — entity matching failures
    - `Io(#[from] std::io::Error)` (line 25) — file I/O errors with automatic From
    - `Json(#[from] serde_json::Error)` (line 28) — JSON parse/encode failures
    - `Yaml(#[from] serde_yaml::Error)` (line 31) — YAML parse failures
    - `RateLimited { retry_after_secs: u64 }` (line 68) — typed rate-limit
  - IMPLEMENTS: `Debug`, `Error`, `Display` via `thiserror::Error`; `From` for io/json/yaml.
  - CONTEXT PRESERVED: Excellent. RateLimited is typed (orchestrator can read the retry window). String variants receive contextual messages at the call site.
  - QUALITY: **GOOD**
  - PROBLEM: None.

#### colossus-graph

- **`colossus-graph/src/error.rs:5-15`** — TYPE: `pub enum GraphAccessError`
  - VARIANTS:
    - `QueryFailed(String)` — Neo4j query execution errors
    - `NodeNotFound(String)` — no matching node by ID (defined but **never constructed in code**)
    - `PropertyExtraction(String)` — property type coercion failures (defined but **never constructed in code**)
  - IMPLEMENTS: `Debug`, `Error` via `thiserror::Error`.
  - CONTEXT PRESERVED: Partial. neo4rs::Error structural type is collapsed to a string via `From<neo4rs::Error>`.
  - QUALITY: **POOR**
  - PROBLEM: No distinction between transient (timeout, network) and permanent (auth, syntax) failures; `NodeNotFound` and `PropertyExtraction` variants are dead — `Ok(None)` and `Ok(Value::Null)` are used instead, which is exactly the silent-failure shape this enum was meant to avoid.

#### colossus-pdf

- **`colossus-pdf/src/error.rs:5-21`** — TYPE: `pub enum PdfError`
  - VARIANTS:
    - `OpenError(String)` — file open/format detection failures
    - `ExtractionError { page: u32, message: String }` — per-page extraction with page context
    - `PageOutOfRange(u32, u32)` — bounds validation (page, total_pages)
    - `NoTextLayer` — scanned PDF detection
    - `Io(#[from] std::io::Error)` — transparent I/O wrapping
  - IMPLEMENTS: `Debug`, `Error` via `thiserror::Error`.
  - CONTEXT PRESERVED: `ExtractionError` carries page number. `Io` variant via `#[from]` carries no path/operation context.
  - QUALITY: **ADEQUATE**
  - PROBLEM: Top-level enum has no `///` doc comment (see 7a). `Io(#[from] std::io::Error)` is transparent — when a path I/O error bubbles up, the file path is lost. `OpenError` variants at `extractor.rs:54-60, 92-93` do not include the file path in the error message (lose context).

#### colossus-pipeline

- **`colossus-pipeline/src/error.rs:13-55`** — TYPE: `pub enum PipelineError` (namespace collision with `colossus-extract::PipelineError`)
  - VARIANTS:
    - `Database(String)` — sqlx error wrapped
    - `NotFound(Uuid)` — job not found
    - `JobNotCancellable(Uuid)` — invalid FSM state for cancel
    - `JobNotResumable(Uuid)` — invalid FSM state for resume
    - `JobRunning(Uuid)` — cannot delete running job
    - `DuplicateJob { job_type, job_key }` — duplicate, with typed context
    - `InvalidTransition { from, to }` — FSM violation with typed context
    - `Serialization(String)`
    - `Cleanup(String)`
    - `LlmProvider(String)` — *odd placement*: the pipeline crate is supposed to be domain-agnostic, yet it has an LLM-named variant. This appears to be a leak from earlier development; nothing in the crate constructs it. **Recommendation in 1c.**
  - IMPLEMENTS: `Debug`, `Error` via `thiserror::Error`; `From<sqlx::Error>`, `From<serde_json::Error>`.
  - CONTEXT PRESERVED: Excellent — typed variants for FSM and duplicate-key cases.
  - QUALITY: **GOOD** (with one cleanup recommendation: the `LlmProvider` variant should be removed; it breaks the domain-agnostic invariant in CLAUDE.md.)

#### colossus-rag

- **`colossus-rag/src/error.rs:34-83`** — TYPE: `pub enum RagError`
  - VARIANTS:
    - `InvalidInput(String)` (line 40)
    - `EmbeddingError(String)` (line 47)
    - `SearchError(String)` (line 54)
    - `ExpandError(String)` (line 61)
    - `AssemblyError(String)` (line 69)
    - `SynthesisError(String)` (line 76)
    - `ConfigError(String)` (line 82)
  - IMPLEMENTS: `Debug`, `Clone`, `Error`, `Display` via `thiserror::Error`.
  - CONTEXT PRESERVED: Good — flat enum, one variant per pipeline stage.
  - QUALITY: **GOOD**
  - PROBLEM: None.

#### spike

- No custom error types. All errors propagated via `.expect()` in test/spike context.

### 1b. Error Conversion / From Impls

#### colossus-auth

No `From<X> for Y` impls beyond what `IntoResponse` provides. No `.map_err()` calls.

#### colossus-extract

| FILE:line | Conversion | Context Added | Quality |
|---|---|---|---|
| `error.rs:25, 28, 31` | `From<io::Error/serde_json::Error/serde_yaml::Error>` | None — transparent | GOOD (carries source chain via `Error::source()`) |
| `prompt.rs:60-65` | `io::Error → PipelineError::Template` | `"Failed to load template '{}': {}"` with path | GOOD |
| `prompt.rs:157` | `serde_json::Error` via `?` | Implicit From | GOOD |
| `schema.rs:224-229` | `io::Error → PipelineError::Schema` | `"Failed to read schema file {}: {}"` with path | GOOD |
| `schema.rs:231,239` | `serde_yaml::Error` via `?` | Implicit From | GOOD |
| `schema.rs:366` | `serde_json::Error` via `?` | Implicit From | GOOD |
| `structure_splitter.rs:92-94` | `regex::Error → Vec::new()` | **DISCARDED — no log, no error** | **POOR (Critical Silent Failure)** |
| `structure_splitter.rs:99-101` | `regex::Error → None` for response marker | Silent fall-through | POOR |
| `providers/anthropic.rs:201-203` | reqwest builder error → LlmProvider | `"Failed to build Anthropic HTTP client: {e}"` | GOOD |
| `providers/anthropic.rs:276-290` | reqwest network error → LlmProvider | Distinguishes timeout vs other; includes timeout_secs | EXCELLENT |
| `providers/anthropic.rs:340-341` | serde_json error → LlmProvider | `"Failed to parse Anthropic response: {e}"` | GOOD |
| `providers/vllm.rs:181-183` | reqwest builder → LlmProvider | `"Failed to build vLLM HTTP client: {e}"` | GOOD |
| `providers/vllm.rs:226-240` | reqwest network → LlmProvider | Timeout-aware mapping | GOOD |
| `providers/vllm.rs:282-283` | serde_json → LlmProvider | `"Failed to parse vLLM response: {e}"` | GOOD |
| `providers/vllm_embed.rs:165-169` | reqwest builder → LlmProvider | path-aware | GOOD |
| `providers/vllm_embed.rs:206-220` | reqwest network → LlmProvider | Timeout-aware | GOOD |
| `providers/vllm_embed.rs:259-260` | serde_json → LlmProvider | Embedded response parse | GOOD |
| `providers/fastembed.rs:87-92` | u32 conversion error → LlmProvider | Dimension out-of-range with value | GOOD |
| `providers/fastembed.rs:100,128` | fastembed init → LlmProvider | `"fastembed init: {e:#}"` (pretty) | GOOD |
| `providers/fastembed.rs:156-159, 192-196` | fastembed embed → LlmProvider | model name included | GOOD |
| `providers/fastembed.rs:161-165` | Empty batch → LlmProvider | Explicit error | GOOD |
| `providers/fastembed.rs:168, 199` | task::JoinError → LlmProvider | "spawn_blocking failed" | GOOD |
| `providers/factory.rs:334-335, 341-345, 196-200` | std::fs/parse → LlmProvider | env var name + raw value | GOOD |

#### colossus-graph

- **`error.rs:18-22`** — `From<neo4rs::Error> for GraphAccessError`: `neo4rs::Error → QueryFailed(e.to_string())`. CONTEXT LOST: the structured neo4rs error type (timeout / connection / auth / constraint) is reduced to a string. Caller cannot distinguish transient from permanent failures. QUALITY: **POOR**.
- **`queries.rs:92, 124, 190, 211, 250, 269, 292, 318`** — implicit propagation via `?` using the `From` impl above. Same quality concern (POOR).

#### colossus-pdf

| FILE:line | Conversion | Context Added | Quality |
|---|---|---|---|
| `extractor.rs:54-56, 57-60` | pdf_oxide error → `OpenError(format!("{e}"))` / "Failed to read page count: {e}" | No file path | POOR (path lost) |
| `extractor.rs:92-93` | pdf_oxide error from bytes → `OpenError` | No "from bytes" mention | MEDIUM |
| `extractor.rs:138-143` | pdf_oxide page extract → `ExtractionError { page, message }` | Page number included | GOOD |
| `format_detection.rs:91-97` | mimetype_detector → `OpenError` | Path + library error | EXCELLENT |
| `docx_extractor.rs:64-70, 72-78` | docx-rust open/parse → `OpenError` | Path + `{:?}` debug | GOOD |
| `plain_text_extractor.rs:47-53` | std::fs read_to_string → `OpenError` | Path + error | EXCELLENT |
| `error.rs:20` | `#[from] std::io::Error → Io` | Transparent — **no path** | POOR |

#### colossus-pipeline

| FILE:line | Conversion | Context Added | Quality |
|---|---|---|---|
| `error.rs:57-60` | `From<sqlx::Error> → Database(e.to_string())` | None | ADEQUATE (sqlx error display is detailed) |
| `error.rs:63-66` | `From<serde_json::Error> → Serialization` | None | ADEQUATE |
| `worker/handler.rs:77, 104` | serde_json::to_value → `Serialization` | None beyond Display | GOOD |
| `worker/executor.rs:54-56` | deserialize → `ExecutionResult::Failed(format!(...))` | format!-wrapped | GOOD |
| `scheduler.rs:106-113` | sqlx unique-violation → `DuplicateJob { job_type, job_key }` | Typed context | EXCELLENT |
| `scheduler.rs:258-259` | serde_json::from_value → `Serialization` | None | GOOD |

#### colossus-rag

| FILE:line | Conversion | Context Added | Quality |
|---|---|---|---|
| `retriever.rs:151, 180` | provider/Qdrant error → `EmbeddingError`/`SearchError` (.to_string()) | None | GOOD |
| `reranker.rs:117, 128` | provider error → `EmbeddingError` | None | GOOD |
| `expander.rs:282, 288-289` | neo4rs error → `ExpandError` | `"Type resolution query failed: {e}"` | GOOD |
| `expander_queries.rs:48-49` | helper `map_neo4j_err` → `ExpandError` | None | GOOD |
| `expander_queries_minor.rs:20-22` | duplicate helper — **same code as above, redefined** | None | LOW (code duplication) |
| `graph_retriever.rs:67, 74, 142, 149, 225, 234` | neo4rs error → `SearchError` | `"Graph {...} query failed: {e}"` | GOOD |
| `synthesizer.rs:144` | LlmProvider → `SynthesisError` | None | GOOD |
| `decomposer.rs:252` | LlmProvider error → fallback (special handling) | Logs warn, uses no-op decomposition | INTENTIONAL (graceful degradation) |

#### spike

No From impls. Test code uses `.expect()` with explicit messages.

### 1c. Poor Result Return Patterns

- **`colossus-graph/src/queries.rs:86-99` (`get_node_by_id`)** — Returns `Result<Option<GraphNode>>`. The `NodeNotFound` variant exists but is unused; `Ok(None)` is returned for missing nodes. Consumer cannot tell whether the query succeeded or simply matched no rows.
- **`colossus-graph/src/queries.rs:112-174` (`get_node_neighbors`)** — Missing center node returns `Ok(NodeNeighborhood { node: None, ... })`, not an error. Inverts the success/error signal.
- **`colossus-graph/src/queries.rs:180-199, 205-220, 230-259, 286-303, 310-342`** — Every "get X by criteria" function returns `Ok(Vec::new())` when no rows match. Indistinguishable from "query succeeded but found zero results." This is intentional REST/SQL semantics but caller has no way to distinguish data-state from misconfiguration.
- **`colossus-pipeline/src/error.rs:55` — `LlmProvider(String)` variant** — defined but never constructed in this crate (verified by grep). Violates the CLAUDE.md domain-agnostic rule by name. The variant should be removed.
- **`colossus-extract/src/structure_splitter.rs:92-95` (`detect_units`)** — `regex::Error → Vec::new()`. Caller has no way to distinguish a malformed regex from a regex that legitimately matches nothing. This is **the most operationally hostile silent failure in the workspace**.
- **`colossus-pipeline/src/recorder.rs:51-56, 64-69, 77-82`** — Recorder trait returns `Result<i64, Box<dyn Error + Send + Sync>>`. Opaque error is **intentional decoupling** between pipeline framework and application step-recording, and is the right choice — flagged here for completeness only.

---

## Section 2: Silent Failures

#### colossus-auth

| FILE:line | Pattern | What's lost | Consequence | Severity |
|---|---|---|---|---|
| `extractor.rs:101` | `extract_header(...).unwrap_or_default()` | Missing `x-authentik-email` becomes empty string | Email field may be empty; documented as optional | LOW (intentional) |
| `extractor.rs:103` | `.unwrap_or_else(|| username.clone())` | Missing display name falls back to username | Documented fallback | LOW (intentional) |
| `extractor.rs:177` | `.map(...).unwrap_or_default()` on groups header | Missing groups header → empty Vec | User correctly has no groups | LOW (intentional) |
| `extractor.rs:64` | Anonymous user gets `vec!["admin".to_string()]` in Optional mode | None — but unauthenticated user gets admin in Optional mode | Documented but **risky design** | MEDIUM |

#### colossus-extract

| FILE:line | Pattern | What's lost | Consequence | Severity |
|---|---|---|---|---|
| `config.rs:61, 66, 70, 74, 78` | `.unwrap_or(default)` on coerce chains | Failed type coercion → default | By design (ConfigAccess trait pattern) | INTENTIONAL |
| `prompt.rs:69` | `self.cache.get(name).expect("just inserted")` | None — proof-carrying | Safe (just inserted) | OK |
| `prompt.rs:105, 128, 133` | `unwrap_or` defaults for missing template name/context/admin-instructions | None — caller-facing defaults | Documented | OK |
| `providers/anthropic.rs:195, 301, 314, 327` | timeout/header/body defaults | Missing retry-after → 60s default; failed body read → empty string in error | Reasonable | OK |
| `providers/vllm.rs:176, 251, 265, 294, 297` | Same patterns as anthropic.rs | Same | OK |
| `providers/vllm_embed.rs:160, 230, 243` | Same | Same | OK |
| `resolver.rs:200` | `.unwrap_or("individual")` for missing `party_type` | Defaulted to "individual" | Documented | OK |
| `structure_splitter.rs:92-94` | regex Err → `Vec::new()` (no log, no error) | regex compile error indistinguishable from zero matches | Operator cannot diagnose bad schema | **HIGH (Critical)** |
| `structure_splitter.rs:99-101` | response marker regex Err → `None` (no log) | `has_response` metadata silently omitted | Misconfiguration undetectable | HIGH |

#### colossus-graph

| FILE:line | Pattern | What's lost | Consequence | Severity |
|---|---|---|---|---|
| `queries.rs:35-54` | Property type fallback chain → `Value::Null` | Unknown property type (Map, nested object) | Consumer sees `null` indistinguishable from real null | **HIGH** |
| `queries.rs:64` | `row.get(labels_key).ok().unwrap_or_default()` | Missing `labels` column | Nodes have empty label set | MEDIUM |
| `queries.rs:69` | `.unwrap_or("")` on `id` extraction | Empty-string ID is semantically invalid | Multiple nodes share empty ID; deduplication breaks | **HIGH** |
| `queries.rs:138-142` | Empty `m_id` neighbor skipped | A neighbor with no `id` is dropped from the neighborhood | Edges silently invisible | **HIGH** |
| `queries.rs:151-152, 273-274, 296-297, 322-325` | `unwrap_or(default)` on extracted fields | Missing field → default (booleans default to `true`, strings to `""`) | `outgoing: true` default reverses relationship direction silently | MEDIUM |
| `queries.rs:275` | `if !label.is_empty()` filter on labels | Unlabeled nodes silently dropped from `get_label_counts` | Data loss in introspection | LOW |

#### colossus-pdf

| FILE:line | Pattern | What's lost | Consequence | Severity |
|---|---|---|---|---|
| `extractor.rs:154` | `self.page_cache[index].as_deref().unwrap_or_default()` | Cache slot unexpectedly None → empty string | Logic-error masked as blank page | MEDIUM |
| `docx_extractor.rs:89-93` | `if let BodyContent::Paragraph(p) = content` — tables/headers/footers silently skipped | Table data, headers, footers, footnotes | Extracted text is incomplete; no per-file warning | **HIGH** |
| `page_grounder.rs:141-146` | `.unwrap_or(GroundingResult { … NotFound })` | Should never trigger (invariant), but no log if it does | Brittle | LOW |
| `format_detection.rs:125-126` | `.split(';').next().unwrap_or(full_mime).trim()` | None (defensive on infallible split) | None | NONE |

#### colossus-pipeline

| FILE:line | Pattern | What's lost | Consequence | Severity |
|---|---|---|---|---|
| `progress.rs:56-65` | `report()` swallows errors | Failed progress write | Frontend stale; documented intentional | MEDIUM (intentional) |
| `worker/mod.rs:315-326` | `events::log()` failures warn-only | Event log row | Audit trail incomplete; intentional | MEDIUM (intentional) |
| `worker/mod.rs:328-339` | `recorder.on_step_start()` failure returns None | Step recording abandoned | Application step history incomplete | MEDIUM (intentional) |
| `worker/mod.rs:385-392` | `recorder.on_step_success()/on_step_failure()` warn-only | Same | MEDIUM (intentional) |
| `worker/heartbeat.rs:54-71` | DB errors in heartbeat loop warn-only | Single heartbeat update | Self-corrects next interval | MEDIUM (intentional) |
| `worker/fetcher_recovery.rs:133-141` | `resolve_step_config` errors → defaults | Per-step config | Falls back to compiled defaults | LOW (intentional) |

#### colossus-rag

| FILE:line | Pattern | What's lost | Consequence | Severity |
|---|---|---|---|---|
| `retriever.rs:240-254` | Person/Collection filters matched but **not applied** to Qdrant query | User-requested filter silently ignored | Query returns over-broad results | **MEDIUM-HIGH** |
| `pipeline_helpers.rs:135-139` | Graph sub-queries return `Vec::new()` when `neo4j` feature disabled | User configured decomposer but didn't enable feature | Empty results, no error | MEDIUM |
| `expander.rs:212-234` | Per-seed Neo4j error: one seed fails, others continue silently | Partial expansion | User receives partial graph without knowing | MEDIUM |
| `decomposer.rs:254-262` | LLM error → fallback no-op decomposition | LLM provider failure | Logged as warn; intentional graceful degradation | LOW (intentional) |
| `reranker.rs:91-161` | Chunks below threshold dropped from rerank output | Filtered nodes | Stats populated; behavior intentional | LOW |
| `synthesizer.rs:161-165` | `.unwrap_or(0)` on missing token counts | Token counts | Cannot distinguish "unknown" from "zero" | MEDIUM |
| `assembler.rs:205-217` | Chunks truncated when budget exhausted | Low-score chunks dropped | Tracing.debug logs; budget-driven | LOW |
| `graph_retriever.rs:77-79, 151-154, 237-268` | `if id.is_empty() { continue; }` skips Neo4j rows silently | Optional MATCH null rows | Expected behavior; not logged | LOW |
| `noop.rs:44, 64` | NoOp impls return `Ok` with empty/default | None — Null Object Pattern | Intentional | LOW |

#### spike

No silent failures — all errors are `.expect()`-handled with explicit messages.

---

## Section 3: Hardcoded Values

#### colossus-auth

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `extractor.rs:32-36` | Header names (`x-authentik-username`, etc.), groups separator `'|'` | Module-private const (already is); documented | LOW |
| `lib.rs:45, 48, 51, 54` | `GROUP_ADMIN="admin"`, `GROUP_LEGAL_EDITOR="legal_editor"`, `GROUP_AI_USER="ai_user"`, `GROUP_LEGAL_VIEWER="legal_viewer"` | Coupled to Authentik group naming | LOW (documented contract) |
| `extractor.rs:64` | Anonymous user constructed with `vec!["admin".to_string()]` (Optional auth mode) | Configurable or removed | **MEDIUM** — unauthenticated user gets admin in Optional mode |

#### colossus-extract

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `providers/anthropic.rs:56` | `"https://api.anthropic.com/v1/messages"` endpoint URL | Stable; per Anthropic docs | LOW |
| `providers/anthropic.rs:59` | `"2023-06-01"` API version | Anthropic-mandated | LOW |
| `providers/anthropic.rs:70` | `600` (default request timeout secs) | Configurable via constructor | MEDIUM (constructor override exists) |
| `providers/anthropic.rs:77` | `60` TCP keep-alive | Network-stability driven | LOW |
| `providers/anthropic.rs:84` | `60` fallback retry-after | Anthropic guarantees header present; safe fallback | LOW |
| `providers/anthropic.rs:89` | `"text"` content block type | OpenAI-compat | LOW |
| `providers/vllm.rs:35` | `/v1/chat/completions` | OpenAI spec | LOW |
| `providers/vllm.rs:43, 48, 54, 60` | Same family of constants as Anthropic | Same | LOW–MEDIUM |
| `providers/vllm_embed.rs:39, 46, 51, 57` | `/v1/embeddings`, 600s timeout, keep-alives | OpenAI spec + reasonable | LOW–MEDIUM |
| `structure_splitter.rs:25` | Default boundary regex `r"^\d+\.\s"` | Configurable via YAML | LOW |
| `structure_splitter.rs:27` | `25` units_per_chunk default | Configurable via YAML | LOW |
| `structure_splitter.rs:30` | `0` unit_overlap default | Configurable | LOW |
| `structure_splitter.rs:33, 35` | Multiline regex prefix `(?m)`, chunk joiner `"\n\n"` | Standard | LOW |
| `splitter.rs:35` | `4000` default chunk size (chars) | Configurable via constructor | LOW |
| `splitter.rs:36` | `200` default chunk overlap | Configurable | LOW |
| `resolver.rs:38` | `0.85` default Jaro-Winkler fuzzy threshold | Configurable via `.with_threshold()` but not constructor param | **MEDIUM** (critical to entity dedup) |
| `merger.rs:31-43` | ID prefix constants (`"chunk"`, `':'`, `"-c"`, `"unnamed-"`) | Algorithm semantics | LOW |
| `providers/factory.rs:73, 76, 79` | Default `"anthropic"`, `"fastembed"`, `"huggingface"` | Per spec | LOW |
| `providers/factory.rs:83` | `32_000` default LLM_MAX_TOKENS | Overridable via env | MEDIUM |

#### colossus-graph

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `queries.rs:90` | `LIMIT 1` in `get_node_by_id` | Configurable or documented | LOW–MEDIUM (duplicate IDs silently take first) |
| `queries.rs:203, 209` | `labels(n)[0]` — primary label index | Configurable or documented | MEDIUM (assumes one-label-per-node) |
| `queries.rs:267, 290` | `ORDER BY count DESC` | Configurable | LOW |
| `queries.rs:41-50` | Property type coercion order: String → i64 → f64 → bool → Vec<String> → Null | Configurable or documented | LOW–MEDIUM |

#### colossus-pdf

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `classifier.rs:21` | `TEXT_CHAR_THRESHOLD: usize = 50` | Configurable | MEDIUM (sparse documents misclassified) |
| `text_search.rs:34, 36` | `context_chars: 50`, `max_results: 0` (unlimited) | Configurable via `SearchConfig` | LOW |
| `plain_text_extractor.rs:26` | `FORM_FEED: char = '\x0C'` | Standard | LOW |
| `docx_extractor.rs:52` | `PARAGRAPH_SEPARATOR: &str = "\n"` | Spec-driven | LOW |
| `docx_extractor.rs:55` | `SINGLE_PAGE_NUMBER: i32 = 1` | **All DOCX content mapped to page 1** | **CRITICAL** (page grounding broken for multi-page DOCX) |
| `format_detection.rs:34-52` | MIME constants (`application/pdf`, `application/vnd.openxmlformats-…`, `text/plain`, `txt`) | IANA standard | LOW |
| `normalize.rs:106, 113, 120` | Regex patterns `(?m)^(\d+\.)([A-Z])`, `\n{3,}`, `(?m)[ \t]+$` | Documented | LOW |

#### colossus-pipeline

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `worker/config.rs:22` | `DEFAULT_MAX_CONCURRENT = 4` | Configurable via env | LOW |
| `worker/config.rs:25, 28, 31, 35, 38, 41, 44` | Default poll/drain/heartbeat/zombie/recovery/cancel intervals | All env-configurable | LOW |
| `worker/config.rs:48` | `DEFAULT_NOTIFY_CHANNEL = "pipeline_jobs_changed"` | Must match migration 001 | LOW (well-coordinated) |
| `worker/retry.rs:20` | `JITTER_RANGE_PCT = 15` (±15%) | Nanosecond seed (no `rand`) | LOW |
| `events.rs:22` | `MAX_EVENT_MESSAGE_LEN = 500` | Bloat prevention | LOW |
| `scheduler.rs:35` | `PIPELINE_VERSION = 1` | Increment on schema change | LOW |
| `scheduler.rs:55` | `PG_UNIQUE_VIOLATION = "23505"` | PostgreSQL standard | LOW |
| `worker/mod.rs:54, 57, 62` | `CANCELLED_BY_USER`, `STEP_TIMED_OUT`, `WAITING_FOR_INPUT_STATUS` | String constants | LOW |
| `worker/fetcher.rs:37` | `CANCELLED_BY_USER = "Cancelled by user"` | Error message | LOW |
| `worker/executor.rs:93` | `if control == "cancel_requested"` | **MUST BE `JobControl::CancelRequested.as_str()`** per CLAUDE.md | **CRITICAL** |

#### colossus-rag

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `retriever.rs:95` | `default_score_threshold: f32` (injected) | Configurable | LOW |
| `reranker.rs:59` | `threshold: f32` (injected, "0.2–0.5" typical) | Configurable; no validation `0.0 ≤ x ≤ 1.0` | LOW |
| `reranker.rs:98` | Partition on `c.score > 0.0` | Magic — couples reranker to "graph nodes get 0.0" convention | MEDIUM (brittle) |
| `pipeline.rs:169` | `search_limit` default 10 | Configurable via builder | LOW |
| `expander_queries_minor.rs:127` | `LIMIT 20` in expand_document Cypher | Configurable | MEDIUM |
| `expander_queries_minor.rs:180` | `LIMIT 15` in expand_person | Configurable | MEDIUM |
| `expander_queries_minor.rs:235` | `LIMIT 15` in expand_organization | Configurable | MEDIUM |
| `graph_retriever.rs:61, 136, 218` | `LIMIT 20` in three document/person/contradiction queries | Configurable | MEDIUM |
| `assembler.rs:54` | `CHARS_PER_TOKEN: usize = 4` | Documented approximation | LOW |
| `assembler.rs:75-85` | `DEFAULT_SYSTEM_PROMPT` ~250 words hardcoded | Externalizable via `with_system_prompt` | LOW |
| `decomposer.rs:66-93` | `DEFAULT_SYSTEM_TEMPLATE` with Awad-vs-CFS persona | Configurable via param | **MEDIUM (domain leakage — case-specific text in domain-agnostic crate)** |
| `decomposer.rs:101-102` | `DEFAULT_USER_TEMPLATE = "{question}\n{strategy}"` | Configurable | LOW |
| `expander_queries.rs:90-105` | Hardcoded Cypher relationship types: `STATED_BY`, `ABOUT`, `CONTAINED_IN`, `CHARACTERIZES`, `REBUTS`, `CONTRADICTS` | Migrated from colossus-legal; marked "DO NOT MODIFY" | LOW (but **domain leakage** — these are legal-specific) |
| `expansion_category.rs:102-131` | Relationship whitelists per category | Hardcoded; must match Cypher | LOW |

#### spike

| FILE:line | Value | Should be | Risk |
|---|---|---|---|
| `tests/rig_spike.rs:276` | `"http://10.10.100.200:6333"` DEV Qdrant fallback | Env override exists | LOW (spike) |
| `tests/rig_spike.rs:278` | `:6333 → :6334` port replacement | Documented | LOW |
| `tests/rig_spike.rs:281` | `"colossus_evidence"` collection name | Spike test | LOW |
| `tests/rig_spike.rs:175, 195` | `768` embedding dimensionality | Nomic-embed-text V1.5 dimension | LOW |
| `tests/rig_spike.rs:533` | `"claude-haiku-4-5-20251001"` fallback model ID | Env override exists | LOW |
| `tests/structured_output_spike.rs:42, 106` | `"claude-sonnet-4-6"` hardcoded | Spike test | LOW |

---

## Section 4: Text Extraction Quality

### 4a. PDF Extraction

- **Library:** `pdf_oxide = "=0.3.8"` (`colossus-pdf/Cargo.toml:8`) — **exact-pinned** version, unusual for a library dependency.
- **Scanned vs native:** `extractor.rs:202-250 classify()` counts trimmed characters across all pages; threshold `TEXT_CHAR_THRESHOLD = 50` (`classifier.rs:21`). No distinction between native PDF text and any OCR output that pdf_oxide may produce internally.
- **Mixed content:** Per-page char count; pages below threshold flagged `needs_ocr: true`. `ContentType::Mixed` returned when both text and scanned pages exist.
- **Text normalization after extraction:** **None applied automatically**. Normalization (`normalize::normalize_text`) is opt-in; consumer must invoke it. Raw pdf_oxide output (including extra whitespace, missing spaces after paragraph numbers, ligatures, smart quotes) flows to the caller unless they explicitly normalize.
- **Known issues — gaps in normalization (`normalize.rs`):**
  - No rule for merged words (e.g., `wordone` from column extraction). **HIGH severity** for legal documents.
  - No rule for missing-space-after-punctuation (e.g., `U.S.A.recent`). MEDIUM.
  - No smart-quote/em-dash normalization in extraction path (only inside page_grounder's matcher at `page_grounder.rs:57-58`).
  - No ligature expansion in extraction (only in page_grounder at `page_grounder.rs:65-66`).
- **Page boundaries:** Preserved correctly — `extractor.rs:160-176 extract_all_pages()` returns `Vec<PageText>` with explicit page numbers.
- **Headers/footers:** Not specially handled; pdf_oxide returns all text on a page including running headers. Pollutes extracted text with "Page X of Y" etc. MEDIUM severity for legal documents.
- **pdf_oxide failure → fallback:** **None**. `extractor.rs:137-151 extract_page()` propagates the error; a single corrupt page halts extraction of the whole document. MEDIUM severity.

| FILE:line | Issue | Impact on extraction | Severity |
|---|---|---|---|
| `extractor.rs:137-143` | No fallback on per-page extraction failure | Single corrupt page breaks document | MEDIUM |
| `extractor.rs:202-250` | No way to know if pdf_oxide internally OCR'd or read native text | Cannot guide downstream confidence | HIGH |
| `extractor.rs` overall | Headers/footers included in output | Page text polluted with page numbers, running titles | MEDIUM |
| `normalize.rs` (gap) | Missing merged-word rule | `wordone` stays `wordone` | MEDIUM |
| `normalize.rs` (gap) | Missing post-punctuation rule | `U.S.A.recent` stays | MEDIUM |
| `normalize.rs` (gap) | Smart quotes / ligatures retained in extracted text | Extracted text differs from canonical | LOW |
| `extractor.rs` consumer contract | Normalization NOT auto-applied | Caller may forget → raw artifacts in LLM input | **HIGH** |
| `classifier.rs:21` | TEXT_CHAR_THRESHOLD=50 not configurable | Sparse docs misclassified | MEDIUM |

### 4b. DOCX Extraction

- **Library:** `docx-rust = "0.1.11"` (`colossus-pdf/Cargo.toml:15`) — **stale** (0.1.x line, ≈3 years unmaintained).
- **Paragraphs:** Preserved via `Paragraph.text()`, joined with single newline (`docx_extractor.rs:52, 88-95`).
- **Tables:** `BodyContent::Table` is **not matched** in the extraction loop (`docx_extractor.rs:89-93`). Tables silently dropped. Module comment at `docx_extractor.rs:17-22` says "will revisit." **HIGH severity** for legal documents.
- **Headers/footers/footnotes:** Not accessed; they live in separate parts of the OOXML package. Not extracted. **HIGH severity**.
- **Track changes / comments:** Not accessible via docx-rust's public API; not extracted. MEDIUM.
- **Complex formatting:** Bold, italics, hyperlinks are flattened by `Paragraph.text()`. Acceptable for LLM input.
- **Page numbering:** **CRITICAL** — `SINGLE_PAGE_NUMBER: i32 = 1` (`docx_extractor.rs:55`). All DOCX content is mapped to page 1 (`docx_extractor.rs:104`). Module comment (`docx_extractor.rs:6-13`) acknowledges that page breaks live inside Run elements as `<w:br w:type="page"/>` and aren't currently detected. Result: page-grounded snippet citations are **wrong** for any DOCX longer than one page.

| FILE:line | Issue | Impact | Severity |
|---|---|---|---|
| `docx_extractor.rs:55, 104` | All content → page 1 | Multi-page DOCX cannot be grounded; all snippets cite page 1 | **CRITICAL** |
| `docx_extractor.rs:89-93` | Tables silently skipped | Tabular data lost | HIGH |
| `docx_extractor.rs` (gap) | Headers/footers silently skipped | Section headers, doc titles lost | HIGH |
| `docx_extractor.rs` (gap) | Track changes/comments not extracted | Revision history lost | MEDIUM |
| Cargo.toml:15 | `docx-rust = "0.1.11"` stale | Bugs won't be fixed upstream | HIGH |

### 4c. Plain Text Extraction

- **Encoding:** `std::fs::read_to_string` (`plain_text_extractor.rs:47`) — UTF-8 only. Latin-1 or legacy-encoded files fail at read.
- **Line endings:** Splits on `FORM_FEED ('\x0C')` (`plain_text_extractor.rs:61`); does **not** normalize CRLF↔LF.
- **Binary detection:** None; relies on `read_to_string` returning a UTF-8 decode error.

| FILE:line | Issue | Impact | Severity |
|---|---|---|---|
| `plain_text_extractor.rs:47` | UTF-8 only | Non-UTF-8 files fail with `OpenError` | MEDIUM |
| `plain_text_extractor.rs:61` | No CRLF normalization | LF/CRLF mismatch with page grounder | LOW |

### 4d. Format Detection

- **Library:** `mimetype-detector = "0.3.8"` (`colossus-pdf/Cargo.toml:16`) — magic-byte based.
- **Detection flow:** `format_detection.rs:90-115 detect_format()`. MIME comparison strips charset param (`format_detection.rs:125`).
- **Wrong detection:** Explicit error (`format_detection.rs:147-152`) with path and detected MIME. `.txt` fallback path (`format_detection.rs:139-145`) when MIME differs but extension is `.txt` — emits `tracing::warn!` (visible, not silent).
- **Consumer override:** **None.** No API to override detection; caller must rewrite the file to force a format.

| FILE:line | Issue | Impact | Severity |
|---|---|---|---|
| `format_detection.rs:90-175` | No override hook | If detection misfires, caller has no escape hatch | LOW |

---

## Section 5: Text Normalization

`colossus-pdf::normalize` is the only normalization module in the workspace. It exposes 3 rules behind a `NormalizationRule` enum, applied via `normalize_text(text, &rules)`.

| FILE:line | Rule | Configurable | Risk of meaning change |
|---|---|---|---|
| `normalize.rs:50, 104-108` | `NumberedParagraphSpacing` — `(?m)^(\d+\.)([A-Z])` → `$1 $2` (inserts space after paragraph number) | YES (opt-in) | **LOW** — only matches digits.Uppercase pattern; doesn't touch decimals or lowercase |
| `normalize.rs:53, 111-115` | `CollapseBlankLines` — `\n{3,}` → `\n\n` | YES (opt-in) | **MEDIUM** — legal documents may use intentional multi-blank-line section separators; this collapses them |
| `normalize.rs:56, 118-122` | `TrimTrailingWhitespace` — `(?m)[ \t]+$` → `` | YES (opt-in) | LOW — trailing spaces have no semantic meaning |
| `normalize.rs` (gap) | Merged-word splitting | NO | HIGH (missing) |
| `normalize.rs` (gap) | Smart-quote normalization in extraction path | NO (only in page_grounder for matching) | MEDIUM |
| `normalize.rs` (gap) | Ligature expansion in extraction path | NO (only in page_grounder) | LOW |
| `normalize.rs` (gap) | Encoding normalization | NO | LOW |

Consumer can fully disable rules by passing `&[]` (verified in test at `normalize.rs:203-207`). Normalization is **NOT applied by extractors**; consumer must invoke it. (Restated from Section 4.)

A separate `normalize_text()` function exists at `colossus-pdf/src/page_grounder.rs:44-73` for grounding-time matching only — it removes invisible chars, rejoins hyphenated breaks, replaces pilcrow with space, normalizes smart quotes/dashes, expands `fi`/`fl` ligatures, collapses whitespace, and lowercases. This is **never written back to extracted text** — it's matcher-only. Good separation.

Other crates apply targeted normalization within their own domains (e.g., `colossus-extract::resolver::normalize_name` at `resolver.rs:217-261` strips corporate suffixes for fuzzy entity matching; `colossus-extract::splitter::adjust_end` walks to whitespace; `colossus-extract::merger::normalize_name` lowercases). None of these alter long-form document text.

---

## Section 6: Chunking / Splitting

The workspace has three splitting/grounding components, all in `colossus-extract` and `colossus-pdf`.

### `colossus-extract::FixedSizeSplitter` (`splitter.rs`)

- **Strategy:** Character-count with word-boundary walk-back (ASCII whitespace).
- **Defaults:** `chunk_size = 4000` chars (`splitter.rs:35`), `chunk_overlap = 200` chars (`splitter.rs:36`).
- **Boundary detection:** `adjust_end` (`splitter.rs:119-145`) walks backward to nearest ASCII whitespace.
- **Boundary-not-found fallback:** Returns original `approx_end` position (`splitter.rs:140-141`). Words longer than `chunk_size` will be split mid-word (documented at `splitter.rs:90-91`).
- **Configurable:** Yes (`FixedSizeSplitter::with_config`).
- **Quality:** GOOD.

### `colossus-extract::StructureAwareSplitter` (`structure_splitter.rs`)

- **Strategy:** Regex-boundary detection + grouping atomic units into chunks of N units.
- **Defaults:** Boundary `r"^\d+\.\s"` (line 25), 25 units/chunk (line 27), 0 unit overlap (line 30).
- **Boundary detection (line 103-106):** All regex matches; zero matches → fallback chunk; **invalid regex → silently returns empty Vec (`structure_splitter.rs:92-94`)**.
- **Atomic guarantee:** Units never split.
- **Fallback chunk:** When no matches, entire text is one chunk with metadata `{"fallback": true, "reason": "no_boundary_matches"}` (`structure_splitter.rs:246-253`). **However, this metadata is identical for "bad regex" and "regex matched zero times"**, defeating the audit's intent. This is the same silent-failure called out in Sections 1c and 2.
- **Unit overlap clamp:** `overlap >= units_per_chunk` clamped to no overlap (`structure_splitter.rs:168-172`). Prevents infinite loop.
- **Preamble handling:** Trimmed before prepending to every chunk (`structure_splitter.rs:174-176, 186-188`); metadata records `preamble_included`, `preamble_length_chars`.

### `colossus-pdf::PageGrounder` (`page_grounder.rs`)

- **Strategy:** Per-page exact match → per-page normalized match → adjacent-page-pair normalized match.
- **Boundary concatenation:** `format!("{} {}", page_a, page_b)` (`page_grounder.rs:237`) — simple space join. Assumes no mid-word split at page break; if a sentence is split mid-word, the break is preserved in the matched text.
- **Fallback:** `GroundingResult { page_number: None, match_type: NotFound }` plus `tracing::warn!` (`page_grounder.rs:262-265`).
- **Configurable sizes:** None — context sizes are in `text_search::SearchConfig` (separate module).

---

## Section 7: API Surface Quality

### 7a. Undocumented Public Items

Per-crate audit. Items listed here lack a struct/enum/trait-level `///` doc comment (field-level docs may exist).

| FILE:line | Item | Missing |
|---|---|---|
| `colossus-pdf/src/error.rs:5` | `PdfError` enum | top-level doc |
| `colossus-pdf/src/extractor.rs:18` | `PageText` struct | struct-level doc (fields documented) |
| `colossus-pdf/src/page_grounder.rs:16` | `GroundingResult` struct | struct-level doc |
| `colossus-pdf/src/page_grounder.rs:29` | `MatchType` enum | enum-level doc |
| `colossus-pdf/src/text_search.rs:10` | `SearchHit` struct | struct-level doc |
| `colossus-pdf/src/text_search.rs:30` | `SearchConfig` struct | struct-level doc |
| `colossus-extract/src/prompt.rs:42` | `PromptBuilder` struct | struct-level doc (methods documented) |

**All other public items across the workspace have `///` doc comments.** Notably, every public item in `colossus-auth`, `colossus-pipeline`, `colossus-rag`, `colossus-extract` (except PromptBuilder) is documented. `colossus-graph` is fully documented despite having no tests.

### 7b. Trait Design

| Crate | Trait | Object-safe | Send+Sync | Defaults | Quality |
|---|---|---|---|---|---|
| colossus-auth | (none custom; only Axum `FromRequestParts` impl for `AuthUser` at `extractor.rs:84`) | n/a | yes | n/a | GOOD |
| colossus-extract | `LlmProvider` (`traits.rs:36`) | yes (via `Arc<dyn …>`) | `Send + Sync + 'static` | `invoke_with_system` default | EXCELLENT |
| colossus-extract | `EmbeddingProvider` (`traits.rs:185`) | yes | `Send + Sync + 'static` | `embed_batch` serial default | EXCELLENT |
| colossus-extract | `EntityResolver` (`traits.rs:226`) | yes | `Send + Sync` | none | GOOD |
| colossus-extract | `TextSplitter` (`traits.rs:243`) | yes | `Send + Sync` | none | GOOD |
| colossus-graph | (none custom) | n/a | — | — | Crate couples API to `&neo4rs::Graph` directly — limits mocking |
| colossus-pdf | `DocumentExtractor` (`document_extractor.rs:45-70`) | yes | `Send + Sync` | none | GOOD; no capability-query method (e.g., `supports_multi_page`) so DOCX single-page limitation is not surfaceable |
| colossus-pipeline | `Step` (`step.rs:87`) | yes (via `async_trait`) | `Send + Sync + 'static` (`Serialize + Deserialize + Clone` also required) | `on_cancel`, `on_delete` defaults; associated consts `DEFAULT_RETRY_LIMIT`, `DEFAULT_RETRY_DELAY_SECS`, `DEFAULT_TIMEOUT_SECS`, `MAX_CONCURRENCY` | EXCELLENT |
| colossus-pipeline | `Task` (`task.rs:33`) | NO (generic associated type `Context`) | `Send + Sync + 'static` | none | GOOD (non-object-safety is intentional) |
| colossus-pipeline | `StepRecorder` (`recorder.rs:41`) | yes | `Send + Sync + 'static` | none | EXCELLENT |
| colossus-rag | `VectorRetriever` (`traits.rs:106`) | yes | `Send + Sync` | none | GOOD |
| colossus-rag | `GraphExpander` (`traits.rs:136`) | yes | `Send + Sync` | none | GOOD |
| colossus-rag | `ContextAssembler` (`traits.rs:169`) | yes | `Send + Sync` | none (sync trait — exception is documented) | GOOD |
| colossus-rag | `Synthesizer` (`traits.rs:188`) | yes | `Send + Sync` | none | GOOD |
| colossus-rag | `QueryRouter` (`traits.rs:66`) | yes | `Send + Sync` | none | GOOD |
| colossus-rag | `QueryDecomposer` (`traits.rs:85`) | yes | `Send + Sync` | none | GOOD |

**Step trait constraint not enforced:** CLAUDE.md rule G3 ("Steps must never call `tokio::spawn` internally") is documented in `step.rs:82` and `executor.rs:7` but not enforced at compile time. RISK: MEDIUM — relies on code review.

### 7c. Breaking Change Risk

| Concern | Location | Severity |
|---|---|---|
| `AuthError` public fields | `colossus-auth/error.rs:24-33` | LOW (serialized as JSON; intentional contract) |
| `AuthUser` public fields | `colossus-auth/extractor.rs:46-52` | LOW |
| `MeResponse` public fields | `colossus-auth/handler.rs:18-25` | LOW |
| `Permissions` public fields | `colossus-auth/permissions.rs:25-31` | LOW |
| `colossus-graph::GraphAccessError` collapses neo4rs error type | `error.rs:18-22` | MEDIUM (fixing breaks consumers expecting `QueryFailed`) |
| `colossus-graph` silent-failure shapes | `queries.rs:35-54, 64, 69, 138-142` | HIGH (fixing → error-typed return is breaking) |
| `colossus-graph` `LIMIT 1` semantics | `queries.rs:90` | MEDIUM (consumers may rely on "take first") |
| `colossus-graph` Cypher injection whitelist | `queries.rs:230-259` | MEDIUM (relaxing the whitelist is observable) |
| `colossus-graph` label index assumption | `queries.rs:203, 209` (`labels(n)[0]`) | MEDIUM |
| `colossus-pdf` direct `&neo4rs::Graph`-style coupling: API takes `pdf_oxide::PdfDocument` indirectly | extractor.rs:14 | LOW (already wrapped) |
| `colossus-pdf` `PdfError` `Io(#[from] …)` | error.rs:20 | LOW |
| `colossus-pdf` DOCX page=1 → real pagination | docx_extractor.rs:55, 104 | HIGH (correct fix changes `ExtractedPage::page_number` for every multi-page DOCX consumer) |
| `colossus-pipeline::PipelineError::LlmProvider` removal | error.rs:55 | LOW (variant currently never constructed) |
| `colossus-pipeline::Step` trait additions of required methods | `step.rs:87` | HIGH (breaks all implementations); defaults mitigate |
| `colossus-rag::RagError` enum additions | `error.rs:34-83` | LOW (additive); fields are non-exhaustive-friendly |
| `colossus-rag` Cypher relationship constants | `expander_queries.rs:90-105` | LOW (intentional, marked "DO NOT MODIFY") |
| `spike/Cargo.toml` workspace membership | root `Cargo.toml:2` | LOW (deletion is the documented next step) |

---

## Section 8: Dependency Health

Latest versions noted as "check needed" where the audit couldn't verify against crates.io live; otherwise sourced from the agents' analysis at audit time.

### Workspace (`Cargo.toml`)

| Dependency | Workspace version | Pinned | Status | Concern |
|---|---|---|---|---|
| axum | 0.7 | range | active | none |
| serde | 1 (derive) | range | active | none |
| serde_json | 1 | range | active | none |
| serde_yaml | 0.9 | range | **archived** | serde_yaml is **deprecated**; maintainer announced sunset in 2024. Workspace should migrate to `serde_yml` or `serde_yaml_ng`. **MEDIUM** |
| tracing | 0.1 | range | active | none |
| tokio | 1 (full) | range | active | full feature in workspace is broad; individual crates re-pin features tighter (good practice) |
| regex | 1 | range | active | none |

### colossus-auth

All deps via workspace; no concerns. `async-trait = "0.1"` is the only direct addition.

### colossus-extract

| Dep | Ver | Pinned | Status | Concern |
|---|---|---|---|---|
| async-trait | 0.1 | range | active | none |
| fastembed | 4 | range | active | none |
| regex | 1 | workspace | active | none |
| reqwest | 0.12 (json) | range | active | none |
| schemars | 1 | range | active | none |
| sha2 | 0.10 | range | active | none |
| strsim | 0.11 | range | active | none |
| thiserror | 2 | range | active | major-2 is stable; mitigates against future thiserror 3 break |
| tempfile (dev) | 3 | range | active | none |

### colossus-graph

| Dep | Ver | Pinned | Status | Concern |
|---|---|---|---|---|
| serde / serde_json | workspace | — | active | none |
| thiserror | 2 | range | active | none |
| tracing | workspace | — | active | none |
| neo4rs | 0.8 (optional, `neo4j` feature) | range | active | pin matches colossus-rag and colossus-legal for compat |
| tokio (dev) | workspace | — | active | none |

### colossus-pdf

| Dep | Ver | Pinned | Status | Concern |
|---|---|---|---|---|
| pdf_oxide | =0.3.8 | **EXACT** | active | **Unusual exact pin.** A library that pins exact won't get even patch fixes. Investigate motivation. MEDIUM |
| docx-rust | 0.1.11 | range | **stale (≈3 years)** | No table/header support; bugs won't be fixed. Consider migration to `docx` or `ooxml-rs`. HIGH |
| mimetype-detector | 0.3.8 | range | active | none |
| tempfile | 3 | range | active | none |
| thiserror | 2 | range | active | none |
| regex | workspace | — | active | none |
| serde / serde_json | workspace | — | active | none |
| tracing | workspace | — | active | none |

### colossus-pipeline

| Dep | Ver | Pinned | Status | Concern |
|---|---|---|---|---|
| async-trait | 0.1 | range | active | none |
| chrono | 0.4 (serde) | range | active | none |
| futures | 0.3 | range | active | none |
| serde / serde_json | workspace | — | active | none |
| thiserror | 2 | range | active | none |
| tokio | 1 (macros, sync, time, rt-multi-thread) | range | active | features curated (good) |
| tracing | workspace | — | active | none |
| uuid | 1 (v4, v7, serde) | range | active | v7 for time-ordering |
| sqlx | 0.8 (runtime-tokio-rustls, postgres, uuid, chrono, json, macros) | range | active | 0.8 mature; 0.9 in alpha |

### colossus-rag

| Dep | Ver | Pinned | Status | Concern |
|---|---|---|---|---|
| async-trait | 0.1 | range | active | none |
| colossus-extract | path | — | internal | tight coupling expected |
| serde / serde_json | workspace | — | active | none |
| thiserror | 2 | range | active | none |
| tokio | workspace | — | active | none |
| tracing | workspace | — | active | none |
| rig-core | 0.33 | range | active | imports `EmbeddingModel` trait; rig API churns. MEDIUM |
| qdrant-client | 1.16 (default-features=false, optional `qdrant`) | range | active | snapshots disabled (good) |
| neo4rs | 0.8 (optional `neo4j`) | range | active | matches workspace |
| fastembed (dev) | 4 | range | active | matches colossus-extract |

### spike

| Dep | Ver | Pinned | Status | Concern |
|---|---|---|---|---|
| rig-core | 0.33 | range | active | **Dual versions in tree**: rig-qdrant 0.1.37 and rig-fastembed 0.2.23 transitively pull rig-core 0.31, coexisting with 0.33 |
| rig-qdrant | 0.1 | range | active | pulls rig-core 0.31 |
| rig-fastembed | 0.2 | range | active | pulls rig-core 0.31 |
| qdrant-client | 1.14 (default-features=false) | range | active | **Cargo.toml spec is 1.14 but resolves to 1.17** transitively |
| schemars | 1 | range | active | none |
| tokio | workspace | — | active | none |

**Workspace-level concerns:**
- `serde_yaml = "0.9"` is archived by its author. **MEDIUM** — pick a fork.
- `pdf_oxide = "=0.3.8"` exact pin. Investigate.
- `docx-rust = "0.1.11"` stale and unmaintained. Consider alternatives.

---

## Section 9: Test Coverage

#### colossus-auth

- 23 inline unit tests across `extractor.rs` (9), `mode.rs` (4), `permissions.rs` (10). No `tests/` dir.
- **Gaps:**
  - `handler::me_handler` — not tested (would need Axum test harness)
  - `AuthUser::FromRequestParts::from_request_parts` — only helper functions tested
  - `error.rs`, `handler.rs`, `lib.rs` — no module tests (but contain only data structures / re-exports)
  - No integration tests; no property/fuzz tests

#### colossus-extract

- 14 integration test files in `tests/`, ~3625 lines total (per agent count).
- Inline tests in: `error.rs` (3), `resolver.rs` (14), `providers/anthropic.rs` (4), `providers/vllm.rs` (1), `providers/vllm_embed.rs` (7), `providers/factory.rs` (9).
- **Gaps:**
  - `structure_splitter.rs:92-94` invalid-regex silent-failure path — **not tested**.
  - HTTP timeout mapping not unit-tested (only via integration tests).
  - No property/fuzz tests.
  - No concurrent-provider stress tests.

#### colossus-graph

- **Zero tests.** No `tests/` directory; no inline `#[cfg(test)]` modules.
- Every public function — `get_node_by_id`, `get_node_neighbors`, `get_nodes_by_document`, `get_nodes_by_label`, `get_nodes_with_property`, `get_label_counts`, `get_relationship_type_counts`, `get_all_relationships_for_node` — has **no unit test**.
- Helper functions `extract_node_properties`, `row_to_graph_node` — no tests of type-coercion or empty-ID edge cases.
- **CRITICAL coverage gap** given Section 2 silent-failure findings in this crate.

#### colossus-pdf

- **No `tests/` directory.** Inline `#[cfg(test)]` modules totalling 46 tests.
- Per-file: classifier (5), document_extractor (0), docx_extractor (3), error (0), extractor (6), format_detection (5), normalize (8), page_grounder (8), plain_text_extractor (4), text_search (5), pdf_oxide_adapter (2).
- **Gaps:**
  - No cross-extractor integration tests (PDF → search → ground).
  - No corrupted-PDF / malformed-DOCX tests.
  - No property tests (e.g., "normalize_text is idempotent").
  - No DOCX limitation tests (tables, headers, multi-page).
  - Plain-text encoding tests cover only UTF-8.
  - Classifier inline tests don't exercise actual PDF classification.

#### colossus-pipeline

- 11 inline test modules: `events_tests.rs`, `step_tests.rs`, `recorder_tests.rs`, `scheduler_tests.rs`, `progress.rs` (inline tests at lines 107-158), `schema/tests.rs`, `worker/config_tests.rs`, `worker/retry.rs` (inline), `worker/heartbeat.rs` (inline, 2 ignored), `worker/executor_tests.rs`, `worker/recovery_tests.rs`, `worker/handler_tests.rs`, `worker/mod_tests.rs`, `worker/fetcher_tests.rs`, `worker/fetcher_recovery_tests.rs`.
- One integration test (`tests/integration.rs`) — full FSM lifecycle, requires `TEST_DATABASE_URL`.
- **Gaps:**
  - Recovery race-conditions (concurrent worker claims after zombie reset) — not stress-tested.
  - The hardcoded `"cancel_requested"` literal at `executor.rs:93` would not have been caught by tests if the enum drifted (no parallel string-equality test).

#### colossus-rag

- 9 integration test files in `tests/`. ~2100 lines.
- **Gaps:**
  - `graph_retriever.rs` (315 lines) — no dedicated tests.
  - `noop.rs` — no dedicated tests (covered indirectly).
  - `pipeline_helpers.rs` — no direct tests.
  - `expander_queries.rs` / `expander_queries_minor.rs` — tested only via mocked `Graph`.
  - `expansion_category.rs` — no direct tests.
  - Error-path coverage: `AssemblyError` is never constructed and not tested; `ExpandError` paths from Neo4j failure not fully tested.

#### spike

- 7 tests across `tests/rig_spike.rs` (5) and `tests/structured_output_spike.rs` (2). All require external services (Anthropic API, Qdrant). Two tests gracefully skip on missing `ANTHROPIC_API_KEY`; two others panic. By design (it's a spike).

---

## Section 10: Dead Code

#### colossus-auth

- No dead code, no TODOs, no commented-out blocks. All variants and fields are constructed and used.

#### colossus-extract

- `providers/vllm_embed.rs:67, 70, 93, 96` — `#[allow(dead_code)]` on `EmbeddingResponse::model`, `EmbeddingResponse::usage`, `EmbeddingUsage::prompt_tokens`, `EmbeddingUsage::total_tokens`. **Justified** — wire-format fields needed for deserialization but currently unused.
- No TODO/FIXME/HACK comments.
- No commented-out code blocks.

#### colossus-graph

- `error.rs:6-15` — `GraphAccessError::NodeNotFound` and `GraphAccessError::PropertyExtraction` variants are **defined but never constructed** in the crate. They are effectively dead since the code paths use `Ok(None)` and `Ok(Value::Null)` instead.
- No TODO/FIXME/HACK comments.

#### colossus-pdf

- `docx_extractor.rs:6-13, 17-22` — Module-level comments document follow-up work for DOCX page detection and table extraction ("will revisit"). Acceptable as inline documentation of known limits, but the issue is the **behavior**, not the comment.
- No `TODO`/`FIXME`/`HACK` *keyword* comments.
- No commented-out code blocks.

#### colossus-pipeline

- `error.rs:55` — `PipelineError::LlmProvider(String)` variant is defined but never constructed in this crate. It also violates the CLAUDE.md domain-agnostic invariant by name.
- `worker/mod.rs:236-239` — TODO(Phase 2): timeout_at not populated from `Step::DEFAULT_TIMEOUT_SECS` at claim time. Documented; feature gap.
- `worker/mod.rs:83` — `_phantom: PhantomData<T>` — intentional, marks Worker as logically generic over T.

#### colossus-rag

- `expander_queries.rs:48-49` and `expander_queries_minor.rs:20-22` — `map_neo4j_err` helper defined twice with identical bodies. Low-severity code smell; consolidate into a shared helper.
- All `pub` items are used (noop module is intentional Null Object pattern).
- No TODO/FIXME/HACK comments.

#### spike

- The **entire crate** is documented dead code by design:
  - `Cargo.toml:5` — "Throwaway spike"
  - `Cargo.toml:6` — `publish = false`
  - `Cargo.toml:8-9` — comment: "easy to delete later"
  - `src/lib.rs:9-10` — "**DO NOT** build production features on this crate"
- **Recommendation:** Delete `/spike/` directory and remove from `Cargo.toml` workspace members once results are archived.

---

## Section 11: Thread Safety

#### colossus-auth

- No `unsafe` blocks.
- All types are implicitly `Send + Sync` (strings, vecs, Option, bool, enum of unit variants).
- `impl FromRequestParts<S> for AuthUser` requires `S: Send + Sync` (`extractor.rs:84-86`).
- No interior mutability.

#### colossus-extract

- No `unsafe` blocks.
- All trait objects (`LlmProvider`, `EmbeddingProvider`, `EntityResolver`, `TextSplitter`) require `Send + Sync` (+ `'static` for the provider traits).
- All implementations have `Send + Sync` fields (`reqwest::Client`, `Arc<TextEmbedding>`, owned primitive config).
- **`PromptBuilder` cache is not behind a lock** (`prompt.rs:42-54`): `HashMap<String, String>` accessed via `&mut self`. Currently safe because PromptBuilder is constructed locally per-task and not stored as a trait object. **Footgun** if someone wraps it in `Arc<PromptBuilder>` or makes it a trait object. MEDIUM.

#### colossus-graph

- No `unsafe` blocks.
- All types (`GraphAccessError`, `GraphNode`, `GraphRelationship`, `NodeNeighborhood`, `LabelCount`) are `Send + Sync` via owned components.
- Query functions take `&neo4rs::Graph` (the Bolt driver maintains its own connection pool internally; documented thread-safe).

#### colossus-pdf

- No `unsafe` blocks.
- `DocumentExtractor` trait requires `Send + Sync` (`document_extractor.rs:60`); stateless implementations (`PdfOxideAdapter`, `DocxExtractor`, `PlainTextExtractor`) inherit those bounds.
- **`PdfTextExtractor` is NOT `Send + Sync`** — contains `PdfDocument` and `Vec<Option<String>>` mutable cache. By design: one extractor per document per thread/task. Acceptable.
- `PageGrounder<'a>` borrows `PdfTextExtractor` mutably; cannot be shared across threads. Acceptable.

#### colossus-pipeline

- No `unsafe` blocks outside of test code.
  - `worker/config_tests.rs:37, 55, 63, 71, 89, 97` — `unsafe std::env::set_var` in tests; intentional, test-only.
- `Step` trait requires `Send + Sync + 'static`; enforced.
- `Task::Context` requires `Send + Sync + 'static`; enforced.
- `tokio::spawn` calls in `worker/mod.rs:246` (job execution) and `worker/heartbeat.rs:40` (heartbeat) are correct — they are Worker-side, not Step-side, per the CLAUDE.md G3 rule.
- **CLAUDE.md G3 constraint** (Steps must not call `tokio::spawn` internally): documented at `step.rs:82` and `executor.rs:7`, **not compile-time enforced**. MEDIUM risk; mitigated by code review.
- `CancellationToken` uses `Arc<AtomicBool>` with `SeqCst` ordering (`cancel.rs:39-47`). Correct.
- `ProgressReporter.step_result: std::sync::Mutex<serde_json::Value>` (`progress.rs:39`) — non-async lock held only across non-await sections; correct.
- Heartbeat self-stops on `rows_affected == 0` (`heartbeat.rs:54-71`).
- Recovery race (concurrent claim after zombie reset, `recovery.rs:34-39`): benign — zombie reset puts `wakeup_at` in future, so race window is small and the second claim simply finds the job in Ready state.

#### colossus-rag

- **No `unsafe` blocks anywhere** in 5,000+ lines.
- All async traits (`QueryRouter`, `QueryDecomposer`, `VectorRetriever`, `GraphExpander`, `ContextAssembler`, `Synthesizer`) require `Send + Sync`.
- Implementations use `Arc<dyn EmbeddingProvider>`, `Arc<Graph>`, `Arc<Qdrant>` for shared state.
- Pipeline execution is sequential by data dependency; no concurrent access to mutable state.

#### spike

- No `unsafe`. No threads spawned beyond `#[tokio::test]`. Each test owns its client instances.

---

## Appendix A: Complete File List Audited

```
Cargo.toml (workspace)

colossus-auth/Cargo.toml
colossus-auth/src/error.rs
colossus-auth/src/extractor.rs
colossus-auth/src/handler.rs
colossus-auth/src/lib.rs
colossus-auth/src/mode.rs
colossus-auth/src/permissions.rs

colossus-extract/Cargo.toml
colossus-extract/src/config.rs
colossus-extract/src/error.rs
colossus-extract/src/lib.rs
colossus-extract/src/merger.rs
colossus-extract/src/prompt.rs
colossus-extract/src/providers/anthropic.rs
colossus-extract/src/providers/factory.rs
colossus-extract/src/providers/fastembed.rs
colossus-extract/src/providers/mod.rs
colossus-extract/src/providers/vllm.rs
colossus-extract/src/providers/vllm_embed.rs
colossus-extract/src/resolver.rs
colossus-extract/src/schema.rs
colossus-extract/src/splitter.rs
colossus-extract/src/structure_splitter.rs
colossus-extract/src/traits.rs
colossus-extract/src/types.rs
colossus-extract/tests/anthropic_live.rs
colossus-extract/tests/anthropic_provider_tests.rs
colossus-extract/tests/config_tests.rs
colossus-extract/tests/factory_tests.rs
colossus-extract/tests/fastembed_provider_tests.rs
colossus-extract/tests/merger_tests.rs
colossus-extract/tests/prompt_tests.rs
colossus-extract/tests/schema_tests.rs
colossus-extract/tests/splitter_tests.rs
colossus-extract/tests/structure_splitter_tests.rs
colossus-extract/tests/trait_tests.rs
colossus-extract/tests/types_tests.rs
colossus-extract/tests/vllm_embedding_provider_tests.rs
colossus-extract/tests/vllm_provider_tests.rs

colossus-graph/Cargo.toml
colossus-graph/src/error.rs
colossus-graph/src/lib.rs
colossus-graph/src/queries.rs
colossus-graph/src/types.rs

colossus-pdf/Cargo.toml
colossus-pdf/src/classifier.rs
colossus-pdf/src/document_extractor.rs
colossus-pdf/src/docx_extractor.rs
colossus-pdf/src/error.rs
colossus-pdf/src/extractor.rs
colossus-pdf/src/format_detection.rs
colossus-pdf/src/lib.rs
colossus-pdf/src/normalize.rs
colossus-pdf/src/page_grounder.rs
colossus-pdf/src/pdf_oxide_adapter.rs
colossus-pdf/src/plain_text_extractor.rs
colossus-pdf/src/text_search.rs

colossus-pipeline/Cargo.toml
colossus-pipeline/src/cancel.rs
colossus-pipeline/src/error.rs
colossus-pipeline/src/events.rs
colossus-pipeline/src/events_tests.rs
colossus-pipeline/src/lib.rs
colossus-pipeline/src/progress.rs
colossus-pipeline/src/recorder.rs
colossus-pipeline/src/recorder_tests.rs
colossus-pipeline/src/scheduler.rs
colossus-pipeline/src/scheduler_tests.rs
colossus-pipeline/src/schema.rs
colossus-pipeline/src/schema/tests.rs
colossus-pipeline/src/step.rs
colossus-pipeline/src/step_tests.rs
colossus-pipeline/src/task.rs
colossus-pipeline/src/worker/config.rs
colossus-pipeline/src/worker/config_tests.rs
colossus-pipeline/src/worker/executor.rs
colossus-pipeline/src/worker/executor_tests.rs
colossus-pipeline/src/worker/fetcher.rs
colossus-pipeline/src/worker/fetcher_api.rs
colossus-pipeline/src/worker/fetcher_recovery.rs
colossus-pipeline/src/worker/fetcher_recovery_tests.rs
colossus-pipeline/src/worker/fetcher_tests.rs
colossus-pipeline/src/worker/handler.rs
colossus-pipeline/src/worker/handler_tests.rs
colossus-pipeline/src/worker/heartbeat.rs
colossus-pipeline/src/worker/mod.rs
colossus-pipeline/src/worker/mod_tests.rs
colossus-pipeline/src/worker/recovery.rs
colossus-pipeline/src/worker/recovery_tests.rs
colossus-pipeline/src/worker/retry.rs
colossus-pipeline/tests/integration.rs

colossus-rag/Cargo.toml
colossus-rag/src/assembler.rs
colossus-rag/src/decomposer.rs
colossus-rag/src/error.rs
colossus-rag/src/expander.rs
colossus-rag/src/expander_queries.rs
colossus-rag/src/expander_queries_minor.rs
colossus-rag/src/expansion_category.rs
colossus-rag/src/graph_retriever.rs
colossus-rag/src/lib.rs
colossus-rag/src/noop.rs
colossus-rag/src/pipeline.rs
colossus-rag/src/pipeline_helpers.rs
colossus-rag/src/reranker.rs
colossus-rag/src/retriever.rs
colossus-rag/src/router.rs
colossus-rag/src/synthesizer.rs
colossus-rag/src/traits.rs
colossus-rag/src/types.rs
colossus-rag/tests/assembler_tests.rs
colossus-rag/tests/decomposer_tests.rs
colossus-rag/tests/expander_tests.rs
colossus-rag/tests/pipeline_tests.rs
colossus-rag/tests/reranker_tests.rs
colossus-rag/tests/retriever_tests.rs
colossus-rag/tests/router_tests.rs
colossus-rag/tests/synthesizer_tests.rs
colossus-rag/tests/types_tests.rs

spike/Cargo.toml
spike/src/lib.rs
spike/tests/rig_spike.rs
spike/tests/structured_output_spike.rs
```

---

## Appendix B: Issue Index by Severity

### Critical

1. `colossus-pipeline/src/worker/executor.rs:93` — Hardcoded `"cancel_requested"` string literal; violates CLAUDE.md JobControl::as_str() invariant. [Section 1c, 3]
2. `colossus-pdf/src/docx_extractor.rs:55, 104` — `SINGLE_PAGE_NUMBER = 1`; all DOCX content mapped to page 1. [Section 3, 4b]
3. `colossus-extract/src/structure_splitter.rs:92-94` — Invalid regex silently becomes empty Vec; indistinguishable from zero-match. [Section 1b, 1c, 2]
4. `colossus-graph/src/queries.rs:35-54, 64, 69, 138-142` — Property type fallback to `Null`, empty `id`-string default, and empty-`id` neighbor skip together silently corrupt graph topology. [Section 2]

### High

5. `colossus-graph/src/error.rs:5-15` — `NodeNotFound` and `PropertyExtraction` variants defined but never constructed; library returns `Ok` with empty/null instead of typed errors. [Section 1a, 10]
6. `colossus-pdf/src/docx_extractor.rs:89-93` — Tables silently skipped. [Section 2, 4b]
7. `colossus-pdf/src/docx_extractor.rs` (gap) — Headers/footers not extracted. [Section 4b]
8. `colossus-pdf/Cargo.toml:15` — `docx-rust = "0.1.11"` stale/unmaintained. [Section 8]
9. `colossus-pdf` — Text normalization not auto-applied; consumer must invoke explicitly. [Section 4a]
10. `colossus-pdf/src/normalize.rs` (gap) — Missing merged-word rule. [Section 4a, 5]
11. `colossus-pdf/src/extractor.rs:202-250` — No way to know if pdf_oxide returned native text vs internal OCR. [Section 4a]
12. `colossus-extract/src/structure_splitter.rs:99-101` — Response-marker regex error silently ignored. [Section 1b, 2]
13. `colossus-rag/src/retriever.rs:240-254` — Person/Collection filters matched but not applied to Qdrant query. [Section 2]
14. `colossus-pipeline/src/error.rs:55` — `LlmProvider(String)` variant violates domain-agnostic invariant. [Section 1a, 10]
15. `colossus-pipeline/src/step.rs:82, worker/executor.rs:7` — `tokio::spawn` constraint not compile-time enforced. [Section 11]
16. `colossus-pipeline` — `Step::DEFAULT_TIMEOUT_SECS` not auto-applied at claim time (TODO Phase 2). [Section 10]
17. `colossus-pdf/Cargo.toml:8` — `pdf_oxide = "=0.3.8"` exact-pinned. [Section 8]

### Medium

18. `colossus-auth/src/extractor.rs:64` — Anonymous user gets admin in Optional mode. [Section 3]
19. `colossus-graph/src/error.rs:18-22` — `From<neo4rs::Error>` loses structured error type. [Section 1b]
20. `colossus-graph/src/queries.rs:90` — Hardcoded `LIMIT 1` in `get_node_by_id`. [Section 3]
21. `colossus-graph/src/queries.rs:203, 209` — `labels(n)[0]` assumes one label per node. [Section 3]
22. `colossus-graph/src/queries.rs:151-152, 273-274, 296-297, 322-325` — `unwrap_or(true)` on `outgoing` reverses relationship direction silently. [Section 2]
23. `colossus-pdf/src/extractor.rs:154` — `unwrap_or_default()` on page_cache masks logic errors. [Section 2]
24. `colossus-pdf/src/classifier.rs:21` — `TEXT_CHAR_THRESHOLD = 50` not configurable. [Section 3]
25. `colossus-pdf/src/normalize.rs:53, 111-115` — `CollapseBlankLines` may lose intentional section separators. [Section 5]
26. `colossus-pdf/src/extractor.rs:137-143` — No fallback for per-page extraction failure. [Section 4a]
27. `colossus-pdf/src/extractor.rs` overall — Headers/footers included in extracted text. [Section 4a]
28. `colossus-pdf/src/plain_text_extractor.rs:47` — UTF-8 only. [Section 4c]
29. `colossus-pdf/src/normalize.rs` (gap) — No smart-quote normalization in extraction path. [Section 4a, 5]
30. `colossus-extract/src/prompt.rs:42-54` — `PromptBuilder` cache not thread-safe. [Section 11]
31. `colossus-extract/src/resolver.rs:38` — Fuzzy threshold not in constructor. [Section 3]
32. `colossus-extract/src/providers/factory.rs:83` — `32_000` default LLM_MAX_TOKENS. [Section 3]
33. `colossus-pipeline/src/progress.rs:56-65` — Progress write errors swallowed by design. [Section 2]
34. `colossus-pipeline/src/worker/mod.rs:315-339, 385-392` — Event/recorder failures warn-only by design. [Section 2]
35. `colossus-pipeline/src/worker/heartbeat.rs:54-71` — Heartbeat DB errors warn-only. [Section 2]
36. `colossus-rag/src/expander_queries_minor.rs:127, 180, 235` — Hardcoded Cypher `LIMIT 20/15/15` not configurable. [Section 3]
37. `colossus-rag/src/graph_retriever.rs:61, 136, 218` — Three more hardcoded `LIMIT 20` Cypher constants. [Section 3]
38. `colossus-rag/src/reranker.rs:98` — Partition on `c.score > 0.0` couples to "graph nodes = 0.0" convention. [Section 3]
39. `colossus-rag/src/expander.rs:212-234` — Per-seed expansion failure not propagated. [Section 2]
40. `colossus-rag/src/synthesizer.rs:161-165` — `.unwrap_or(0)` on missing token counts. [Section 2]
41. `colossus-rag/src/pipeline_helpers.rs:135-139` — Graph sub-queries silently empty when `neo4j` feature disabled. [Section 2]
42. `colossus-rag/src/decomposer.rs:66-93` — `DEFAULT_SYSTEM_TEMPLATE` contains Awad-vs-CFS persona (domain leakage). [Section 3]
43. `colossus-rag/src/expander_queries.rs:90-105` — Hardcoded legal-specific relationship types in domain-agnostic crate. [Section 3]
44. Workspace `Cargo.toml:9` — `serde_yaml = "0.9"` is archived. [Section 8]
45. `colossus-graph` — Direct coupling to `&neo4rs::Graph` API; limits testability. [Section 7b]
46. `colossus-rag/src/graph_retriever.rs` — 315 lines, zero dedicated tests. [Section 9]
47. `colossus-pdf/src/extractor.rs:54-60, 92-93` — `OpenError` variants lose file path. [Section 1b]
48. `colossus-extract/src/providers/{anthropic,vllm,vllm_embed}.rs` — `600s` default timeouts everywhere; overridable but homogeneous. [Section 3]

### Low (selected — others enumerated in body)

49. `colossus-auth/src/error.rs:22-34`, `extractor.rs:46-52`, `handler.rs:18-25`, `permissions.rs:25-31` — Public fields on AuthError / AuthUser / MeResponse / Permissions. [Section 7c]
50. `colossus-rag/src/expander_queries.rs:48-49` vs `expander_queries_minor.rs:20-22` — `map_neo4j_err` defined twice. [Section 1b, 10]
51. `colossus-pdf` — `PdfError`, `PageText`, `GroundingResult`, `MatchType`, `SearchHit`, `SearchConfig` lack struct-level doc comments. [Section 7a]
52. `colossus-extract/src/prompt.rs:42` — `PromptBuilder` lacks struct-level doc comment. [Section 7a]
53. `spike/` entire crate — flagged as dead-code-by-design; should be removed once results archived. [Section 10]
54. `spike/Cargo.toml:15` — `qdrant-client = "1.14"` resolves to 1.17 transitively; spec is stale. [Section 8]
55. `spike` — dual rig-core versions (0.31 transitively, 0.33 directly) in dependency tree. [Section 8]
56. `colossus-pdf/src/error.rs:20` — `Io(#[from] std::io::Error)` is transparent; path/operation context lost. [Section 1b]

---

## Appendix C: Issue Index by File

### colossus-auth/

- `src/extractor.rs:64` — Anonymous user gets admin in Optional mode [Section 2, 3]
- `src/extractor.rs:101, 103, 177` — Documented-fallback unwrap_or patterns [Section 2]
- `src/error.rs:24-33`, `extractor.rs:46-52`, `handler.rs:18-25`, `permissions.rs:25-31` — Public fields (intentional contract) [Section 7c]

### colossus-extract/

- `src/structure_splitter.rs:92-94` — **Critical** invalid regex silently empty [Section 1b, 1c, 2]
- `src/structure_splitter.rs:99-101` — Response-marker regex error silently None [Section 1b, 2]
- `src/prompt.rs:42-54` — PromptBuilder cache not thread-safe (footgun) [Section 11]
- `src/prompt.rs:42` — Struct lacks doc comment [Section 7a]
- `src/resolver.rs:38` — Fuzzy threshold not constructor param [Section 3]
- `src/providers/{anthropic,vllm,vllm_embed}.rs` — `600s` default timeouts (configurable) [Section 3]
- `src/providers/factory.rs:83` — `32_000` LLM_MAX_TOKENS default [Section 3]

### colossus-graph/

- `src/error.rs:5-15` — Unused `NodeNotFound`, `PropertyExtraction` variants [Section 1a, 10]
- `src/error.rs:18-22` — `From<neo4rs::Error>` collapses structured errors [Section 1b]
- `src/queries.rs:35-54` — **Critical** property type → `Null` fallback [Section 2]
- `src/queries.rs:64` — Missing `labels` → empty Vec silently [Section 2]
- `src/queries.rs:69` — `unwrap_or("")` on `id` extraction [Section 2]
- `src/queries.rs:138-142` — Empty-`m_id` neighbor silently skipped [Section 2]
- `src/queries.rs:151-152, 273-274, 296-297, 322-325` — `unwrap_or` defaults on extracted fields [Section 2]
- `src/queries.rs:90` — Hardcoded `LIMIT 1` [Section 3]
- `src/queries.rs:203, 209` — `labels(n)[0]` assumption [Section 3]
- **Zero unit tests** across the crate [Section 9]

### colossus-pdf/

- `src/docx_extractor.rs:55, 104` — **Critical** all content → page 1 [Section 3, 4b]
- `src/docx_extractor.rs:89-93` — Tables silently skipped [Section 2, 4b]
- `src/docx_extractor.rs` — Headers/footers not extracted [Section 4b]
- `src/extractor.rs:54-60, 92-93` — `OpenError` lacks file path [Section 1b]
- `src/extractor.rs:137-143` — No fallback for per-page failure [Section 4a]
- `src/extractor.rs:154` — `unwrap_or_default()` on page cache [Section 2]
- `src/extractor.rs:202-250` — Cannot distinguish native text from internal OCR [Section 4a]
- `src/classifier.rs:21` — `TEXT_CHAR_THRESHOLD = 50` not configurable [Section 3]
- `src/plain_text_extractor.rs:47` — UTF-8 only [Section 4c]
- `src/normalize.rs` (gap) — Missing merged-word, post-punctuation, smart-quote rules in extraction path [Section 5]
- `src/error.rs:5, extractor.rs:18, page_grounder.rs:16, 29, text_search.rs:10, 30` — Undocumented public items [Section 7a]
- `Cargo.toml:8, 15` — `pdf_oxide` exact-pin; `docx-rust` stale [Section 8]
- **No `tests/` directory** [Section 9]

### colossus-pipeline/

- `src/worker/executor.rs:93` — **Critical** hardcoded `"cancel_requested"` literal [Section 1c, 3]
- `src/error.rs:55` — `LlmProvider` variant never constructed; domain-name violation [Section 1a, 10]
- `src/worker/mod.rs:236-239` — TODO: timeout_at not auto-populated [Section 10]
- `src/step.rs:82`, `src/worker/executor.rs:7` — `tokio::spawn` constraint not compile-time enforced [Section 7b, 11]
- `src/progress.rs:56-65`, `worker/mod.rs:315-339, 385-392`, `worker/heartbeat.rs:54-71` — Intentional warn-only swallowing [Section 2]

### colossus-rag/

- `src/retriever.rs:240-254` — Person/Collection filters silently dropped [Section 2]
- `src/expander.rs:212-234` — Per-seed expansion error not propagated [Section 2]
- `src/synthesizer.rs:161-165` — Token-count `unwrap_or(0)` [Section 2]
- `src/pipeline_helpers.rs:135-139` — Graph sub-queries silently empty without `neo4j` feature [Section 2]
- `src/expander_queries_minor.rs:127, 180, 235`, `graph_retriever.rs:61, 136, 218` — Hardcoded Cypher `LIMIT` constants [Section 3]
- `src/reranker.rs:98` — Partition on `c.score > 0.0` brittle [Section 3]
- `src/decomposer.rs:66-93` — Legal-specific persona in domain-agnostic default [Section 3]
- `src/expander_queries.rs:90-105` — Legal-specific Cypher relationship types [Section 3]
- `src/expander_queries.rs:48-49` and `expander_queries_minor.rs:20-22` — `map_neo4j_err` duplicated [Section 10]
- `src/graph_retriever.rs` — 315 lines, zero dedicated tests [Section 9]

### spike/

- Entire crate is documented dead code; should be deleted [Section 10]
- Dual rig-core versions in dep tree [Section 8]
- `qdrant-client = "1.14"` Cargo spec stale (resolves to 1.17) [Section 8]

### Workspace

- `Cargo.toml:9` — `serde_yaml = "0.9"` archived [Section 8]

---

*End of audit.*
