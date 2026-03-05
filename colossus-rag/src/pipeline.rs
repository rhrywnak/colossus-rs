//! RagPipeline — wires all five components into a single `ask()` method.
//!
//! This is the milestone module: it connects Router → Retriever → Expander →
//! Assembler → Synthesizer into one cohesive pipeline with a builder pattern.
//!
//! ## Why a builder pattern?
//!
//! The pipeline requires 5 components, each as a trait object (`Box<dyn Trait>`).
//! A constructor with 5 parameters would be unwieldy and hard to read:
//!
//! ```text
//! // ❌ Hard to read — which parameter is which?
//! RagPipeline::new(router, retriever, expander, assembler, synthesizer)
//!
//! // ✅ Self-documenting — each component is named
//! RagPipeline::builder()
//!     .router(router)
//!     .retriever(retriever)
//!     .expander(expander)
//!     .assembler(assembler)
//!     .synthesizer(synthesizer)
//!     .build()?
//! ```
//!
//! The builder also lets us provide defaults (e.g., `NoOpExpander` if no
//! graph database is available) and validate that all required components
//! are set before building.
//!
//! ## Rust Learning: Trait objects (`Box<dyn Trait>`)
//!
//! Each pipeline component is stored as `Box<dyn Trait>` — a heap-allocated
//! trait object. This enables runtime polymorphism: the pipeline doesn't know
//! (or care) whether it's using a `RuleBasedRouter` or a future `LlmRouter`.
//!
//! The tradeoff vs generics:
//! - **Generics** (`RagPipeline<R: QueryRouter>`) → zero-cost, but the concrete
//!   type must be known at compile time. Different routers = different pipeline types.
//! - **Trait objects** (`Box<dyn QueryRouter>`) → tiny heap allocation + vtable
//!   lookup per call, but any router works at runtime. One pipeline type for all.
//!
//! We use trait objects because the pipeline is constructed at startup and
//! called many times — the one-time allocation cost is negligible, and the
//! flexibility to swap components at runtime (e.g., in tests) is valuable.
//!
//! ## Pipeline orchestration: why sequential, not parallel?
//!
//! Each stage depends on the previous stage's output:
//! ```text
//! route(question) → strategy     (need strategy for search params)
//! search(question, params) → chunks   (need chunks for expansion seeds)
//! expand(seed_ids) → more chunks      (need all chunks for assembly)
//! assemble(question, chunks) → context (need context for synthesis)
//! synthesize(context) → answer         (final output)
//! ```
//!
//! There's no opportunity for parallelism — it's a strict data pipeline.
//! The only potential parallel step would be running retriever + expander
//! simultaneously if the expander used separate seed IDs, but our design
//! feeds retriever results INTO the expander.

use std::collections::HashSet;
use std::time::Instant;

use crate::error::RagError;
use crate::noop::NoOpExpander;
use crate::traits::{ContextAssembler, GraphExpander, QueryRouter, Synthesizer, VectorRetriever};
use crate::types::{
    ContextChunk, PipelineStats, RagResult, RetrievalStrategy, ScopeFilter,
};

// ---------------------------------------------------------------------------
// RagPipeline — the assembled pipeline ready to answer questions
// ---------------------------------------------------------------------------

/// A fully-assembled RAG pipeline that can answer questions end-to-end.
///
/// Created via [`RagPipelineBuilder`] — all five components must be provided
/// (or defaulted) before `build()` succeeds.
///
/// ## Rust Learning: Ownership transfer via builder
///
/// The builder takes ownership of each `Box<dyn Trait>` via `self` methods.
/// When `build()` is called, it moves all components into this struct.
/// After that, the builder is consumed (can't be reused). This is Rust's
/// way of preventing "use after move" bugs — the compiler enforces it.
// ## Rust Learning: No `Debug` derive for trait objects
//
// We can't `#[derive(Debug)]` here because `Box<dyn QueryRouter>` doesn't
// implement `Debug` — the trait doesn't require it. Instead, we implement
// `Debug` manually with a simple placeholder. This satisfies the compiler
// for things like `Result::unwrap_err()` which needs `T: Debug`.
pub struct RagPipeline {
    router: Box<dyn QueryRouter>,
    retriever: Box<dyn VectorRetriever>,
    expander: Box<dyn GraphExpander>,
    assembler: Box<dyn ContextAssembler>,
    synthesizer: Box<dyn Synthesizer>,

    /// Maximum tokens for the assembled context window.
    max_context_tokens: usize,

    /// Maximum number of vector search results per query.
    search_limit: usize,
}

impl std::fmt::Debug for RagPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagPipeline")
            .field("max_context_tokens", &self.max_context_tokens)
            .field("search_limit", &self.search_limit)
            .finish_non_exhaustive()
    }
}

/// Builder for [`RagPipeline`] — set each component, then call `build()`.
///
/// ## Rust Learning: Option fields in builders
///
/// Each component starts as `None`. The setter methods replace `None` with
/// `Some(component)`. At `build()` time, we check that all required fields
/// are `Some` — if any are `None`, we return `RagError::ConfigError`.
///
/// This pattern prevents constructing an incomplete pipeline at compile time
/// (you can't call `ask()` on a builder) while giving clear error messages
/// at runtime if a component is missing.
pub struct RagPipelineBuilder {
    router: Option<Box<dyn QueryRouter>>,
    retriever: Option<Box<dyn VectorRetriever>>,
    expander: Option<Box<dyn GraphExpander>>,
    assembler: Option<Box<dyn ContextAssembler>>,
    synthesizer: Option<Box<dyn Synthesizer>>,
    max_context_tokens: usize,
    search_limit: usize,
}

impl RagPipeline {
    /// Start building a new pipeline.
    ///
    /// Returns a [`RagPipelineBuilder`] with all components unset and
    /// sensible defaults for `max_context_tokens` (6000) and `search_limit` (10).
    pub fn builder() -> RagPipelineBuilder {
        RagPipelineBuilder {
            router: None,
            retriever: None,
            expander: None,
            assembler: None,
            synthesizer: None,
            max_context_tokens: 6000,
            search_limit: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder methods — each takes `self` and returns `self` for chaining
// ---------------------------------------------------------------------------

/// ## Rust Learning: Builder method pattern (`self` → `Self`)
///
/// Each setter takes `mut self` (ownership) and returns `Self`. This enables
/// method chaining: `.router(r).retriever(ret).build()`. The `mut self`
/// means the builder is consumed and returned — you can't accidentally use
/// an intermediate state.
///
/// In Java, builders typically use `this` (mutable reference). In Rust, we
/// move ownership through the chain, which is both safe and ergonomic.
impl RagPipelineBuilder {
    /// Set the query router component.
    pub fn router(mut self, router: Box<dyn QueryRouter>) -> Self {
        self.router = Some(router);
        self
    }

    /// Set the vector retriever component.
    pub fn retriever(mut self, retriever: Box<dyn VectorRetriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the graph expander component (optional — defaults to NoOpExpander).
    pub fn expander(mut self, expander: Box<dyn GraphExpander>) -> Self {
        self.expander = Some(expander);
        self
    }

    /// Set the context assembler component.
    pub fn assembler(mut self, assembler: Box<dyn ContextAssembler>) -> Self {
        self.assembler = Some(assembler);
        self
    }

    /// Set the synthesizer component.
    pub fn synthesizer(mut self, synthesizer: Box<dyn Synthesizer>) -> Self {
        self.synthesizer = Some(synthesizer);
        self
    }

    /// Set the maximum token budget for the assembled context.
    ///
    /// Default: 6000 tokens. Increase for complex questions that need more
    /// context, decrease to reduce Claude API costs.
    pub fn max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = tokens;
        self
    }

    /// Set the maximum number of vector search results.
    ///
    /// Default: 10. The retriever will return at most this many chunks
    /// from Qdrant.
    pub fn search_limit(mut self, limit: usize) -> Self {
        self.search_limit = limit;
        self
    }

    /// Build the pipeline, consuming the builder.
    ///
    /// Returns `RagError::ConfigError` if any required component is missing.
    /// The expander defaults to [`NoOpExpander`] if not set.
    pub fn build(self) -> Result<RagPipeline, RagError> {
        let router = self.router.ok_or_else(|| {
            RagError::ConfigError("Pipeline requires a router".into())
        })?;
        let retriever = self.retriever.ok_or_else(|| {
            RagError::ConfigError("Pipeline requires a retriever".into())
        })?;
        let assembler = self.assembler.ok_or_else(|| {
            RagError::ConfigError("Pipeline requires an assembler".into())
        })?;
        let synthesizer = self.synthesizer.ok_or_else(|| {
            RagError::ConfigError("Pipeline requires a synthesizer".into())
        })?;

        // Expander is optional — default to NoOpExpander (returns empty vec).
        let expander = self
            .expander
            .unwrap_or_else(|| Box::new(NoOpExpander));

        Ok(RagPipeline {
            router,
            retriever,
            expander,
            assembler,
            synthesizer,
            max_context_tokens: self.max_context_tokens,
            search_limit: self.search_limit,
        })
    }
}

// ---------------------------------------------------------------------------
// Pipeline execution — the ask() method
// ---------------------------------------------------------------------------

impl RagPipeline {
    /// Ask a question and get an answer with citations.
    ///
    /// This is the main entry point. It orchestrates all five pipeline stages
    /// sequentially, collecting timing stats along the way.
    ///
    /// ## Error propagation across pipeline stages
    ///
    /// Each stage can fail with a different `RagError` variant. The `?` operator
    /// propagates errors immediately — if the router fails, we don't attempt
    /// retrieval. This is intentional: later stages depend on earlier ones,
    /// so a failure at any point means the pipeline cannot produce a valid result.
    ///
    /// ## Timing with `Instant::now()`
    ///
    /// We use `std::time::Instant` to measure wall-clock time for each stage.
    /// `Instant::now()` captures a monotonic timestamp (immune to clock adjustments),
    /// and `.elapsed()` returns the `Duration` since that timestamp.
    /// `.as_millis()` converts to milliseconds as `u128`, which we cast to `u64`.
    ///
    /// Total time is measured independently (not summed from stages) to capture
    /// any overhead between stages (logging, allocations, etc.).
    pub async fn ask(&self, question: &str) -> Result<RagResult, RagError> {
        let total_start = Instant::now();
        let mut stats = PipelineStats::default();

        // --- Stage 1: Route the question to a retrieval strategy ---
        let route_start = Instant::now();
        let strategy = self.router.route(question).await?;
        stats.route_ms = route_start.elapsed().as_millis() as u64;

        // Convert the strategy into search parameters (filters + limit).
        let (filters, limit) = strategy_to_search_params(&strategy, self.search_limit);
        stats.strategy = format_strategy_name(&strategy);

        tracing::info!(
            strategy = %stats.strategy,
            filters = filters.len(),
            limit = limit,
            "Pipeline: routed question"
        );

        // --- Stage 2: Vector search (embed + search Qdrant) ---
        //
        // The retriever handles both embedding and searching internally.
        // We measure them together as "search_ms" because the retriever
        // doesn't expose separate timing for embed vs search.
        let search_start = Instant::now();
        let mut chunks = self.retriever.search(question, limit, &filters).await?;
        stats.search_ms = search_start.elapsed().as_millis() as u64;
        stats.qdrant_hits = chunks.len();

        tracing::info!(
            hits = chunks.len(),
            search_ms = stats.search_ms,
            "Pipeline: vector search complete"
        );

        // --- Stage 3: Graph expansion (follow Neo4j relationships) ---
        //
        // Extract seed node IDs from the retriever results. The expander
        // will follow relationships from these nodes to find additional context.
        let seed_ids: Vec<String> = chunks.iter().map(|c| c.node_id.clone()).collect();

        let expand_start = Instant::now();
        let expanded = self.expander.expand(&seed_ids, 1).await?;
        stats.expand_ms = expand_start.elapsed().as_millis() as u64;
        stats.graph_nodes_expanded = expanded.len();

        tracing::info!(
            expanded = expanded.len(),
            expand_ms = stats.expand_ms,
            "Pipeline: graph expansion complete"
        );

        // Merge expanded chunks into the retriever results.
        merge_expansion(&mut chunks, expanded);

        // --- Stage 4: Assemble context (format chunks into prompt) ---
        //
        // The assembler is synchronous — no async needed for string formatting.
        // It sorts chunks by score (highest first) and formats them within
        // the token budget, dropping low-scored chunks if necessary.
        let assemble_start = Instant::now();
        let context = self.assembler.assemble(question, &chunks, self.max_context_tokens);
        stats.assemble_ms = assemble_start.elapsed().as_millis() as u64;
        stats.context_tokens_approx = context.token_estimate;

        tracing::info!(
            tokens = context.token_estimate,
            assemble_ms = stats.assemble_ms,
            "Pipeline: context assembled"
        );

        // --- Stage 5: Synthesize answer (call Claude API) ---
        let synth_start = Instant::now();
        let synthesis = self.synthesizer.synthesize(&context, question).await?;
        stats.synthesize_ms = synth_start.elapsed().as_millis() as u64;

        // Fill in LLM-reported stats from the synthesis result.
        stats.input_tokens = Some(synthesis.input_tokens);
        stats.output_tokens = Some(synthesis.output_tokens);
        stats.provider = synthesis.provider.clone();
        stats.model = synthesis.model.clone();

        // Total pipeline time (measured independently, not summed).
        stats.total_ms = total_start.elapsed().as_millis() as u64;

        tracing::info!(
            total_ms = stats.total_ms,
            input_tokens = synthesis.input_tokens,
            output_tokens = synthesis.output_tokens,
            "Pipeline: synthesis complete"
        );

        // --- Build final result ---
        Ok(RagResult {
            answer: synthesis.answer,
            strategy_used: strategy,
            chunks,
            citations: synthesis.citations,
            stats,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: convert RetrievalStrategy into search parameters
// ---------------------------------------------------------------------------

/// Convert a [`RetrievalStrategy`] into Qdrant search parameters.
///
/// Returns `(filters, limit)` where:
/// - `filters`: Scope filters to pass to the retriever (may be empty for Broad)
/// - `limit`: Maximum number of results to request from Qdrant
///
/// ## Strategy mapping
///
/// | Strategy | Filters | Limit |
/// |----------|---------|-------|
/// | Focused  | scope filters from router | configured limit |
/// | Broad    | none | configured limit |
/// | Hybrid   | all scope filters | configured limit |
/// | Direct   | none (falls back to Broad) | configured limit |
///
/// The `Direct` strategy is a placeholder for future graph-only queries.
/// For v1, we fall back to Broad search — this is safe because the
/// assembler and synthesizer will still produce a useful answer.
fn strategy_to_search_params(
    strategy: &RetrievalStrategy,
    default_limit: usize,
) -> (Vec<ScopeFilter>, usize) {
    match strategy {
        RetrievalStrategy::Focused { scope } => {
            (scope.clone(), default_limit)
        }
        RetrievalStrategy::Broad { .. } => {
            (Vec::new(), default_limit)
        }
        RetrievalStrategy::Hybrid { scopes, .. } => {
            (scopes.clone(), default_limit)
        }
        // Direct strategy: fall back to Broad for v1.
        RetrievalStrategy::Direct { .. } => {
            tracing::warn!("Direct strategy not yet implemented — falling back to Broad");
            (Vec::new(), default_limit)
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: merge expanded chunks into retriever results
// ---------------------------------------------------------------------------

/// Merge graph-expanded chunks into the retriever's vector search results.
///
/// Expanded chunks are appended after retriever chunks, with deduplication
/// by `node_id`. If an expanded chunk has the same `node_id` as a retriever
/// chunk, it's skipped (the retriever version has a real similarity score,
/// so it's more useful for ranking).
///
/// The assembler will sort all chunks by score anyway (retriever chunks
/// have score > 0, expanded chunks have score = 0.0), so the order of
/// appending doesn't matter for the final prompt.
fn merge_expansion(chunks: &mut Vec<ContextChunk>, expanded: Vec<ContextChunk>) {
    // Build a set of existing node IDs for O(1) dedup lookups.
    //
    // ## Rust Learning: Owned `String` to avoid borrow conflict
    //
    // We clone into `HashSet<String>` instead of borrowing `&str` because
    // we need to mutate `chunks` (push new items) while checking the set.
    // If we borrowed `&str` from `chunks`, the borrow checker would prevent
    // us from pushing — you can't have an immutable borrow (the &str refs)
    // and a mutable borrow (push) active at the same time.
    let existing_ids: HashSet<String> = chunks.iter().map(|c| c.node_id.clone()).collect();

    for chunk in expanded {
        if !existing_ids.contains(&chunk.node_id) {
            chunks.push(chunk);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: format strategy name for stats
// ---------------------------------------------------------------------------

/// Produce a human-readable name for the strategy (used in PipelineStats).
fn format_strategy_name(strategy: &RetrievalStrategy) -> String {
    match strategy {
        RetrievalStrategy::Focused { scope } => {
            let types: Vec<&str> = scope.iter().map(|s| match s.filter_type {
                crate::types::ScopeFilterType::Document => "document",
                crate::types::ScopeFilterType::Person => "person",
                crate::types::ScopeFilterType::NodeType => "node_type",
                crate::types::ScopeFilterType::Collection => "collection",
            }).collect();
            format!("Focused({})", types.join(", "))
        }
        RetrievalStrategy::Broad { .. } => "Broad".into(),
        RetrievalStrategy::Hybrid { scopes, .. } => {
            format!("Hybrid({} scopes)", scopes.len())
        }
        RetrievalStrategy::Direct { .. } => "Direct".into(),
    }
}
