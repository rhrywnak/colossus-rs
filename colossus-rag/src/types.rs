//! Core data types for the Colossus RAG pipeline.
//!
//! This module defines the structs and enums that flow through the RAG pipeline.
//! Each type represents a stage or artifact of the retrieval-augmented generation
//! process:
//!
//! ```text
//! Question ──► Router ──► Retriever ──► Expander ──► Assembler ──► Synthesizer ──► Answer
//!              │            │             │             │              │
//!              ▼            ▼             ▼             ▼              ▼
//!           Strategy    Chunks       More Chunks   Context Text    RagResult
//! ```
//!
//! ## Rust Learning: Derive macros
//!
//! Every type here derives several traits via `#[derive(...)]`:
//! - `Debug`: Enables `{:?}` formatting for logging and error messages.
//! - `Clone`: Allows creating deep copies. We use `Clone` (not `Copy`) because
//!   these types contain heap-allocated data (`String`, `Vec`).
//! - `Serialize, Deserialize`: Enables JSON serialization via serde. This is
//!   essential for API responses, logging, and caching.
//! - `PartialEq`: Enables `==` comparison, useful in tests.
//!
//! We do NOT derive `Copy` because that requires all fields to be `Copy`,
//! and `String`/`Vec` are heap-allocated and cannot be `Copy`.

use serde::{Deserialize, Serialize};

// ===========================================================================
// Retrieval Strategy — how the router decides to search
// ===========================================================================

/// Describes HOW the pipeline should retrieve context for a question.
///
/// The [`QueryRouter`](crate::QueryRouter) analyzes the user's question and
/// produces one of these strategies. Each strategy tells the retriever and
/// expander how aggressively to search.
///
/// ## Rust Learning: Enum variants with data
///
/// Unlike Java enums (which are fixed constants), Rust enums can carry
/// different data in each variant — like a tagged union in C. This lets us
/// express "a Focused search needs scope filters, but a Broad search doesn't"
/// in a single type-safe enum.
///
/// ## Rust Learning: `#[serde(rename_all = "snake_case")]`
///
/// This attribute tells serde to convert variant names to snake_case in JSON.
/// So `Focused { ... }` serializes as `{ "focused": { "scope": [...] } }`.
/// Without this, serde would use the exact Rust name (PascalCase).
///
/// ## Rust Learning: `#[serde(tag = "type", content = "params")]`
///
/// This is "adjacently tagged" representation. Each variant serializes as:
/// ```json
/// { "type": "focused", "params": { "scope": [...] } }
/// ```
/// This makes the JSON self-describing — you can always see the strategy type
/// in the `"type"` field, regardless of which variant was chosen.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type", content = "params")]
pub enum RetrievalStrategy {
    /// Search within specific scope boundaries (e.g., one document, one person).
    ///
    /// Example: "What did Phillips say in his deposition?"
    /// → Focused with scope = [Person("Phillips"), NodeType("Evidence")]
    Focused { scope: Vec<ScopeFilter> },

    /// Search broadly across the whole knowledge graph.
    ///
    /// Example: "Summarize the key claims in this case"
    /// → Broad with no specific node_type filter
    ///
    /// The optional `node_types` field narrows results to specific categories
    /// without restricting to a particular person or document.
    Broad {
        node_types: Option<Vec<String>>,
    },

    /// Combine focused scopes with cross-scope synthesis.
    ///
    /// Example: "Compare Phillips' testimony with Awad's claims"
    /// → Hybrid with two Person scopes + synthesize_across = true
    ///
    /// When `synthesize_across` is true, the synthesizer is instructed
    /// to draw connections between the different scopes rather than
    /// treating them independently.
    Hybrid {
        scopes: Vec<ScopeFilter>,
        synthesize_across: bool,
    },

    /// Bypass vector search — use a direct graph query or lookup.
    ///
    /// Example: "List all exhibits mentioned in document D-001"
    /// → Direct with query_hint describing the graph traversal
    ///
    /// The `query_hint` is a human-readable description of what to look for.
    /// It may be used by the graph expander to construct a Cypher query.
    Direct { query_hint: String },
}

/// A filter that narrows the scope of a retrieval search.
///
/// Scope filters are used by the [`VectorRetriever`](crate::VectorRetriever)
/// to add Qdrant payload filters to the search, and by the
/// [`GraphExpander`](crate::GraphExpander) to constrain graph traversals.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScopeFilter {
    /// What kind of scope boundary this filter represents.
    pub filter_type: ScopeFilterType,

    /// The value to filter on (e.g., "Phillips", "motion-001", "Evidence").
    pub value: String,
}

/// The type of scope boundary for a [`ScopeFilter`].
///
/// ## Rust Learning: Simple enums
///
/// Unlike the [`RetrievalStrategy`] enum above (which carries data), this is
/// a "simple" enum — just named constants with no associated data. In memory
/// it's just a small integer (like a C enum). But unlike C, Rust ensures you
/// can only use valid variants — no casting from arbitrary integers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ScopeFilterType {
    /// Filter by source document (e.g., "motion-001", "deposition-phillips").
    Document,

    /// Filter by person/party name (e.g., "Phillips", "Awad").
    Person,

    /// Filter by graph node type (e.g., "Evidence", "MotionClaim", "Harm").
    NodeType,

    /// Filter by a named collection or grouping.
    Collection,
}

// ===========================================================================
// Context — chunks retrieved from vector search and graph expansion
// ===========================================================================

/// A single piece of context retrieved from the knowledge graph or vector store.
///
/// This is the fundamental unit of information flowing through the pipeline.
/// Both the [`VectorRetriever`](crate::VectorRetriever) and
/// [`GraphExpander`](crate::GraphExpander) produce these. They get assembled
/// into a single context window by the [`ContextAssembler`](crate::ContextAssembler).
///
/// ## Example (from Awad v. CFS case)
///
/// ```text
/// ContextChunk {
///     node_id: "evidence-phillips-q74",
///     node_type: "Evidence",
///     title: "Phillips: Emil wanted $50K returned",
///     content: "Q: Did Emil ever ask for the money back? A: Yes, multiple times...",
///     score: 0.7056,
///     source: SourceReference { document_title: Some("Phillips Deposition"), ... },
///     relationships: [RelatedNode { node_id: "harm-003", ... }],
///     metadata: {}
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextChunk {
    /// Unique identifier for this node in the knowledge graph.
    pub node_id: String,

    /// The type/category of this node (e.g., "Evidence", "MotionClaim", "Harm").
    pub node_type: String,

    /// Human-readable title summarizing this chunk's content.
    pub title: String,

    /// The actual text content to include in the context window.
    pub content: String,

    /// Similarity score from vector search (cosine similarity, 0.0–1.0).
    /// For graph-expanded nodes (not from vector search), this is 0.0.
    pub score: f32,

    /// Where this information came from (document, page, quote).
    pub source: SourceReference,

    /// Other nodes related to this one in the knowledge graph.
    pub relationships: Vec<RelatedNode>,

    /// Arbitrary metadata for future extensibility.
    ///
    /// ## Rust Learning: `serde_json::Value`
    ///
    /// `Value` is serde_json's "any JSON value" type — it can hold a string,
    /// number, array, object, bool, or null. We use it here as an escape hatch
    /// for metadata fields we haven't formalized yet. In Java, this would be
    /// like `Map<String, Object>`.
    ///
    /// `skip_serializing_if` omits the field from JSON when the value is
    /// JSON `null`, keeping API responses clean.
    #[serde(default)]
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

/// Reference to the source material where a piece of evidence originated.
///
/// Not all fields will be populated — for example, a graph-expanded node
/// may have no document reference, and a MotionClaim may have no page number.
///
/// ## Rust Learning: `Option<T>` fields with serde
///
/// `#[serde(skip_serializing_if = "Option::is_none")]` omits the field entirely
/// from JSON when it's `None`. Without this, you'd get `"document_title": null`
/// in every response, which is noisy. With it, the field simply doesn't appear.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SourceReference {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbatim_quote: Option<String>,
}

/// A node related to a [`ContextChunk`] in the knowledge graph.
///
/// These come from graph expansion (Neo4j traversal). When the expander
/// follows relationships from a seed node, it records what it finds as
/// `RelatedNode` entries attached to the parent chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelatedNode {
    /// The related node's ID in the graph.
    pub node_id: String,

    /// The related node's type (e.g., "Harm", "Party", "LegalConcept").
    pub node_type: String,

    /// The relationship type connecting this to the parent chunk
    /// (e.g., "SUPPORTS", "CONTRADICTS", "FILED_BY").
    pub relationship: String,

    /// Whether this relationship goes FROM the parent TO this node (Outbound)
    /// or FROM this node TO the parent (Inbound).
    pub direction: RelationDirection,

    /// A short summary of the related node's content.
    pub summary: String,
}

/// Direction of a graph relationship relative to the parent node.
///
/// In Neo4j, relationships have a direction: `(a)-[:REL]->(b)`.
/// - Outbound: the parent chunk POINTS TO this related node
/// - Inbound: this related node POINTS TO the parent chunk
///
/// This distinction matters for legal reasoning — "Phillips SUPPORTS claim X"
/// vs "claim X is SUPPORTED_BY Phillips" carry different semantic weight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RelationDirection {
    Outbound,
    Inbound,
}

// ===========================================================================
// Pipeline output — the final result of a RAG query
// ===========================================================================

/// The complete result of a RAG pipeline execution.
///
/// This is what the caller receives after asking a question. It includes
/// the synthesized answer, the evidence used, citations, and performance stats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RagResult {
    /// The synthesized natural-language answer from Claude.
    pub answer: String,

    /// Which retrieval strategy the router chose for this question.
    pub strategy_used: RetrievalStrategy,

    /// All context chunks that were assembled into the prompt.
    pub chunks: Vec<ContextChunk>,

    /// Structured citations extracted from the synthesis.
    pub citations: Vec<Citation>,

    /// Timing and token usage statistics for this pipeline run.
    pub stats: PipelineStats,
}

/// A citation referencing a specific piece of evidence.
///
/// Citations are extracted from the synthesizer's response to provide
/// traceable, verifiable references back to source material.
///
/// All fields are optional because not every citation has a document
/// reference, page number, or verbatim quote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Citation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub quote_excerpt: Option<String>,
}

/// Timing and usage statistics for a single RAG pipeline run.
///
/// Every stage of the pipeline records its duration in milliseconds.
/// Token counts come from the synthesizer's response (Claude API).
///
/// ## Rust Learning: `Default` derive
///
/// `#[derive(Default)]` auto-generates a `Default::default()` implementation
/// where every field gets its type's default value:
/// - `u64` → 0
/// - `usize` → 0
/// - `String` → "" (empty string)
/// - `Option<T>` → None
///
/// This is useful for building stats incrementally — start with defaults,
/// then fill in each timing field as the pipeline progresses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PipelineStats {
    /// Which strategy was used (human-readable name).
    pub strategy: String,

    /// Time spent in the router (classifying the question).
    pub route_ms: u64,

    /// Time spent embedding the query text.
    pub embed_ms: u64,

    /// Time spent searching the vector store (Qdrant).
    pub search_ms: u64,

    /// Time spent expanding context via the knowledge graph (Neo4j).
    pub expand_ms: u64,

    /// Time spent assembling the context window.
    pub assemble_ms: u64,

    /// Time spent in the synthesizer (Claude API call).
    pub synthesize_ms: u64,

    /// Total end-to-end time for the pipeline run.
    pub total_ms: u64,

    /// Number of hits returned from Qdrant vector search.
    pub qdrant_hits: usize,

    /// Number of nodes expanded from the knowledge graph.
    pub graph_nodes_expanded: usize,

    /// Estimated token count of the assembled context.
    pub context_tokens_approx: usize,

    /// Actual input tokens reported by the LLM provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,

    /// Actual output tokens reported by the LLM provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,

    /// Which LLM provider was used (e.g., "anthropic").
    pub provider: String,

    /// Which model was used (e.g., "claude-sonnet-4-20250514").
    pub model: String,
}

// ===========================================================================
// Internal pipeline types — used between stages
// ===========================================================================

/// The assembled context ready to send to the synthesizer.
///
/// The [`ContextAssembler`](crate::ContextAssembler) produces this by
/// formatting the context chunks into a single text block with a system
/// prompt, then estimating the token count.
///
/// This is an intermediate type — it flows from assembler to synthesizer
/// and is not part of the public API response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AssembledContext {
    /// The system prompt instructing Claude how to behave.
    pub system_prompt: String,

    /// The formatted context text containing all evidence chunks.
    pub formatted_context: String,

    /// Estimated token count (used to check against model limits).
    pub token_estimate: usize,
}

/// The result from the synthesizer before it's merged into [`RagResult`].
///
/// This carries the raw LLM response plus token usage metrics.
/// The pipeline coordinator merges this with chunks and stats to produce
/// the final [`RagResult`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynthesisResult {
    /// The synthesized natural-language answer.
    pub answer: String,

    /// Citations extracted from the answer.
    pub citations: Vec<Citation>,

    /// Input tokens consumed by the LLM.
    pub input_tokens: u32,

    /// Output tokens generated by the LLM.
    pub output_tokens: u32,

    /// Provider name (e.g., "anthropic").
    pub provider: String,

    /// Model name (e.g., "claude-sonnet-4-20250514").
    pub model: String,
}
