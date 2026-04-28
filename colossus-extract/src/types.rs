//! Core data types for extraction pipeline output.
//!
//! ## Rust Learning: Serde attributes
//!
//! `#[serde(skip_serializing_if = "Option::is_none")]` omits null fields
//! from JSON output, keeping extraction results clean and compact.
//! `#[serde(rename_all = "snake_case")]` ensures JSON keys match Rust
//! naming conventions automatically.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A detected atomic unit within a document — the smallest meaningful
/// structural element that must never be split across chunks.
///
/// Examples of atomic units:
/// - One Q&A pair in a discovery response
/// - One numbered paragraph in a complaint
/// - One sworn statement in an affidavit
/// - One section under a heading
///
/// The StructureAwareSplitter detects these units using boundary patterns,
/// then groups them into TextChunks for LLM processing. The key invariant:
/// an atomic unit is NEVER split across chunk boundaries.
///
/// ## Rust Learning: Named fields vs HashMap
///
/// `text`, `index`, `start_offset`, and `end_offset` are named fields
/// because the splitter algorithm depends on them at compile time — they
/// are structural invariants. The `metadata` HashMap carries everything
/// else: question numbers, page hints, section titles — things that vary
/// by document type and that downstream consumers (not the splitter) use.
/// This gives compile-time safety for the core and runtime flexibility
/// for the periphery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicUnit {
    /// The full text of this atomic unit.
    pub text: String,
    /// Position of this unit in the sequence of detected units (0-based).
    pub index: usize,
    /// Byte offset where this unit starts in the original document text.
    pub start_offset: usize,
    /// Byte offset where this unit ends in the original document text (exclusive).
    pub end_offset: usize,
    /// Flexible metadata — identifier, page number, section title, anything
    /// the boundary detector finds that downstream consumers might need.
    /// Empty by default.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A chunk of text split from a document.
///
/// ## Rust Learning: Why not just a String?
///
/// We need to track which chunk a piece of text came from so we can:
/// 1. Prefix entity IDs with chunk index (prevents collisions)
/// 2. Store per-chunk metrics (tokens, duration, errors)
/// 3. Link extracted entities back to their source chunk
///
/// ## Rust Learning: Flexible metadata with HashMap
///
/// The `metadata` field uses `HashMap<String, serde_json::Value>` instead
/// of named fields for extensibility. Different splitters attach different
/// metadata — the FixedSizeSplitter might record character offsets, while
/// a StructureAwareSplitter records unit ranges and identifiers. Using a
/// HashMap means adding new metadata never requires changing this struct.
///
/// `#[serde(default)]` makes the field optional during deserialization —
/// existing serialized TextChunks without metadata will deserialize with
/// an empty HashMap. This is backward compatibility by design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// The chunk text content.
    pub text: String,
    /// Position of this chunk in the document (0-based).
    pub index: usize,
    /// Flexible metadata — splitter-specific information about this chunk.
    /// Examples: unit_range, unit_identifiers, preamble_included, fallback reason.
    /// Empty by default — existing splitters continue to work unchanged.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Statistics from graph pruning — what was removed and why.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PruningStats {
    /// Nodes removed because their label was not in the schema.
    pub nodes_not_in_schema: usize,
    /// Nodes removed because they were missing required properties.
    pub nodes_missing_properties: usize,
    /// Relationships removed because their type was not in the schema.
    pub rels_not_in_schema: usize,
    /// Relationships removed because start or end node was pruned.
    pub rels_orphaned: usize,
    /// Relationships removed because the pattern was invalid.
    pub rels_invalid_pattern: usize,
    /// Total nodes pruned.
    pub total_nodes_pruned: usize,
    /// Total relationships pruned.
    pub total_rels_pruned: usize,
}

/// A single entity extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity type name matching the schema (e.g. "Party", "FactualAllegation")
    pub entity_type: String,

    /// Unique identifier assigned during extraction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Human-readable label/title
    pub label: String,

    /// All properties as key-value pairs
    pub properties: serde_json::Value,

    /// Verbatim quote from the source document, if applicable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbatim_quote: Option<String>,

    /// Page number where this entity was found (set by PageGrounder)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,

    /// Grounding status after PageGrounder verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_status: Option<GroundingStatus>,
}

/// Result of PageGrounder verification for a verbatim quote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GroundingStatus {
    /// Exact substring match found in PDF text
    Exact,
    /// Match found after normalization (whitespace, case)
    Normalized,
    /// Quote not found in PDF text
    NotFound,
    /// Not yet verified
    Pending,
}

/// A relationship between two extracted entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// Relationship type matching the schema (e.g. "STATED_BY", "SUPPORTS")
    pub relationship_type: String,

    /// Source entity reference (id or label)
    pub from_entity: String,

    /// Target entity reference (id or label)
    pub to_entity: String,

    /// Optional properties on the relationship
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
}

/// Complete result of an LLM extraction pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// All extracted entities
    pub entities: Vec<ExtractedEntity>,

    /// All extracted relationships
    pub relationships: Vec<ExtractedRelationship>,

    /// Model that produced this extraction
    pub model: String,

    /// Number of input tokens consumed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,

    /// Number of output tokens produced
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
}

/// A known entity from the existing knowledge graph, used for entity resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownEntity {
    pub entity_type: String,
    pub id: String,
    pub label: String,
    pub properties: serde_json::Value,
}

/// Result of entity resolution — an extracted entity matched to a known entity
/// or marked as new.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedEntity {
    pub extracted: ExtractedEntity,
    /// If Some, this entity matches an existing known entity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_to: Option<String>,
    /// How the match was determined
    pub resolution: ResolutionMethod,
}

/// How an entity was resolved during entity resolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionMethod {
    /// Exact string match on label
    ExactMatch,
    /// Match after normalization (lowercase, trim, whitespace)
    NormalizedMatch,
    /// Fuzzy string matching (Jaro-Winkler, Levenshtein)
    FuzzyMatch,
    /// Semantic similarity via embeddings
    SemanticMatch,
    /// No match found — this is a new entity
    NewEntity,
}
