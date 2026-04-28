//! Chunk-level merge of LLM extraction output.
//!
//! When a document is split into multiple chunks and each chunk is extracted
//! independently, reference entities (people, organizations, ...) may appear
//! in multiple chunks while evidence entities (specific quotes, signed
//! statements) are unique per chunk. The merger:
//!
//! 1. Prefixes every entity ID with `chunk{N}:` to avoid cross-chunk
//!    collisions during processing.
//! 2. Deduplicates entities whose `entity_type` is in a configurable
//!    set, grouping by `(entity_type, normalize_name(label))` and keeping
//!    the survivor with the most non-null property values.
//! 3. Remaps relationship endpoints from discarded IDs to surviving IDs.
//! 4. Strips chunk prefixes from dedup survivors and applies a stable
//!    `-c{N}` suffix to non-dedup entities so cross-chunk ID collisions
//!    cannot reappear in the final output.
//!
//! ## Rust Learning: Why `HashSet<String>` instead of a trait?
//!
//! Callers know which entity types are "reference" types from the
//! extraction schema. A plain set is explicit, testable, serializable for
//! logging, and avoids closure-lifetime complexity. Future versions could
//! grow into a trait if more nuanced dedup logic is needed.

use crate::types::{ExtractedEntity, ExtractedRelationship};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Internal prefix used to namespace entity IDs by source chunk during
/// processing. Stripped (or replaced with a `-c{N}` suffix) before output.
pub(crate) const CHUNK_PREFIX: &str = "chunk";
/// Length of [`CHUNK_PREFIX`] — used for fast slicing in
/// [`strip_chunk_prefix`] without a magic number.
pub(crate) const CHUNK_PREFIX_LEN: usize = CHUNK_PREFIX.len();
/// Character separating the chunk index from the original ID inside a
/// prefixed identifier (e.g., `chunk0:party-001`).
pub(crate) const CHUNK_DELIMITER: char = ':';
/// Suffix applied to non-dedup entity IDs to preserve chunk provenance
/// without leaking the `chunk` prefix into final output.
pub(crate) const NON_DEDUP_SUFFIX: &str = "-c";
/// Prefix used when an extracted entity arrived without an `id`. The
/// `entity_idx` within the chunk disambiguates multiple unnamed entities.
pub(crate) const UNNAMED_ID_PREFIX: &str = "unnamed-";

/// Outcome of merging extraction results from multiple chunks.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Deduplicated entities with chunk prefixes resolved.
    pub entities: Vec<ExtractedEntity>,
    /// Relationships with endpoints remapped to surviving entity IDs.
    pub relationships: Vec<ExtractedRelationship>,
    /// Per-merge statistics for logging and debugging.
    pub stats: MergeStats,
}

/// Statistics from a chunk merge — captured for JSONB logging so operators
/// can see how aggressive dedup was on a given document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeStats {
    /// Total entities across all chunks before merge.
    pub entities_before: usize,
    /// Total entities after dedup.
    pub entities_after: usize,
    /// Number of entities removed by dedup.
    pub entities_deduplicated: usize,
    /// Total relationships across all chunks before merge.
    pub relationships_before: usize,
    /// Relationships after merge — equal to `relationships_before` unless
    /// the algorithm starts dropping relationships in a future revision.
    pub relationships_after: usize,
    /// Number of relationship endpoints that pointed to a discarded
    /// (deduplicated) ID and were rewritten to the surviving ID.
    pub references_remapped: usize,
}

/// Merges extraction results from multiple chunks into a single
/// deduplicated entity-and-relationship set.
///
/// See the module-level docs for the algorithm. The `dedup_entity_types`
/// set is the only configuration: any entity whose `entity_type` is in the
/// set is a dedup candidate; everything else is preserved per-chunk.
pub struct ChunkMerger {
    dedup_entity_types: HashSet<String>,
}

/// Internal book-keeping record paired with each entity during the merge.
struct PrefixedEntity {
    chunk_idx: usize,
    original_id: String,
    prefixed_id: String,
    entity: ExtractedEntity,
}

impl ChunkMerger {
    /// Build a merger that deduplicates entities whose `entity_type`
    /// appears in `dedup_entity_types`.
    pub fn new(dedup_entity_types: HashSet<String>) -> Self {
        Self { dedup_entity_types }
    }

    /// Merge entities and relationships from multiple chunks.
    ///
    /// `chunks` is a `Vec<(chunk_index, entities, relationships)>`. Order
    /// is preserved: result entities/relationships appear in the same
    /// chunk-then-within-chunk order as the input, with discarded
    /// duplicates removed.
    pub fn merge(
        &self,
        chunks: Vec<(usize, Vec<ExtractedEntity>, Vec<ExtractedRelationship>)>,
    ) -> MergeResult {
        let mut all_entities: Vec<PrefixedEntity> = Vec::new();
        let mut all_relationships: Vec<ExtractedRelationship> = Vec::new();
        let mut entities_before: usize = 0;
        let mut relationships_before: usize = 0;

        for (chunk_idx, entities, relationships) in chunks {
            for (entity_idx, entity) in entities.into_iter().enumerate() {
                entities_before += 1;
                let original_id = entity
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("{UNNAMED_ID_PREFIX}{entity_idx}"));
                let prefixed_id =
                    format!("{CHUNK_PREFIX}{chunk_idx}{CHUNK_DELIMITER}{original_id}");
                let mut entity = entity;
                entity.id = Some(prefixed_id.clone());
                all_entities.push(PrefixedEntity {
                    chunk_idx,
                    original_id,
                    prefixed_id,
                    entity,
                });
            }
            for rel in relationships {
                relationships_before += 1;
                let from = format!(
                    "{CHUNK_PREFIX}{chunk_idx}{CHUNK_DELIMITER}{}",
                    rel.from_entity
                );
                let to = format!(
                    "{CHUNK_PREFIX}{chunk_idx}{CHUNK_DELIMITER}{}",
                    rel.to_entity
                );
                all_relationships.push(ExtractedRelationship {
                    relationship_type: rel.relationship_type,
                    from_entity: from,
                    to_entity: to,
                    properties: rel.properties,
                });
            }
        }

        let mut groups: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (idx, pe) in all_entities.iter().enumerate() {
            if self.dedup_entity_types.contains(&pe.entity.entity_type) {
                let key = (pe.entity.entity_type.clone(), normalize_name(&pe.entity.label));
                groups.entry(key).or_default().push(idx);
            }
        }

        let mut id_to_final: HashMap<String, String> = HashMap::new();
        let mut discarded_indices: HashSet<usize> = HashSet::new();
        let mut discarded_prefixed_ids: HashSet<String> = HashSet::new();
        let mut entities_deduplicated: usize = 0;

        for indices in groups.values() {
            if let Some(&survivor_idx) = indices.iter().max_by_key(|&&i| {
                (
                    count_properties(&all_entities[i].entity.properties),
                    std::cmp::Reverse(i),
                )
            }) {
                let survivor_final = strip_chunk_prefix(&all_entities[survivor_idx].prefixed_id);
                id_to_final.insert(
                    all_entities[survivor_idx].prefixed_id.clone(),
                    survivor_final.clone(),
                );

                for &member_idx in indices {
                    if member_idx != survivor_idx {
                        let discarded = all_entities[member_idx].prefixed_id.clone();
                        id_to_final.insert(discarded.clone(), survivor_final.clone());
                        discarded_prefixed_ids.insert(discarded);
                        discarded_indices.insert(member_idx);
                        entities_deduplicated += 1;
                    }
                }
            }
        }

        for pe in &all_entities {
            if !self.dedup_entity_types.contains(&pe.entity.entity_type) {
                let original = &pe.original_id;
                let chunk_idx = pe.chunk_idx;
                let final_id = format!("{original}{NON_DEDUP_SUFFIX}{chunk_idx}");
                id_to_final.insert(pe.prefixed_id.clone(), final_id);
            }
        }

        let mut references_remapped: usize = 0;
        let final_relationships: Vec<ExtractedRelationship> = all_relationships
            .into_iter()
            .map(|rel| {
                let from_final = id_to_final
                    .get(&rel.from_entity)
                    .cloned()
                    .unwrap_or_else(|| strip_chunk_prefix(&rel.from_entity));
                let to_final = id_to_final
                    .get(&rel.to_entity)
                    .cloned()
                    .unwrap_or_else(|| strip_chunk_prefix(&rel.to_entity));
                if discarded_prefixed_ids.contains(&rel.from_entity) {
                    references_remapped += 1;
                }
                if discarded_prefixed_ids.contains(&rel.to_entity) {
                    references_remapped += 1;
                }
                ExtractedRelationship {
                    relationship_type: rel.relationship_type,
                    from_entity: from_final,
                    to_entity: to_final,
                    properties: rel.properties,
                }
            })
            .collect();

        let final_entities: Vec<ExtractedEntity> = all_entities
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !discarded_indices.contains(idx))
            .map(|(_, pe)| {
                let final_id = id_to_final
                    .get(&pe.prefixed_id)
                    .cloned()
                    .unwrap_or_else(|| strip_chunk_prefix(&pe.prefixed_id));
                let mut entity = pe.entity;
                entity.id = Some(final_id);
                entity
            })
            .collect();

        let entities_after = final_entities.len();
        let relationships_after = final_relationships.len();

        MergeResult {
            entities: final_entities,
            relationships: final_relationships,
            stats: MergeStats {
                entities_before,
                entities_after,
                entities_deduplicated,
                relationships_before,
                relationships_after,
                references_remapped,
            },
        }
    }
}

/// Normalize an entity label for dedup grouping — lowercase, collapse
/// internal whitespace runs to a single space, trim outer whitespace.
fn normalize_name(name: &str) -> String {
    name.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Count the non-null property values inside an entity's `properties` JSON.
/// Returns 0 if the value is not a JSON object — those entities are
/// considered "no metadata" for tie-breaking purposes.
fn count_properties(properties: &serde_json::Value) -> usize {
    match properties.as_object() {
        Some(obj) => obj.values().filter(|v| !v.is_null()).count(),
        None => 0,
    }
}

/// Remove a leading `chunk{digits}:` namespace from an ID. Returns the
/// original string unchanged when no valid prefix is present (orphan
/// references and unprefixed IDs survive intact).
fn strip_chunk_prefix(id: &str) -> String {
    if id.starts_with(CHUNK_PREFIX) {
        if let Some(colon_pos) = id.find(CHUNK_DELIMITER) {
            let between = &id[CHUNK_PREFIX_LEN..colon_pos];
            if !between.is_empty() && between.chars().all(|c| c.is_ascii_digit()) {
                return id[colon_pos + 1..].to_string();
            }
        }
    }
    id.to_string()
}
