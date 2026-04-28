//! Behavioral tests for `ChunkMerger`.
//!
//! Each test pins a specific input scenario to specific expected outputs
//! — entity counts, surviving IDs, property completeness, statistics.

use colossus_extract::{ChunkMerger, ExtractedEntity, ExtractedRelationship};
use serde_json::json;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn party(id: &str, name: &str, props: serde_json::Value) -> ExtractedEntity {
    ExtractedEntity {
        entity_type: "Party".to_string(),
        id: Some(id.to_string()),
        label: name.to_string(),
        properties: props,
        verbatim_quote: None,
        page_number: None,
        grounding_status: None,
    }
}

fn evidence(id: &str, title: &str, quote: &str) -> ExtractedEntity {
    ExtractedEntity {
        entity_type: "Evidence".to_string(),
        id: Some(id.to_string()),
        label: title.to_string(),
        properties: json!({"title": title, "significance": "test"}),
        verbatim_quote: Some(quote.to_string()),
        page_number: Some(1),
        grounding_status: None,
    }
}

fn relationship(rel_type: &str, from: &str, to: &str) -> ExtractedRelationship {
    ExtractedRelationship {
        relationship_type: rel_type.to_string(),
        from_entity: from.to_string(),
        to_entity: to.to_string(),
        properties: None,
    }
}

fn dedup_types() -> HashSet<String> {
    ["Party".to_string()].into_iter().collect()
}

// ---------------------------------------------------------------------------
// Entity deduplication tests
// ---------------------------------------------------------------------------

#[test]
fn test_same_party_in_two_chunks_deduplicates_to_one() {
    let chunks = vec![
        (
            0,
            vec![
                party("party-001", "George Phillips", json!({"role": "trustee"})),
                evidence("ev-001", "Statement about trust", "I managed the trust"),
            ],
            vec![relationship("STATED_BY", "ev-001", "party-001")],
        ),
        (
            1,
            vec![
                party(
                    "party-001",
                    "George Phillips",
                    json!({"role": "trustee", "title": "Mr."}),
                ),
                evidence("ev-002", "Another statement", "The assets were distributed"),
            ],
            vec![relationship("STATED_BY", "ev-002", "party-001")],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    let parties: Vec<_> = result
        .entities
        .iter()
        .filter(|e| e.entity_type == "Party")
        .collect();
    assert_eq!(
        parties.len(),
        1,
        "Same party in two chunks must deduplicate to 1. Got: {}",
        parties.len()
    );

    let evidence_count = result
        .entities
        .iter()
        .filter(|e| e.entity_type == "Evidence")
        .count();
    assert_eq!(evidence_count, 2, "Evidence entities must never be deduplicated");

    assert_eq!(
        result.stats.entities_before, 4,
        "2 entities per chunk × 2 chunks = 4"
    );
    assert_eq!(result.stats.entities_after, 3, "1 party + 2 evidence = 3");
    assert_eq!(result.stats.entities_deduplicated, 1);
}

#[test]
fn test_dedup_keeps_most_complete_properties() {
    let chunks = vec![
        (
            0,
            vec![party(
                "party-001",
                "Marie Awad",
                json!({"role": "plaintiff"}),
            )],
            vec![],
        ),
        (
            1,
            vec![party(
                "party-001",
                "Marie Awad",
                json!({"role": "plaintiff", "age": 65, "residence": "Bay City"}),
            )],
            vec![],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    let parties: Vec<_> = result
        .entities
        .iter()
        .filter(|e| e.entity_type == "Party")
        .collect();
    assert_eq!(parties.len(), 1);

    let props = parties[0]
        .properties
        .as_object()
        .expect("properties must be object");
    assert_eq!(
        props.len(),
        3,
        "Surviving entity must have 3 properties (most complete). Got: {}",
        props.len()
    );
    assert_eq!(props["residence"], json!("Bay City"));
}

#[test]
fn test_dedup_normalizes_name_case_and_whitespace() {
    let chunks = vec![
        (
            0,
            vec![party(
                "party-001",
                "George Phillips",
                json!({"role": "trustee"}),
            )],
            vec![],
        ),
        (
            1,
            vec![party(
                "party-001",
                "george  phillips",
                json!({"role": "trustee", "title": "Mr."}),
            )],
            vec![],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    let parties: Vec<_> = result
        .entities
        .iter()
        .filter(|e| e.entity_type == "Party")
        .collect();
    assert_eq!(
        parties.len(),
        1,
        "Name normalization must merge 'George Phillips' with 'george  phillips'"
    );
}

#[test]
fn test_different_parties_not_merged() {
    let chunks = vec![
        (
            0,
            vec![party(
                "party-001",
                "George Phillips",
                json!({"role": "trustee"}),
            )],
            vec![],
        ),
        (
            1,
            vec![party(
                "party-001",
                "Marie Awad",
                json!({"role": "plaintiff"}),
            )],
            vec![],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    let parties: Vec<_> = result
        .entities
        .iter()
        .filter(|e| e.entity_type == "Party")
        .collect();
    assert_eq!(parties.len(), 2, "Different parties must not be merged");
}

#[test]
fn test_evidence_entities_never_deduplicated() {
    let chunks = vec![
        (
            0,
            vec![evidence("ev-001", "Trust Distribution", "Quote A")],
            vec![],
        ),
        (
            1,
            vec![evidence("ev-001", "Trust Distribution", "Quote B")],
            vec![],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert_eq!(
        result.entities.len(),
        2,
        "Evidence entities must never be deduplicated, even with identical titles"
    );
}

#[test]
fn test_entity_without_id_gets_generated_id() {
    let mut entity = party("", "No ID Party", json!({"role": "unknown"}));
    entity.id = None;

    let chunks = vec![(0, vec![entity], vec![])];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert_eq!(result.entities.len(), 1);
    let id = result.entities[0]
        .id
        .as_ref()
        .expect("Entity must have an ID after merge");
    assert!(!id.is_empty(), "Generated ID must not be empty");
    assert!(
        !id.starts_with("chunk"),
        "Chunk prefix must be stripped from final output"
    );
}

// ---------------------------------------------------------------------------
// Relationship remapping tests
// ---------------------------------------------------------------------------

#[test]
fn test_relationships_remapped_to_surviving_entity() {
    let chunks = vec![
        (
            0,
            vec![
                party("party-001", "George Phillips", json!({"role": "trustee"})),
                evidence("ev-001", "Statement 1", "Quote 1"),
            ],
            vec![relationship("STATED_BY", "ev-001", "party-001")],
        ),
        (
            1,
            vec![
                party(
                    "party-001",
                    "George Phillips",
                    json!({"role": "trustee", "title": "Mr."}),
                ),
                evidence("ev-002", "Statement 2", "Quote 2"),
            ],
            vec![relationship("STATED_BY", "ev-002", "party-001")],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    let stated_by_targets: HashSet<_> = result
        .relationships
        .iter()
        .filter(|r| r.relationship_type == "STATED_BY")
        .map(|r| r.to_entity.clone())
        .collect();
    assert_eq!(
        stated_by_targets.len(),
        1,
        "Both STATED_BY must point to the same surviving Party ID. Got: {stated_by_targets:?}"
    );

    let surviving_party_id = stated_by_targets
        .into_iter()
        .next()
        .expect("set has one entry");
    let party_exists = result
        .entities
        .iter()
        .any(|e| e.id.as_deref() == Some(&surviving_party_id));
    assert!(
        party_exists,
        "Surviving party ID '{surviving_party_id}' must exist in entities"
    );

    assert!(
        result.stats.references_remapped > 0,
        "At least one reference must be remapped"
    );
}

#[test]
fn test_chunk_prefixes_stripped_from_final_output() {
    let chunks = vec![(
        0,
        vec![
            party("party-001", "Test Person", json!({})),
            evidence("ev-001", "Test Evidence", "Quote"),
        ],
        vec![relationship("STATED_BY", "ev-001", "party-001")],
    )];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    for entity in &result.entities {
        if let Some(ref id) = entity.id {
            assert!(
                !id.starts_with("chunk"),
                "Entity ID '{id}' must not have chunk prefix"
            );
        }
    }
    for rel in &result.relationships {
        assert!(
            !rel.from_entity.starts_with("chunk"),
            "Relationship from_entity '{}' must not have chunk prefix",
            rel.from_entity
        );
        assert!(
            !rel.to_entity.starts_with("chunk"),
            "Relationship to_entity '{}' must not have chunk prefix",
            rel.to_entity
        );
    }
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_single_chunk_no_dedup_needed() {
    let chunks = vec![(
        0,
        vec![
            party("party-001", "George Phillips", json!({"role": "trustee"})),
            evidence("ev-001", "Statement", "Quote"),
        ],
        vec![relationship("STATED_BY", "ev-001", "party-001")],
    )];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert_eq!(result.entities.len(), 2);
    assert_eq!(result.relationships.len(), 1);
    assert_eq!(result.stats.entities_deduplicated, 0);
    assert_eq!(result.stats.references_remapped, 0);
}

#[test]
fn test_empty_input_returns_empty_result() {
    let chunks: Vec<(usize, Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> = vec![];
    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert!(result.entities.is_empty());
    assert!(result.relationships.is_empty());
    assert_eq!(result.stats.entities_before, 0);
    assert_eq!(result.stats.entities_after, 0);
}

#[test]
fn test_empty_dedup_set_keeps_all_entities() {
    let chunks = vec![
        (
            0,
            vec![party("party-001", "George Phillips", json!({}))],
            vec![],
        ),
        (
            1,
            vec![party("party-001", "George Phillips", json!({}))],
            vec![],
        ),
    ];

    let merger = ChunkMerger::new(HashSet::new());
    let result = merger.merge(chunks);

    assert_eq!(
        result.entities.len(),
        2,
        "Empty dedup set means no dedup — both entities survive"
    );
    assert_eq!(result.stats.entities_deduplicated, 0);
}

#[test]
fn test_three_chunks_same_party_deduplicates_to_one() {
    let chunks = vec![
        (0, vec![party("p", "Test Person", json!({"a": 1}))], vec![]),
        (
            1,
            vec![party("p", "Test Person", json!({"a": 1, "b": 2}))],
            vec![],
        ),
        (
            2,
            vec![party("p", "Test Person", json!({"a": 1, "b": 2, "c": 3}))],
            vec![],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert_eq!(result.entities.len(), 1);
    let props = result.entities[0]
        .properties
        .as_object()
        .expect("properties must be object");
    assert_eq!(
        props.len(),
        3,
        "Must keep the most property-complete version (3 props from chunk 2)"
    );
    assert_eq!(result.stats.entities_deduplicated, 2);
}

#[test]
fn test_different_entity_types_same_name_not_merged() {
    let dedup: HashSet<String> = ["Party".to_string(), "Organization".to_string()]
        .into_iter()
        .collect();

    let mut org = party("org-001", "CFS", json!({}));
    org.entity_type = "Organization".to_string();

    let chunks = vec![
        (0, vec![party("party-001", "CFS", json!({}))], vec![]),
        (1, vec![org], vec![]),
    ];

    let merger = ChunkMerger::new(dedup);
    let result = merger.merge(chunks);

    assert_eq!(
        result.entities.len(),
        2,
        "Same name but different entity_type must NOT be merged"
    );
}

#[test]
fn test_id_collision_across_chunks_resolved_by_prefix() {
    let chunks = vec![
        (0, vec![evidence("ev-001", "Evidence A", "Quote A")], vec![]),
        (1, vec![evidence("ev-001", "Evidence B", "Quote B")], vec![]),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert_eq!(result.entities.len(), 2);
    let ids: Vec<_> = result
        .entities
        .iter()
        .filter_map(|e| e.id.as_deref())
        .collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(
        ids[0], ids[1],
        "Evidence entities from different chunks with same original ID must get distinct final IDs"
    );
}

#[test]
fn test_merge_stats_are_accurate() {
    let chunks = vec![
        (
            0,
            vec![
                party("p-001", "Person A", json!({"x": 1})),
                party("p-002", "Person B", json!({"x": 1})),
                evidence("e-001", "Ev1", "Q1"),
            ],
            vec![
                relationship("R1", "e-001", "p-001"),
                relationship("R2", "e-001", "p-002"),
            ],
        ),
        (
            1,
            vec![
                party("p-001", "Person A", json!({"x": 1, "y": 2})),
                evidence("e-002", "Ev2", "Q2"),
            ],
            vec![relationship("R1", "e-002", "p-001")],
        ),
    ];

    let merger = ChunkMerger::new(dedup_types());
    let result = merger.merge(chunks);

    assert_eq!(result.stats.entities_before, 5, "3 + 2 = 5 entities before");
    assert_eq!(
        result.stats.entities_after, 4,
        "2 parties (A deduped) + 2 evidence = 4"
    );
    assert_eq!(result.stats.entities_deduplicated, 1, "Person A deduplicated");
    assert_eq!(result.stats.relationships_before, 3);
    assert_eq!(result.stats.relationships_after, 3, "No relationships lost");
}
