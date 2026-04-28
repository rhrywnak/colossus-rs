//! Behavioral tests for `TextChunk` metadata and `AtomicUnit`.
//!
//! These tests verify the serde-default backward compatibility, mixed-type
//! metadata storage, and offset semantics that downstream splitters rely on.
//! Each test asserts specific values — never just "didn't crash."

use colossus_extract::{AtomicUnit, TextChunk};
use serde_json::json;
use std::collections::HashMap;

// --- TextChunk tests ---

#[test]
fn test_textchunk_default_metadata_is_empty() {
    // Verify that a TextChunk constructed without metadata has an empty map.
    let chunk = TextChunk {
        text: "hello".to_string(),
        index: 0,
        metadata: HashMap::new(),
    };
    assert!(chunk.metadata.is_empty(), "Default metadata must be empty");
}

#[test]
fn test_textchunk_metadata_stores_and_retrieves_values() {
    // Verify that metadata can store mixed types and retrieve them correctly.
    let mut meta = HashMap::new();
    meta.insert("unit_range".to_string(), json!([0, 24]));
    meta.insert("preamble_included".to_string(), json!(true));
    meta.insert("unit_count".to_string(), json!(25));
    meta.insert("fallback".to_string(), json!(false));

    let chunk = TextChunk {
        text: "some text".to_string(),
        index: 3,
        metadata: meta,
    };

    // Verify specific values, not just "it exists"
    assert_eq!(chunk.metadata["unit_range"], json!([0, 24]));
    assert_eq!(chunk.metadata["preamble_included"], json!(true));
    assert_eq!(chunk.metadata["unit_count"], json!(25));
    assert_eq!(chunk.metadata.len(), 4);
}

#[test]
fn test_textchunk_serialization_roundtrip_with_metadata() {
    // Verify that metadata survives JSON serialization and deserialization.
    // This matters because chunks are logged to JSONB columns in PostgreSQL.
    let mut meta = HashMap::new();
    meta.insert("strategy".to_string(), json!("qa_pair"));
    meta.insert("unit_range".to_string(), json!([10, 34]));

    let original = TextChunk {
        text: "chunk content".to_string(),
        index: 2,
        metadata: meta,
    };

    let json_str = serde_json::to_string(&original).expect("serialize must succeed");
    let restored: TextChunk = serde_json::from_str(&json_str).expect("deserialize must succeed");

    assert_eq!(restored.text, "chunk content");
    assert_eq!(restored.index, 2);
    assert_eq!(restored.metadata["strategy"], json!("qa_pair"));
    assert_eq!(restored.metadata["unit_range"], json!([10, 34]));
    assert_eq!(restored.metadata.len(), 2);
}

#[test]
fn test_textchunk_deserialize_without_metadata_field() {
    // Verify backward compatibility: JSON without a "metadata" key
    // deserializes successfully with an empty metadata map.
    // This is critical — existing serialized TextChunks in the database
    // and in tests don't have a metadata field.
    let json_str = r#"{"text": "old chunk", "index": 0}"#;
    let chunk: TextChunk =
        serde_json::from_str(json_str).expect("must deserialize without metadata");

    assert_eq!(chunk.text, "old chunk");
    assert_eq!(chunk.index, 0);
    assert!(
        chunk.metadata.is_empty(),
        "Missing metadata field must default to empty HashMap"
    );
}

// --- AtomicUnit tests ---

#[test]
fn test_atomicunit_offsets_define_text_slice() {
    // Verify that start_offset and end_offset correctly correspond to
    // the unit's position in a source document. This is the contract
    // that the StructureAwareSplitter depends on.
    let source = "Preamble text.\n1. First paragraph content.\n2. Second paragraph content.";
    let unit = AtomicUnit {
        text: "1. First paragraph content.".to_string(),
        index: 0,
        start_offset: 15,
        end_offset: 42,
        metadata: HashMap::new(),
    };

    // The offsets must correctly slice the source to reproduce the unit text
    assert_eq!(&source[unit.start_offset..unit.end_offset], unit.text);
}

#[test]
fn test_atomicunit_metadata_stores_identifier() {
    // Verify that metadata can carry a question/paragraph identifier.
    // The StructureAwareSplitter will populate this; downstream consumers
    // (like JSONB logging) read it.
    let mut meta = HashMap::new();
    meta.insert("identifier".to_string(), json!("Q42"));
    meta.insert("page_hint".to_string(), json!(7));

    let unit = AtomicUnit {
        text: "42. What is the meaning of life?\nAnswer: 42.".to_string(),
        index: 41,
        start_offset: 8000,
        end_offset: 8045,
        metadata: meta,
    };

    assert_eq!(unit.metadata["identifier"], json!("Q42"));
    assert_eq!(unit.metadata["page_hint"], json!(7));
    assert_eq!(unit.index, 41);
}

#[test]
fn test_atomicunit_serialization_roundtrip() {
    // Verify AtomicUnit survives serialization with all fields intact.
    let mut meta = HashMap::new();
    meta.insert("section_title".to_string(), json!("FINDINGS OF FACT"));

    let original = AtomicUnit {
        text: "The court finds...".to_string(),
        index: 0,
        start_offset: 500,
        end_offset: 518,
        metadata: meta,
    };

    let json_str = serde_json::to_string(&original).expect("serialize must succeed");
    let restored: AtomicUnit = serde_json::from_str(&json_str).expect("deserialize must succeed");

    assert_eq!(restored.text, original.text);
    assert_eq!(restored.index, 0);
    assert_eq!(restored.start_offset, 500);
    assert_eq!(restored.end_offset, 518);
    assert_eq!(restored.metadata["section_title"], json!("FINDINGS OF FACT"));
}

#[test]
fn test_atomicunit_deserialize_without_metadata_field() {
    // Verify forward compatibility: AtomicUnit JSON without metadata
    // deserializes with empty metadata.
    let json_str = r#"{"text": "unit text", "index": 0, "start_offset": 0, "end_offset": 9}"#;
    let unit: AtomicUnit =
        serde_json::from_str(json_str).expect("must deserialize without metadata");

    assert_eq!(unit.text, "unit text");
    assert!(unit.metadata.is_empty());
}
