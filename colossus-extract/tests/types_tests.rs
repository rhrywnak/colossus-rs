//! Backward-compat tests for `TextChunk` and `AtomicUnit` deserialization.
//!
//! Pin the contract that older serialized payloads (without the `metadata`
//! field) in PostgreSQL JSONB columns still round-trip via `#[serde(default)]`.
//! The compiler does not catch removal of the attribute; these tests do.
//!
//! All struct-construct/derive/round-trip tests deleted in the test audit
//! (Batch 9, rules 1/3/5).

use colossus_extract::{AtomicUnit, TextChunk};

#[test]
fn test_textchunk_deserialize_without_metadata_field() {
    // Existing serialized TextChunks in the database and in tests don't
    // have a metadata field. They must continue to deserialize cleanly.
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

#[test]
fn test_atomicunit_deserialize_without_metadata_field() {
    // Same backward-compat contract for AtomicUnit.
    let json_str = r#"{"text": "unit text", "index": 0, "start_offset": 0, "end_offset": 9}"#;
    let unit: AtomicUnit =
        serde_json::from_str(json_str).expect("must deserialize without metadata");

    assert_eq!(unit.text, "unit text");
    assert!(unit.metadata.is_empty());
}
