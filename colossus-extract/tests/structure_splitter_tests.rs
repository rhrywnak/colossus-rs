//! Behavioral tests for `StructureAwareSplitter`.
//!
//! Each test exercises the actual splitting logic with specific inputs
//! and asserts specific output values — never just "didn't crash."

use colossus_extract::{StructureAwareSplitter, TextSplitter};
use serde_json::json;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: build config maps for common test scenarios
// ---------------------------------------------------------------------------

fn qa_config(units_per_chunk: i64) -> HashMap<String, serde_json::Value> {
    let mut config = HashMap::new();
    config.insert("boundary_pattern".to_string(), json!(r"^\d+\.\s"));
    config.insert("response_marker".to_string(), json!(r"^Answer:\s"));
    config.insert("units_per_chunk".to_string(), json!(units_per_chunk));
    config
}

fn paragraph_config(units_per_chunk: i64) -> HashMap<String, serde_json::Value> {
    let mut config = HashMap::new();
    config.insert("boundary_pattern".to_string(), json!(r"^\d+\.\s"));
    config.insert("units_per_chunk".to_string(), json!(units_per_chunk));
    config
}

// ---------------------------------------------------------------------------
// detect_units tests
// ---------------------------------------------------------------------------

#[test]
fn test_detect_units_numbered_paragraphs() {
    let text = "\
1. First paragraph content here.
2. Second paragraph with more text.
3. Third paragraph is shorter.
4. Fourth paragraph goes on.
5. Fifth and final paragraph.";

    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let units = splitter.detect_units(text);

    assert_eq!(units.len(), 5, "Must detect exactly 5 numbered paragraphs");
    assert_eq!(units[0].index, 0);
    assert_eq!(units[4].index, 4);

    assert!(
        units[0].text.starts_with("1. First"),
        "First unit must start with '1. First', got: '{}'",
        &units[0].text[..20.min(units[0].text.len())]
    );

    assert!(
        units[4].text.starts_with("5. Fifth"),
        "Last unit must start with '5. Fifth'"
    );

    assert_eq!(units[0].metadata["identifier"], json!("1"));
    assert_eq!(units[4].metadata["identifier"], json!("5"));
}

#[test]
fn test_detect_units_qa_pairs_with_response_marker() {
    let text = "\
1. What is your name?
Answer: My name is John Smith.
2. Where do you live?
Answer: I live in Springfield.
3. What is your occupation?";

    let splitter = StructureAwareSplitter::from_config(qa_config(25));
    let units = splitter.detect_units(text);

    assert_eq!(units.len(), 3, "Must detect 3 Q&A units");

    assert!(
        units[0].text.contains("What is your name?"),
        "Q1 question missing"
    );
    assert!(
        units[0].text.contains("My name is John Smith"),
        "Q1 answer missing"
    );
    assert_eq!(units[0].metadata["has_response"], json!(true));

    assert_eq!(units[2].metadata["has_response"], json!(false));
}

#[test]
fn test_detect_units_with_preamble() {
    let text = "\
CASE HEADER: Awad v. CFS
Filed: January 2025

1. First allegation text.
2. Second allegation text.";

    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let units = splitter.detect_units(text);

    assert_eq!(units.len(), 2, "Must detect 2 units, not preamble");
    assert!(
        units[0].text.starts_with("1. First"),
        "First unit must be the first numbered paragraph"
    );

    assert!(
        units[0].start_offset > 0,
        "First unit's start_offset must be after the preamble"
    );
}

#[test]
fn test_detect_units_no_matches_returns_empty() {
    let text = "This document has no numbered paragraphs or structure.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let units = splitter.detect_units(text);
    assert!(units.is_empty(), "No boundary matches must return empty Vec");
}

#[test]
fn test_detect_units_invalid_regex_returns_empty() {
    let mut config = HashMap::new();
    config.insert("boundary_pattern".to_string(), json!("[invalid regex"));
    let splitter = StructureAwareSplitter::from_config(config);
    let units = splitter.detect_units("1. Some text\n2. More text");
    assert!(
        units.is_empty(),
        "Invalid regex must return empty Vec, not panic"
    );
}

#[test]
fn test_detect_units_offsets_slice_source_correctly() {
    let text = "Preamble.\n1. Alpha text.\n2. Beta text.\n3. Gamma text.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let units = splitter.detect_units(text);

    assert_eq!(units.len(), 3);
    for unit in &units {
        let sliced = &text[unit.start_offset..unit.end_offset];
        assert_eq!(
            sliced.trim_end(),
            unit.text,
            "Offset slice must match unit text for unit {}",
            unit.index
        );
    }
}

// ---------------------------------------------------------------------------
// group_into_chunks tests
// ---------------------------------------------------------------------------

#[test]
fn test_group_chunks_basic_grouping() {
    let text = "\
1. Unit one.
2. Unit two.
3. Unit three.
4. Unit four.
5. Unit five.";

    let splitter = StructureAwareSplitter::from_config(paragraph_config(2));
    let units = splitter.detect_units(text);
    let chunks = splitter.group_into_chunks(&units, "");

    assert_eq!(chunks.len(), 3, "5 units / 2 per chunk = 3 chunks");
    assert_eq!(chunks[0].metadata["unit_count"], json!(2));
    assert_eq!(chunks[1].metadata["unit_count"], json!(2));
    assert_eq!(chunks[2].metadata["unit_count"], json!(1));

    assert_eq!(chunks[0].metadata["unit_range"], json!([0, 1]));
    assert_eq!(chunks[1].metadata["unit_range"], json!([2, 3]));
    assert_eq!(chunks[2].metadata["unit_range"], json!([4, 4]));
}

#[test]
fn test_group_chunks_preamble_prepended_to_every_chunk() {
    let text = "\
1. Unit one.
2. Unit two.
3. Unit three.";

    let splitter = StructureAwareSplitter::from_config(paragraph_config(2));
    let units = splitter.detect_units(text);
    let preamble = "CASE: Awad v. CFS\nFiled: 2025";
    let chunks = splitter.group_into_chunks(&units, preamble);

    assert_eq!(chunks.len(), 2);
    for chunk in &chunks {
        assert!(
            chunk.text.starts_with("CASE: Awad v. CFS"),
            "Chunk {} must start with preamble",
            chunk.index
        );
        assert_eq!(chunk.metadata["preamble_included"], json!(true));
    }
}

#[test]
fn test_group_chunks_no_preamble() {
    let text = "1. Only unit.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let units = splitter.detect_units(text);
    let chunks = splitter.group_into_chunks(&units, "");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].metadata["preamble_included"], json!(false));
    assert!(
        !chunks[0].metadata.contains_key("preamble_length_chars"),
        "No preamble_length_chars when preamble is empty"
    );
}

#[test]
fn test_group_chunks_with_overlap() {
    let text = "\
1. A.
2. B.
3. C.
4. D.
5. E.
6. F.";

    let mut config = paragraph_config(3);
    config.insert("unit_overlap".to_string(), json!(1));
    let splitter = StructureAwareSplitter::from_config(config);
    let units = splitter.detect_units(text);
    let chunks = splitter.group_into_chunks(&units, "");

    assert_eq!(chunks.len(), 3, "6 units, step 2 → 3 chunks");
    assert_eq!(chunks[0].metadata["unit_range"], json!([0, 2]));
    assert_eq!(chunks[1].metadata["unit_range"], json!([2, 4]));
    assert_eq!(chunks[2].metadata["unit_range"], json!([4, 5]));

    assert!(chunks[0].text.contains("3. C."), "Unit 2 must be in chunk 0");
    assert!(
        chunks[1].text.contains("3. C."),
        "Unit 2 must also be in chunk 1 (overlap)"
    );
}

#[test]
fn test_group_chunks_overlap_exceeds_chunk_size_degrades() {
    let text = "1. A.\n2. B.\n3. C.\n4. D.";

    let mut config = paragraph_config(2);
    config.insert("unit_overlap".to_string(), json!(5));
    let splitter = StructureAwareSplitter::from_config(config);
    let units = splitter.detect_units(text);
    let chunks = splitter.group_into_chunks(&units, "");

    assert_eq!(chunks.len(), 2, "Must produce chunks, not infinite loop");
    assert_eq!(chunks[0].metadata["unit_range"], json!([0, 1]));
    assert_eq!(chunks[1].metadata["unit_range"], json!([2, 3]));
}

#[test]
fn test_group_chunks_single_unit_per_chunk() {
    let text = "1. Alpha.\n2. Beta.\n3. Gamma.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(1));
    let units = splitter.detect_units(text);
    let chunks = splitter.group_into_chunks(&units, "");

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].metadata["unit_count"], json!(1));
    assert_eq!(chunks[1].metadata["unit_count"], json!(1));
    assert_eq!(chunks[2].metadata["unit_count"], json!(1));
}

#[test]
fn test_group_chunks_all_units_fit_in_one_chunk() {
    let text = "1. Alpha.\n2. Beta.\n3. Gamma.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(100));
    let units = splitter.detect_units(text);
    let chunks = splitter.group_into_chunks(&units, "");

    assert_eq!(chunks.len(), 1, "All units fit in one chunk");
    assert_eq!(chunks[0].metadata["unit_count"], json!(3));
    assert_eq!(chunks[0].metadata["unit_range"], json!([0, 2]));
}

// ---------------------------------------------------------------------------
// split() (TextSplitter trait) integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_split_empty_text_returns_empty() {
    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let chunks = splitter.split("");
    assert!(chunks.is_empty(), "Empty text must produce no chunks");
}

#[test]
fn test_split_no_structure_returns_fallback_chunk() {
    let text = "This is plain text with no numbered items or structure.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(25));
    let chunks = splitter.split(text);

    assert_eq!(
        chunks.len(),
        1,
        "No structure must produce exactly 1 fallback chunk"
    );
    assert_eq!(chunks[0].text, text, "Fallback chunk must contain the entire text");
    assert_eq!(chunks[0].metadata["fallback"], json!(true));
    assert_eq!(chunks[0].metadata["reason"], json!("no_boundary_matches"));
}

#[test]
fn test_split_end_to_end_discovery_response() {
    let text = "\
IN THE CIRCUIT COURT
Case No. 2024-12345

1. What is your role at CFS?
Answer: I am the director of operations.
2. How long have you held this position?
Answer: Since January 2020.
3. Describe your duties.
Answer: I oversee daily operations and staff management.
4. Were you aware of the complaint?
Answer: I became aware in March 2024.
5. What actions did you take?
Answer: I referred the matter to legal counsel.";

    let splitter = StructureAwareSplitter::from_config(qa_config(3));
    let chunks = splitter.split(text);

    assert_eq!(chunks.len(), 2, "5 QA pairs / 3 per chunk = 2 chunks");

    assert!(
        chunks[0].text.contains("IN THE CIRCUIT COURT"),
        "Chunk 0 must include preamble"
    );
    assert!(
        chunks[0].text.contains("What is your role"),
        "Chunk 0 must include Q1"
    );
    assert!(
        chunks[0].text.contains("Describe your duties"),
        "Chunk 0 must include Q3"
    );
    assert!(
        !chunks[0].text.contains("Were you aware"),
        "Chunk 0 must NOT include Q4"
    );

    assert!(
        chunks[1].text.contains("IN THE CIRCUIT COURT"),
        "Chunk 1 must include preamble (prepended to every chunk)"
    );
    assert!(
        chunks[1].text.contains("Were you aware"),
        "Chunk 1 must include Q4"
    );
    assert!(
        chunks[1].text.contains("referred the matter"),
        "Chunk 1 must include Q5's answer"
    );

    assert_eq!(chunks[0].metadata["unit_range"], json!([0, 2]));
    assert_eq!(chunks[1].metadata["unit_range"], json!([3, 4]));
    assert_eq!(chunks[0].metadata["preamble_included"], json!(true));
    assert_eq!(chunks[1].metadata["preamble_included"], json!(true));
}

#[test]
fn test_split_end_to_end_complaint_paragraphs() {
    let mut paragraphs = Vec::new();
    for i in 1..=10 {
        paragraphs.push(format!(
            "{i}. Plaintiff alleges that defendant committed act number {i} \
            causing harm and damages as described herein."
        ));
    }
    let text = paragraphs.join("\n");

    let splitter = StructureAwareSplitter::from_config(paragraph_config(4));
    let chunks = splitter.split(&text);

    assert_eq!(chunks.len(), 3, "10 paragraphs / 4 per chunk = 3 chunks");
    assert_eq!(chunks[0].metadata["unit_count"], json!(4));
    assert_eq!(chunks[1].metadata["unit_count"], json!(4));
    assert_eq!(chunks[2].metadata["unit_count"], json!(2));

    for i in 1..=10 {
        // Trailing space disambiguates "act number 1 " from "act number 10 "
        // — the spec's assertion as written collides on substring prefixes.
        let needle = format!("act number {i} ");
        let containing_chunks: Vec<_> = chunks.iter().filter(|c| c.text.contains(&needle)).collect();
        assert_eq!(
            containing_chunks.len(),
            1,
            "Paragraph {i} must appear in exactly 1 chunk (no overlap configured)"
        );
    }
}

#[test]
fn test_split_section_heading_pattern() {
    let text = "\
FINDINGS OF FACT
The court finds the following facts established by the evidence.

ORDER
IT IS HEREBY ORDERED that the motion is granted.

OPINION
The court's analysis follows.";

    let mut config = HashMap::new();
    config.insert(
        "boundary_pattern".to_string(),
        json!(r"^(?:FINDINGS OF FACT|ORDER|OPINION)"),
    );
    config.insert("units_per_chunk".to_string(), json!(2));
    let splitter = StructureAwareSplitter::from_config(config);
    let chunks = splitter.split(text);

    assert_eq!(chunks.len(), 2, "3 sections / 2 per chunk = 2 chunks");
    assert!(
        chunks[0].text.contains("FINDINGS OF FACT"),
        "Chunk 0 must have first section"
    );
    assert!(chunks[0].text.contains("ORDER"), "Chunk 0 must have second section");
    assert!(chunks[1].text.contains("OPINION"), "Chunk 1 must have third section");
}

#[test]
fn test_split_unit_identifiers_in_metadata() {
    let text = "1. First.\n2. Second.\n3. Third.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(2));
    let chunks = splitter.split(text);

    assert_eq!(chunks.len(), 2);
    let ids = chunks[0]
        .metadata
        .get("unit_identifiers")
        .expect("unit_identifiers must be present");
    assert_eq!(ids, &json!(["1", "2"]));

    let ids2 = chunks[1]
        .metadata
        .get("unit_identifiers")
        .expect("unit_identifiers must be present");
    assert_eq!(ids2, &json!(["3"]));
}

#[test]
fn test_split_chunk_indices_are_sequential() {
    let text = "1. A.\n2. B.\n3. C.\n4. D.\n5. E.";
    let splitter = StructureAwareSplitter::from_config(paragraph_config(2));
    let chunks = splitter.split(text);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.index, i,
            "Chunk indices must be sequential 0..N, got {} at position {}",
            chunk.index, i
        );
    }
}
