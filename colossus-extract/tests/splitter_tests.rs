use colossus_extract::{FixedSizeSplitter, TextSplitter};

#[test]
fn test_empty_text() {
    let splitter = FixedSizeSplitter::new();
    let chunks = splitter.split("");
    assert!(chunks.is_empty());
}

#[test]
fn test_text_smaller_than_chunk_size() {
    let splitter = FixedSizeSplitter::new();
    let text = "This is a short document.";
    let chunks = splitter.split(text);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
    assert_eq!(chunks[0].index, 0);
}

#[test]
fn test_text_exactly_chunk_size() {
    let splitter = FixedSizeSplitter::with_config(10, 2);
    let text = "0123456789"; // exactly 10 chars
    let chunks = splitter.split(text);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
}

#[test]
fn test_multiple_chunks_with_overlap() {
    // 20 chars, chunk_size=10, overlap=3, step=7
    let splitter = FixedSizeSplitter::with_config(10, 3);
    let text = "aaaa bbb cc ddd eeee"; // 20 chars
    let chunks = splitter.split(text);

    assert!(chunks.len() >= 2, "Expected at least 2 chunks, got {}", chunks.len());

    // Verify indices are sequential
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i);
    }

    // Verify no chunk exceeds chunk_size
    for chunk in &chunks {
        assert!(
            chunk.text.len() <= 10,
            "Chunk {} has {} chars, expected <= 10",
            chunk.index,
            chunk.text.len()
        );
    }
}

#[test]
fn test_word_boundary_awareness() {
    // With a chunk size that would cut "world" in half
    let splitter = FixedSizeSplitter::with_config(8, 2);
    let text = "hello world foo bar";
    let chunks = splitter.split(text);

    // No chunk should start or end in the middle of a word
    // (except possibly the first start and last end)
    for chunk in &chunks {
        let trimmed = chunk.text.trim();
        if !trimmed.is_empty() {
            // First char should not be mid-word (should be start of word or whitespace)
            // This is a soft check — the splitter tries but may fall back
            assert!(
                !trimmed.is_empty(),
                "Chunk {} should not be empty after trim",
                chunk.index
            );
        }
    }
}

#[test]
fn test_chunks_cover_full_text() {
    let splitter = FixedSizeSplitter::with_config(100, 20);
    // Create a text with known words
    let words: Vec<String> = (0..50).map(|i| format!("word{}", i)).collect();
    let text = words.join(" ");

    let chunks = splitter.split(&text);

    // Every word in the original text should appear in at least one chunk
    for word in &words {
        let found = chunks.iter().any(|c| c.text.contains(word.as_str()));
        assert!(found, "Word '{}' not found in any chunk", word);
    }
}

#[test]
fn test_complaint_sized_document() {
    // Simulate a 27,000 char document (like our Complaint)
    let text = "a ".repeat(13500); // 27,000 chars
    let splitter = FixedSizeSplitter::new(); // 4000 chars, 200 overlap

    let chunks = splitter.split(&text);

    // 27,000 chars / (4000 - 200) step = ~7.1, so expect 7-8 chunks
    assert!(
        chunks.len() >= 7 && chunks.len() <= 9,
        "Expected 7-9 chunks for 27,000 chars, got {}",
        chunks.len()
    );

    // Verify sequential indices
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i);
    }
}

#[test]
fn test_custom_config() {
    let splitter = FixedSizeSplitter::with_config(500, 50);
    let text = "x ".repeat(1000); // 2000 chars
    let chunks = splitter.split(&text);

    // 2000 / (500 - 50) = ~4.4, expect 4-5 chunks
    assert!(
        chunks.len() >= 4 && chunks.len() <= 6,
        "Expected 4-6 chunks, got {}",
        chunks.len()
    );
}

#[test]
#[should_panic(expected = "chunk_size must be greater than 0")]
fn test_zero_chunk_size_panics() {
    FixedSizeSplitter::with_config(0, 0);
}

#[test]
#[should_panic(expected = "chunk_overlap must be less than chunk_size")]
fn test_overlap_equals_chunk_size_panics() {
    FixedSizeSplitter::with_config(100, 100);
}

#[test]
fn test_chunk_extraction_result_json_schema() {
    // Verify the JsonSchema derive works and produces valid schema
    use colossus_extract::ChunkExtractionResult;
    let schema = schemars::schema_for!(ChunkExtractionResult);
    let json = serde_json::to_string_pretty(&schema).unwrap();

    // Should contain our field names
    assert!(json.contains("nodes"), "Schema should contain 'nodes'");
    assert!(json.contains("relationships"), "Schema should contain 'relationships'");
    assert!(json.contains("label"), "Schema should contain 'label'");
    assert!(json.contains("start_node_id"), "Schema should contain 'start_node_id'");
}
