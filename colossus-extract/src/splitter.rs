//! Text splitting implementations.
//!
//! ## Rust Learning: Module organization
//!
//! The trait lives in traits.rs (interface). The implementation lives
//! here (concrete struct). This separation means consumers depend on
//! the trait, not the implementation — they can swap in a different
//! splitter without touching their code.

use crate::traits::TextSplitter;
use crate::types::TextChunk;
use std::collections::HashMap;

/// Splits text into fixed-size chunks with configurable overlap.
///
/// Modeled after neo4j-graphrag-python's FixedSizeSplitter:
/// - chunk_size: number of characters per chunk (default 4000)
/// - chunk_overlap: characters shared between adjacent chunks (default 200)
/// - Word boundary awareness: avoids cutting words mid-token
///
/// ## Rust Learning: Builder pattern
///
/// `FixedSizeSplitter::new()` uses default values. You can customize
/// with `FixedSizeSplitter::with_config(chunk_size, overlap)`. This is
/// simpler than a full builder for a struct with only two settings.
pub struct FixedSizeSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl FixedSizeSplitter {
    /// Create a splitter with default settings (4000 chars, 200 overlap).
    pub fn new() -> Self {
        Self {
            chunk_size: 4000,
            chunk_overlap: 200,
        }
    }

    /// Create a splitter with custom chunk size and overlap.
    ///
    /// # Panics
    /// Panics if chunk_overlap >= chunk_size or chunk_size == 0.
    pub fn with_config(chunk_size: usize, chunk_overlap: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be greater than 0");
        assert!(
            chunk_overlap < chunk_size,
            "chunk_overlap must be less than chunk_size"
        );
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl Default for FixedSizeSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextSplitter for FixedSizeSplitter {
    fn split(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return vec![];
        }

        let text_len = text.len();

        // If text fits in one chunk, return it as-is
        if text_len <= self.chunk_size {
            return vec![TextChunk {
                text: text.to_string(),
                index: 0,
                metadata: HashMap::new(),
            }];
        }

        let step = self.chunk_size - self.chunk_overlap;
        let mut chunks = Vec::new();
        let mut index = 0;
        let mut start = 0;

        while start < text_len {
            // Calculate approximate end
            let approx_end = (start + self.chunk_size).min(text_len);

            // Adjust end to avoid cutting a word (unless it's the end of text).
            // Start is left unadjusted — only the end walks back to a word
            // boundary, which guarantees end - start <= chunk_size.
            let end = if approx_end >= text_len {
                text_len
            } else {
                adjust_end(text, start, approx_end)
            };

            let chunk_text = &text[start..end];
            if !chunk_text.is_empty() {
                chunks.push(TextChunk {
                    text: chunk_text.to_string(),
                    index,
                    metadata: HashMap::new(),
                });
                index += 1;
            }

            // Move forward by step size
            if end >= text_len {
                break;
            }
            start += step;
        }

        chunks
    }
}

/// Shift end backward to the nearest word boundary.
/// If no whitespace found, return the original position.
fn adjust_end(text: &str, start: usize, approx_end: usize) -> usize {
    if approx_end >= text.len() {
        return text.len();
    }

    let bytes = text.as_bytes();

    // If we're already at a word boundary, no adjustment needed
    if bytes[approx_end].is_ascii_whitespace() || bytes[approx_end - 1].is_ascii_whitespace() {
        return approx_end;
    }

    // Walk backward to find whitespace
    let mut pos = approx_end;
    while pos > start && !bytes[pos - 1].is_ascii_whitespace() {
        pos -= 1;
    }

    // If we walked all the way back to start, use original position
    if pos == start {
        approx_end
    } else {
        pos
    }
}
