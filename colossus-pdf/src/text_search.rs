//! Search for text across all pages of a PDF.
//!
//! Provides a simple full-text search with configurable case sensitivity,
//! context extraction around each hit, and result limiting.

use crate::error::PdfError;
use crate::extractor::PdfTextExtractor;

/// A single search hit within the PDF.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchHit {
    /// 1-indexed page number.
    pub page_number: u32,
    /// Character offset within the page text.
    pub offset: usize,
    /// A snippet of surrounding context (e.g., 50 chars before and after).
    pub context: String,
}

/// Configuration for text search.
///
/// # Rust Learning: `Default` trait
///
/// Implementing `Default` lets callers write `SearchConfig::default()` to
/// get sensible defaults. This works well with struct update syntax:
/// ```rust,ignore
/// let config = SearchConfig { case_insensitive: false, ..Default::default() };
/// ```
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Whether to ignore case when matching.
    pub case_insensitive: bool,
    /// How many characters of context to include around each hit.
    pub context_chars: usize,
    /// Maximum number of results to return (0 = unlimited).
    pub max_results: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            case_insensitive: true,
            context_chars: 50,
            max_results: 0,
        }
    }
}

/// Extract surrounding context for a match at `offset` within `text`.
///
/// Takes up to `context_chars` characters before and after the match,
/// clamping to the text boundaries.
fn extract_context(text: &str, offset: usize, query_len: usize, context_chars: usize) -> String {
    let start = offset.saturating_sub(context_chars);
    let end = (offset + query_len + context_chars).min(text.len());

    // Clamp to char boundaries to avoid panics on multi-byte UTF-8
    let start = text
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| i >= start)
        .unwrap_or(0);

    let end = text
        .char_indices()
        .map(|(i, _)| i)
        .rfind(|&i| i <= end)
        .unwrap_or(text.len());

    let mut context = String::new();
    if start > 0 {
        context.push_str("...");
    }
    context.push_str(&text[start..end]);
    if end < text.len() {
        context.push_str("...");
    }
    context
}

/// Search for a text pattern across all pages.
///
/// Returns all occurrences of `query` across the document, with surrounding
/// context for each hit. Respects `config.max_results` to limit output.
pub fn search_text(
    extractor: &mut PdfTextExtractor,
    query: &str,
    config: &SearchConfig,
) -> Result<Vec<SearchHit>, PdfError> {
    let page_count = extractor.page_count();
    let mut hits = Vec::new();

    let query_normalized = if config.case_insensitive {
        query.to_lowercase()
    } else {
        query.to_owned()
    };

    for page_num in 1..=page_count {
        let page_text = extractor.extract_page(page_num)?;

        let search_text = if config.case_insensitive {
            page_text.to_lowercase()
        } else {
            page_text.to_owned()
        };

        // Find all occurrences on this page
        let mut search_start = 0;
        while let Some(pos) = search_text[search_start..].find(&query_normalized) {
            let absolute_offset = search_start + pos;

            let context = extract_context(
                page_text,
                absolute_offset,
                query.len(),
                config.context_chars,
            );

            hits.push(SearchHit {
                page_number: page_num,
                offset: absolute_offset,
                context,
            });

            // Check max_results limit
            if config.max_results > 0 && hits.len() >= config.max_results {
                return Ok(hits);
            }

            // Move past this match to find the next one
            search_start = absolute_offset + query_normalized.len();
        }
    }

    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a 3-page test PDF with known content.
    fn create_test_pdf() -> PdfTextExtractor {
        use pdf_oxide::writer::DocumentBuilder;

        let mut builder = DocumentBuilder::new();

        builder
            .letter_page()
            .heading(1, "Test Document")
            .paragraph("This is page one with some sample text.")
            .done();

        builder
            .letter_page()
            .heading(1, "Chapter Two")
            .paragraph("The quick brown fox jumps over the lazy dog.")
            .done();

        builder
            .letter_page()
            .heading(1, "Final Page")
            .paragraph("Conclusion and summary of findings.")
            .done();

        let bytes = builder.build().expect("Failed to build test PDF");
        PdfTextExtractor::from_bytes(bytes).expect("Failed to open test PDF")
    }

    #[test]
    fn test_search_finds_hits_with_correct_pages() {
        let mut extractor = create_test_pdf();
        let config = SearchConfig::default();

        let hits = search_text(&mut extractor, "page", &config)
            .expect("search failed");

        // "page" appears on page 1 ("page one") and page 3 ("Final Page")
        assert!(!hits.is_empty(), "Should find at least one hit for 'page'");

        // All hits should have valid page numbers
        for hit in &hits {
            assert!(hit.page_number >= 1 && hit.page_number <= 3);
        }
    }

    #[test]
    fn test_case_insensitive_search() {
        let mut extractor = create_test_pdf();
        let config = SearchConfig::default(); // case_insensitive: true

        let hits = search_text(&mut extractor, "CHAPTER", &config)
            .expect("search failed");

        assert!(
            !hits.is_empty(),
            "Case-insensitive search for 'CHAPTER' should find 'Chapter'"
        );
        assert_eq!(hits[0].page_number, 2);
    }

    #[test]
    fn test_context_extraction() {
        let mut extractor = create_test_pdf();
        let config = SearchConfig {
            context_chars: 20,
            ..Default::default()
        };

        let hits = search_text(&mut extractor, "quick", &config)
            .expect("search failed");

        assert!(!hits.is_empty());
        // Context should contain the matched word plus surrounding text
        assert!(
            hits[0].context.contains("quick"),
            "Context should contain the search term"
        );
    }

    #[test]
    fn test_max_results_limit() {
        let mut extractor = create_test_pdf();
        let config = SearchConfig {
            max_results: 1,
            ..Default::default()
        };

        // Search for something that appears on multiple pages
        let hits = search_text(&mut extractor, "page", &config)
            .expect("search failed");

        // Even if "page" appears many times, we should get at most 1
        assert!(hits.len() <= 1, "Should respect max_results=1");
    }
}
