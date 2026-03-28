//! Map text snippets to PDF page numbers.
//!
//! # Rust Learning: Builder pattern with references
//!
//! `PageGrounder` borrows the extractor rather than owning it,
//! using lifetime annotations. This lets you create a grounder,
//! use it, and still access the extractor afterward. The `'a`
//! lifetime says "this grounder cannot outlive the extractor
//! it borrows from."

use crate::error::PdfError;
use crate::extractor::PdfTextExtractor;

/// Result of grounding a single text snippet to a page.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GroundingResult {
    /// The original snippet that was searched for.
    pub snippet: String,
    /// The 1-indexed page number where it was found (None if not found).
    pub page_number: Option<u32>,
    /// Character offset within the page text where the match starts.
    pub offset: Option<usize>,
    /// Whether the match was exact or normalized (whitespace/case).
    pub match_type: MatchType,
}

/// How a snippet was matched against page text.
#[derive(Debug, Clone, serde::Serialize)]
pub enum MatchType {
    /// Exact substring match.
    Exact,
    /// Match after normalizing whitespace and case.
    Normalized,
    /// No match found.
    NotFound,
}

/// Collapse whitespace and lowercase for fuzzy matching.
///
/// This matches the normalization used by the frontend's `pdfHighlight.ts`:
/// split on whitespace, rejoin with single spaces, then lowercase.
fn normalize_text(text: &str) -> String {
    text.replace('¶', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Maps text snippets to the PDF pages they appear on.
///
/// # Rust Learning: Lifetime `'a`
///
/// The `'a` lifetime parameter ties this struct to the `PdfTextExtractor`
/// it borrows. Rust's borrow checker ensures you can't drop the extractor
/// while a `PageGrounder` still references it. This is compile-time safety
/// with zero runtime cost.
pub struct PageGrounder<'a> {
    extractor: &'a mut PdfTextExtractor,
}

impl<'a> PageGrounder<'a> {
    /// Create a new `PageGrounder` that borrows the given extractor.
    pub fn new(extractor: &'a mut PdfTextExtractor) -> Self {
        Self { extractor }
    }

    /// Ground a single snippet — find which page it appears on.
    ///
    /// Searches all pages sequentially. Returns the FIRST match.
    /// Tries exact match first, then normalized match.
    pub fn ground_snippet(&mut self, snippet: &str) -> Result<GroundingResult, PdfError> {
        let results = self.ground_snippets(&[snippet])?;

        // ground_snippets always returns one result per input snippet
        Ok(results.into_iter().next().unwrap_or(GroundingResult {
            snippet: snippet.to_owned(),
            page_number: None,
            offset: None,
            match_type: MatchType::NotFound,
        }))
    }

    /// Ground multiple snippets in a single pass.
    ///
    /// More efficient than calling `ground_snippet` repeatedly because
    /// it extracts each page's text once and searches all snippets
    /// against it before moving to the next page.
    ///
    /// # Rust Learning: Batch processing pattern
    ///
    /// Instead of N passes through all pages (one per snippet), we do
    /// one pass and check all unmatched snippets on each page. This
    /// turns O(snippets * pages) into O(pages + snippets * pages) in
    /// the worst case but avoids redundant page extraction.
    pub fn ground_snippets(
        &mut self,
        snippets: &[&str],
    ) -> Result<Vec<GroundingResult>, PdfError> {
        let page_count = self.extractor.page_count();

        // Initialize results — all start as NotFound
        let mut results: Vec<GroundingResult> = snippets
            .iter()
            .map(|s| GroundingResult {
                snippet: s.to_string(),
                page_number: None,
                offset: None,
                match_type: MatchType::NotFound,
            })
            .collect();

        // Track which snippets still need matching (by index)
        let mut unmatched: Vec<usize> = (0..snippets.len()).collect();

        for page_num in 1..=page_count {
            if unmatched.is_empty() {
                break; // All snippets found
            }

            let page_text = self.extractor.extract_page(page_num)?;
            let normalized_page = normalize_text(page_text);

            // Check each unmatched snippet against this page
            let mut newly_matched = Vec::new();

            for &idx in &unmatched {
                let snippet = snippets[idx];

                // Try exact match first
                if let Some(offset) = page_text.find(snippet) {
                    results[idx] = GroundingResult {
                        snippet: snippet.to_string(),
                        page_number: Some(page_num),
                        offset: Some(offset),
                        match_type: MatchType::Exact,
                    };
                    newly_matched.push(idx);
                    continue;
                }

                // Fall back to normalized match
                let normalized_snippet = normalize_text(snippet);
                if let Some(offset) = normalized_page.find(&normalized_snippet) {
                    results[idx] = GroundingResult {
                        snippet: snippet.to_string(),
                        page_number: Some(page_num),
                        offset: Some(offset),
                        match_type: MatchType::Normalized,
                    };
                    newly_matched.push(idx);
                }
            }

            // Remove matched snippets from the unmatched list
            for matched_idx in &newly_matched {
                unmatched.retain(|&i| i != *matched_idx);
            }
        }

        // Log warnings for snippets that weren't found
        for &idx in &unmatched {
            let truncated: String = snippets[idx].chars().take(80).collect();
            tracing::warn!("Snippet not found in PDF: {truncated}");
        }

        Ok(results)
    }
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
    fn test_ground_known_snippet() {
        let mut extractor = create_test_pdf();
        let mut grounder = PageGrounder::new(&mut extractor);

        let result = grounder
            .ground_snippet("quick brown fox")
            .expect("ground_snippet failed");

        assert_eq!(result.page_number, Some(2));
        assert!(result.offset.is_some());
        assert!(matches!(result.match_type, MatchType::Exact));
    }

    #[test]
    fn test_ground_multiple_snippets() {
        let mut extractor = create_test_pdf();
        let mut grounder = PageGrounder::new(&mut extractor);

        let results = grounder
            .ground_snippets(&["Test Document", "quick brown fox", "Conclusion"])
            .expect("ground_snippets failed");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].page_number, Some(1));
        assert_eq!(results[1].page_number, Some(2));
        assert_eq!(results[2].page_number, Some(3));
    }

    #[test]
    fn test_snippet_not_found() {
        let mut extractor = create_test_pdf();
        let mut grounder = PageGrounder::new(&mut extractor);

        let result = grounder
            .ground_snippet("this text does not exist anywhere")
            .expect("ground_snippet failed");

        assert_eq!(result.page_number, None);
        assert!(matches!(result.match_type, MatchType::NotFound));
    }

    #[test]
    fn test_normalized_matching() {
        let mut extractor = create_test_pdf();
        let mut grounder = PageGrounder::new(&mut extractor);

        // Extra whitespace + different case — should match via normalization
        let result = grounder
            .ground_snippet("QUICK  BROWN   FOX")
            .expect("ground_snippet failed");

        assert_eq!(result.page_number, Some(2));
        assert!(matches!(result.match_type, MatchType::Normalized));
    }

    #[test]
    fn test_normalize_pilcrow() {
        assert_eq!(
            normalize_text("is a ¶¶Michigan metropolitan"),
            "is a michigan metropolitan"
        );
    }

    #[test]
    fn test_exact_match_preferred_over_normalized() {
        let mut extractor = create_test_pdf();
        let mut grounder = PageGrounder::new(&mut extractor);

        // This exact text should exist on page 1
        let result = grounder
            .ground_snippet("Test Document")
            .expect("ground_snippet failed");

        assert_eq!(result.page_number, Some(1));
        assert!(matches!(result.match_type, MatchType::Exact));
    }
}
