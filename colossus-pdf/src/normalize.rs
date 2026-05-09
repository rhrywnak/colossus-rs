//! Post-extraction text normalization.
//!
//! Applied after any extractor produces text, before the text is
//! stored in the pipeline database. Fixes known artifacts from
//! PDF text extraction (spacing issues, excessive blank lines, etc.)
//!
//! ## Design: Why normalize AFTER extraction, not during?
//!
//! Each extractor produces the best text it can from the source format.
//! Normalization is format-independent cleanup — the same rules apply
//! whether the text came from a PDF, DOCX, or TXT file. Keeping it
//! separate means:
//! 1. Extractors stay focused on their format
//! 2. Normalization rules are testable independently
//! 3. Rules can be enabled/disabled per-deployment via config
//!
//! ## Rust Learning: Function composition for text transforms
//!
//! Each normalization rule is a standalone function that takes `&str`
//! and returns `String`. They're composed by chaining: the output of
//! one feeds into the next. This makes rules independently testable
//! and easy to reorder.

use regex::Regex;
use std::sync::LazyLock;

/// Apply all enabled normalization rules to the text.
///
/// The `rules` parameter controls which rules run. Pass an empty
/// slice to skip normalization entirely (useful for debugging).
///
/// Rules are applied in the order listed in the `rules` slice.
pub fn normalize_text(text: &str, rules: &[NormalizationRule]) -> String {
    let mut result = text.to_string();
    for rule in rules {
        result = rule.apply(&result);
    }
    result
}

/// A named normalization rule.
///
/// Each variant maps to a single text transform. The variant is the
/// stable identifier used in config files; the transform itself is
/// looked up in `apply` so callers never need to know the function name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationRule {
    /// Fix missing space between paragraph number and text.
    /// Transforms `10.During` → `10. During` at line starts.
    NumberedParagraphSpacing,

    /// Collapse 3+ consecutive newlines into 2.
    CollapseBlankLines,

    /// Remove trailing whitespace from each line.
    TrimTrailingWhitespace,
}

impl NormalizationRule {
    /// Apply this rule to the text.
    pub fn apply(&self, text: &str) -> String {
        match self {
            Self::NumberedParagraphSpacing => fix_numbered_paragraph_spacing(text),
            Self::CollapseBlankLines => collapse_blank_lines(text),
            Self::TrimTrailingWhitespace => trim_trailing_whitespace(text),
        }
    }

    /// All available rules in their recommended application order.
    ///
    /// Order matters: spacing fixes run before whitespace trimming so
    /// that newly-introduced spaces between paragraph numbers and
    /// words aren't immediately stripped if they happen to land at a
    /// line end.
    pub fn all() -> Vec<NormalizationRule> {
        vec![
            Self::NumberedParagraphSpacing,
            Self::CollapseBlankLines,
            Self::TrimTrailingWhitespace,
        ]
    }
}

/// Fix missing space between paragraph number and text at line starts.
///
/// PDF text extraction often produces `10.During` instead of
/// `10. During` due to font metric artifacts in the source PDF.
/// This rule inserts the missing space.
///
/// Pattern: `^\d+\.[A-Z]` → `\d+. [A-Z]`
///
/// The pattern requires an uppercase letter after the dot to avoid
/// false positives on decimal numbers (e.g., `$50,000.00` should
/// not become `$50,000. 00`).
///
/// ## Rust Learning: `LazyLock<Regex>` for one-time compile
///
/// `LazyLock` (stable since 1.80) replaces the older `lazy_static!`
/// macro. The closure runs at most once, on first access, and the
/// result is cached for the rest of the process. Thread-safe by
/// construction. We `expect` here because the literal regex is a
/// programmer-controlled constant — a compile failure is a code bug,
/// not a recoverable runtime condition.
fn fix_numbered_paragraph_spacing(text: &str) -> String {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)^(\d+\.)([A-Z])").expect("numbered paragraph regex"));
    RE.replace_all(text, "$1 $2").into_owned()
}

/// Collapse 3+ consecutive newlines into 2 (one blank line).
fn collapse_blank_lines(text: &str) -> String {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\n{3,}").expect("blank lines regex"));
    RE.replace_all(text, "\n\n").into_owned()
}

/// Remove trailing whitespace from each line.
fn trim_trailing_whitespace(text: &str) -> String {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)[ \t]+$").expect("trailing whitespace regex"));
    RE.replace_all(text, "").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_numbered_paragraph_spacing_cases() {
        // (input, expected) — pin both the FIX cases and the PRESERVE
        // (no-modification) cases. Lowercase-after-dot guards against
        // false positives on abbreviations (e.g., "i.e."); decimal
        // guard prevents "$50,000.00" → "$50,000. 00".
        let cases: &[(&str, &str)] = &[
            // basic fix
            (
                "10.During the pendency of the guardianship",
                "10. During the pendency of the guardianship",
            ),
            // already-spaced preserved
            (
                "10. During the pendency of the guardianship",
                "10. During the pendency of the guardianship",
            ),
            // decimal guard
            (
                "The amount was $50,000.00 in total",
                "The amount was $50,000.00 in total",
            ),
            // multi-line
            (
                "10.During the case\n11.Although the court\n12.It was revealed",
                "10. During the case\n11. Although the court\n12. It was revealed",
            ),
            // 3-digit number
            (
                "126.Plaintiff has been harmed",
                "126. Plaintiff has been harmed",
            ),
            // lowercase-after-dot guard
            ("1.example text", "1.example text"),
        ];

        for (input, expected) in cases {
            let result = fix_numbered_paragraph_spacing(input);
            assert_eq!(result, *expected, "case: {input:?}");
        }
    }

    #[test]
    fn collapse_blank_lines_reduces_to_two() {
        let input = "paragraph one\n\n\n\n\nparagraph two";
        let result = collapse_blank_lines(input);
        assert_eq!(result, "paragraph one\n\nparagraph two");
    }

    #[test]
    fn collapse_blank_lines_preserves_single_blanks() {
        let input = "paragraph one\n\nparagraph two";
        let result = collapse_blank_lines(input);
        assert_eq!(result, "paragraph one\n\nparagraph two");
    }

    #[test]
    fn trim_trailing_whitespace_removes_spaces() {
        let input = "line one   \nline two\t\nline three";
        let result = trim_trailing_whitespace(input);
        assert_eq!(result, "line one\nline two\nline three");
    }

    #[test]
    fn normalize_text_applies_all_rules_in_order() {
        let input = "10.During the case   \n\n\n\n11.Although the court";
        let result = normalize_text(input, &NormalizationRule::all());
        assert_eq!(result, "10. During the case\n\n11. Although the court");
    }

    #[test]
    fn normalize_text_with_empty_rules_is_passthrough() {
        let input = "10.During the case";
        let result = normalize_text(input, &[]);
        assert_eq!(result, "10.During the case");
    }
}
