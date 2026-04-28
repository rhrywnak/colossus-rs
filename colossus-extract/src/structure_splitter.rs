//! Structure-aware text splitter.
//!
//! Detects atomic units in document text using configured regex boundary
//! patterns, groups those units into chunks, and prepends a preamble (text
//! before the first detected unit) to every chunk so downstream consumers
//! always have document-level context.
//!
//! ## Rust Learning: Configuration over compile-time types
//!
//! The boundary regex and grouping parameters live in a
//! `HashMap<String, serde_json::Value>` rather than struct fields. This
//! lets a YAML profile change splitter behavior without recompiling and
//! keeps the splitter free of any domain-specific knowledge — the caller
//! supplies the patterns; the splitter just applies them.

use crate::config::ConfigAccess;
use crate::traits::TextSplitter;
use crate::types::{AtomicUnit, TextChunk};
use regex::Regex;
use serde_json::json;
use std::collections::HashMap;

/// Default boundary pattern when none is supplied via config — matches the
/// start of a numbered item (`1. `, `2. `, ...).
const DEFAULT_BOUNDARY_PATTERN: &str = r"^\d+\.\s";
/// Default number of atomic units grouped into a single chunk.
const DEFAULT_UNITS_PER_CHUNK: i64 = 25;
/// Default number of atomic units repeated at the start of the next chunk
/// for context continuity.
const DEFAULT_UNIT_OVERLAP: i64 = 0;
/// Multiline regex prefix — `^` matches the start of every line, not only
/// the start of the input.
const MULTILINE_FLAG_PREFIX: &str = "(?m)";
/// Joiner placed between the preamble and unit texts when assembling a chunk.
const CHUNK_JOIN: &str = "\n\n";

/// Splits document text at structural boundaries detected by regex patterns.
///
/// Unlike `FixedSizeSplitter` which splits at character counts with
/// word-boundary awareness, this splitter understands document structure:
/// it detects atomic units (Q&A pairs, numbered paragraphs, sections) and
/// groups them into chunks that respect those boundaries. An atomic unit
/// is never split across chunks.
///
/// ## Configuration (via `HashMap<String, serde_json::Value>`)
///
/// - `boundary_pattern` (string, optional): regex matching the start of an
///   atomic unit. Default: `^\d+\.\s`. Compiled with the multiline flag so
///   `^` matches every line start.
/// - `response_marker` (string, optional): regex matching the start of a
///   response section within a unit. Metadata-only — does NOT change unit
///   boundaries. When present, each unit's metadata gains `has_response`
///   indicating whether the marker matched within the unit text.
/// - `units_per_chunk` (i64, default 25): how many atomic units per chunk.
/// - `unit_overlap` (i64, default 0): how many trailing units of one chunk
///   are repeated at the start of the next.
///
/// ## Rust Learning: Why regex is compiled in `split`, not stored
///
/// `Regex` is not `Serialize`/`Deserialize` and adds lifetime complexity.
/// `split` is called once per document, so the one-time compile cost is
/// negligible. Keeping the struct purely data makes it trivial to clone,
/// log, and pass between async tasks.
pub struct StructureAwareSplitter {
    config: HashMap<String, serde_json::Value>,
}

impl StructureAwareSplitter {
    /// Build a splitter from a flexible configuration map.
    ///
    /// The config is not validated here — an invalid regex causes
    /// `detect_units` to return an empty Vec, which `split` turns into a
    /// fallback single chunk with a `reason` recorded in metadata. This
    /// graceful degradation keeps a misconfigured profile from crashing
    /// the pipeline.
    pub fn from_config(config: HashMap<String, serde_json::Value>) -> Self {
        Self { config }
    }

    /// Detect atomic units in `text` using the configured boundary pattern
    /// and optional response marker.
    ///
    /// Returns an empty `Vec` when:
    /// - The boundary regex is missing or invalid, OR
    /// - The regex compiles but matches nothing.
    ///
    /// In both cases the caller (`split`) emits a fallback chunk.
    pub fn detect_units(&self, text: &str) -> Vec<AtomicUnit> {
        let pattern_str = self.config.get_str("boundary_pattern", DEFAULT_BOUNDARY_PATTERN);

        let boundary_re =
            match Regex::new(&format!("{MULTILINE_FLAG_PREFIX}{pattern_str}")) {
                Ok(re) => re,
                Err(_) => return Vec::new(),
            };

        let response_re = self
            .config
            .get("response_marker")
            .and_then(|v| v.as_str())
            .and_then(|s| Regex::new(&format!("{MULTILINE_FLAG_PREFIX}{s}")).ok());

        let matches: Vec<_> = boundary_re.find_iter(text).collect();
        if matches.is_empty() {
            return Vec::new();
        }

        let mut units = Vec::with_capacity(matches.len());

        for (i, m) in matches.iter().enumerate() {
            let start = m.start();
            let end = if i + 1 < matches.len() {
                matches[i + 1].start()
            } else {
                text.len()
            };

            let unit_text = text[start..end].trim_end().to_string();

            let mut metadata = HashMap::new();
            let matched_prefix = m.as_str().trim();
            if !matched_prefix.is_empty() {
                metadata.insert(
                    "identifier".to_string(),
                    json!(matched_prefix.trim_end_matches('.')),
                );
            }

            if let Some(ref resp_re) = response_re {
                metadata.insert("has_response".to_string(), json!(resp_re.is_match(&unit_text)));
            }

            units.push(AtomicUnit {
                text: unit_text,
                index: i,
                start_offset: start,
                end_offset: end,
                metadata,
            });
        }

        units
    }

    /// Group atomic units into chunks, prepending `preamble` to every chunk.
    ///
    /// - `units_per_chunk` and `unit_overlap` are read from config.
    /// - When `unit_overlap >= units_per_chunk` the overlap is treated as
    ///   a configuration error and the step degrades to `units_per_chunk`
    ///   (no overlap, no infinite loop).
    /// - `preamble` is trimmed; if non-empty it is included in every chunk
    ///   and `preamble_included` / `preamble_length_chars` are recorded in
    ///   chunk metadata.
    pub fn group_into_chunks(&self, units: &[AtomicUnit], preamble: &str) -> Vec<TextChunk> {
        if units.is_empty() {
            return Vec::new();
        }

        let units_per_chunk = self
            .config
            .get_i64("units_per_chunk", DEFAULT_UNITS_PER_CHUNK)
            .max(1) as usize;
        let unit_overlap = self
            .config
            .get_i64("unit_overlap", DEFAULT_UNIT_OVERLAP)
            .max(0) as usize;

        let step = if unit_overlap >= units_per_chunk {
            units_per_chunk
        } else {
            units_per_chunk - unit_overlap
        };

        let preamble_trimmed = preamble.trim();
        let has_preamble = !preamble_trimmed.is_empty();

        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut start = 0;

        while start < units.len() {
            let end = (start + units_per_chunk).min(units.len());
            let chunk_units = &units[start..end];

            let mut parts = Vec::with_capacity(chunk_units.len() + 1);
            if has_preamble {
                parts.push(preamble_trimmed.to_string());
            }
            for unit in chunk_units {
                parts.push(unit.text.clone());
            }
            let chunk_text = parts.join(CHUNK_JOIN);

            let unit_identifiers: Vec<serde_json::Value> = chunk_units
                .iter()
                .filter_map(|u| u.metadata.get("identifier").cloned())
                .collect();

            let mut metadata = HashMap::new();
            metadata.insert("unit_range".to_string(), json!([start, end - 1]));
            metadata.insert("unit_count".to_string(), json!(chunk_units.len()));
            metadata.insert("preamble_included".to_string(), json!(has_preamble));
            if has_preamble {
                metadata.insert(
                    "preamble_length_chars".to_string(),
                    json!(preamble_trimmed.len()),
                );
            }
            if !unit_identifiers.is_empty() {
                metadata.insert("unit_identifiers".to_string(), json!(unit_identifiers));
            }

            chunks.push(TextChunk {
                text: chunk_text,
                index: chunk_index,
                metadata,
            });

            chunk_index += 1;
            start += step;
        }

        chunks
    }
}

impl TextSplitter for StructureAwareSplitter {
    /// Split `text` into structure-aware chunks.
    ///
    /// Algorithm:
    /// 1. Empty input → empty Vec.
    /// 2. Run `detect_units`. If it returns no units (no matches OR invalid
    ///    regex), emit a single fallback chunk that contains the entire
    ///    input and records `fallback: true` / `reason: "no_boundary_matches"`
    ///    in metadata.
    /// 3. Otherwise, treat text before the first unit's `start_offset` as
    ///    preamble and delegate to `group_into_chunks`.
    fn split(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return vec![];
        }

        let units = self.detect_units(text);

        if units.is_empty() {
            return vec![TextChunk {
                text: text.to_string(),
                index: 0,
                metadata: HashMap::from([
                    ("fallback".to_string(), json!(true)),
                    ("reason".to_string(), json!("no_boundary_matches")),
                ]),
            }];
        }

        let preamble = if units[0].start_offset > 0 {
            &text[..units[0].start_offset]
        } else {
            ""
        };

        self.group_into_chunks(&units, preamble)
    }
}
