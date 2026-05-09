//! Provider factory functions: construct trait-object providers from environment
//! configuration.
//!
//! ## Design: `EnvLookup` abstraction for testability
//!
//! `std::env::var` is process-global mutable state. Rust's default test harness
//! runs tests in parallel within a binary. Writing factory tests with direct env
//! mutation therefore requires either the `serial_test` crate (adds a dependency
//! and serializes the entire test binary) or process isolation. Both are poor
//! trade-offs.
//!
//! Instead, the factory's core logic takes a lookup closure of type
//! `&dyn Fn(&str) -> Option<String>`. Production callers wrap `std::env::var`.
//! Tests supply a `HashMap`-backed closure, which is pure, parallelizable, and
//! requires no test ordering. This is also correct architecture independent of
//! testing: future callers (colossus-ai, config-file-based deployments) can
//! reuse the `_from_lookup` variants with their own config source.
//!
//! ## Design: Required model identifiers, no defaults
//!
//! `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` are required (no
//! defaults). Rationale: LLM model names change frequently (Anthropic deprecates
//! model IDs over time; vLLM models are deployment-specific). A stale default
//! silently pins deployments to an obsolete or wrong model. Forcing operators
//! to set them explicitly surfaces the choice at deployment time. This deviates
//! from v5_2 Parts 4.5 and 5.4 which showed a default of `claude-sonnet-4-6`;
//! the deviation is justified by the project's "no hardcoded defaults" memory
//! rule and the Anthropic SDK evolution pattern.
//!
//! ## Design: `FASTEMBED_MODEL` curated whitelist
//!
//! fastembed 4.9's `EmbeddingModel` enum exposes 44 variants. We accept a
//! curated subset of 11 text-embedding variants (BGE English family, AllMini,
//! Nomic, MultilingualE5 small/base, GTE base). Excluded variants include
//! `ClipVitB32` (image encoder, not text), `JinaEmbeddingsV2BaseCode` (code-
//! only), the 8 Snowflake Arctic variants (uncommon in production), and
//! several others.
//!
//! This curation follows the rig-fastembed 0.3.4 precedent, which exposes 30
//! of 44 variants in its own `FastembedModel` wrapper enum. Curation catches
//! obvious operator errors at config-load time rather than at embed time. To
//! add a variant: add a match arm in `parse_fastembed_model`, verify the model
//! dimension matches your Qdrant collection, add a test, and document the
//! choice.

use std::path::PathBuf;
use std::sync::Arc;

use fastembed::{EmbeddingModel, TokenizerFiles, UserDefinedEmbeddingModel};

use crate::error::PipelineError;
use crate::providers::{AnthropicProvider, FastembedProvider, VllmEmbeddingProvider, VllmProvider};
use crate::traits::{EmbeddingProvider, LlmProvider};

/// Closure type that abstracts environment variable lookup.
///
/// Production code uses [`std_env_lookup`]. Tests pass a `HashMap`-backed
/// closure to avoid touching process-global env state.
pub type EnvLookup<'a> = &'a dyn Fn(&str) -> Option<String>;

/// Production env lookup. Reads from `std::env::var`.
///
/// Returns `None` for both `VarError::NotPresent` and `VarError::NotUnicode`.
/// NotUnicode becoming None means downstream reports "required env var
/// missing" rather than a parse error — acceptable because all our env vars
/// should be ASCII, and a non-unicode value is operator error that surfaces
/// with the same remedy ("set it correctly").
fn std_env_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Default for `LLM_PROVIDER`. Matches v5_2 spec.
const DEFAULT_LLM_PROVIDER: &str = "anthropic";

/// Default for `EMBEDDING_PROVIDER`. Matches v5_2 spec.
const DEFAULT_EMBEDDING_PROVIDER: &str = "fastembed";

/// Default for `FASTEMBED_MODE` when `EMBEDDING_PROVIDER=fastembed`.
const DEFAULT_FASTEMBED_MODE: &str = "huggingface";

/// Default max tokens when `LLM_MAX_TOKENS` is not set. 32 000 suits modern
/// Claude Sonnet and typical vLLM deployments; caller can override via env.
const DEFAULT_LLM_MAX_TOKENS: u32 = 32_000;

// =========================================================================
// LLM provider factory
// =========================================================================

/// Construct an [`LlmProvider`] trait object from process environment.
///
/// Reads `std::env` directly. Tests and alternative config sources should use
/// [`llm_provider_from_lookup`] instead.
///
/// # Errors
///
/// Returns `PipelineError::LlmProvider` with a descriptive message pointing
/// at the relevant env var on any missing or invalid configuration.
pub fn llm_provider_from_env() -> Result<Arc<dyn LlmProvider>, PipelineError> {
    llm_provider_from_lookup(&std_env_lookup)
}

/// Testable core of [`llm_provider_from_env`]. Reads configuration via the
/// provided lookup closure rather than process environment.
///
/// Recognized `LLM_PROVIDER` values: `"anthropic"` (default), `"vllm"`.
///
/// # Errors
///
/// Returns `PipelineError::LlmProvider` on missing required variables, non-
/// numeric `LLM_MAX_TOKENS`, unknown `LLM_PROVIDER`, or provider construction
/// failure (e.g., invalid VLLM_BASE_URL shape).
pub fn llm_provider_from_lookup(
    lookup: EnvLookup<'_>,
) -> Result<Arc<dyn LlmProvider>, PipelineError> {
    let provider = lookup("LLM_PROVIDER").unwrap_or_else(|| DEFAULT_LLM_PROVIDER.to_string());

    match provider.as_str() {
        "anthropic" => build_anthropic(lookup),
        "vllm" => build_vllm_llm(lookup),
        other => Err(PipelineError::LlmProvider(format!(
            "Unknown LLM_PROVIDER: '{other}'. Valid values: anthropic, vllm"
        ))),
    }
}

fn build_anthropic(lookup: EnvLookup<'_>) -> Result<Arc<dyn LlmProvider>, PipelineError> {
    let api_key = lookup("ANTHROPIC_API_KEY").ok_or_else(|| {
        PipelineError::LlmProvider(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic".to_string(),
        )
    })?;

    let model = lookup("LLM_MODEL").ok_or_else(|| {
        PipelineError::LlmProvider(
            "LLM_MODEL is required when LLM_PROVIDER=anthropic (no default; set \
             explicitly to avoid pinning deployments to an obsolete model)"
                .to_string(),
        )
    })?;

    let max_tokens = parse_max_tokens(lookup)?;
    let temperature = parse_llm_temperature(lookup);

    let provider = AnthropicProvider::new(api_key, model, max_tokens, temperature, None)?;
    Ok(Arc::new(provider))
}

/// Parse `LLM_TEMPERATURE` into `Option<f64>`.
///
/// Rules:
/// - Unset → `None` (the API applies its default; required for Opus 4.7 where
///   sending the key at all triggers HTTP 400).
/// - Valid float → `Some(value)` (e.g., pipeline extraction sets `LLM_TEMPERATURE=0`
///   to preserve deterministic output).
/// - Unparseable → `None` with a warning log. An unparseable value is operator
///   error, but failing startup over a bad temperature string is worse than
///   falling through to the API default. Extraction workloads that require
///   deterministic output should set the value directly in code rather than
///   relying on env-var parsing.
fn parse_llm_temperature(lookup: EnvLookup<'_>) -> Option<f64> {
    match lookup("LLM_TEMPERATURE") {
        None => None,
        Some(raw) => match raw.parse::<f64>() {
            Ok(value) => Some(value),
            Err(e) => {
                tracing::warn!(
                    value = %raw,
                    error = %e,
                    "LLM_TEMPERATURE is not a valid f64 — falling back to provider default (None)"
                );
                None
            }
        },
    }
}

fn build_vllm_llm(lookup: EnvLookup<'_>) -> Result<Arc<dyn LlmProvider>, PipelineError> {
    let base_url = lookup("VLLM_BASE_URL").ok_or_else(|| {
        PipelineError::LlmProvider("VLLM_BASE_URL is required when LLM_PROVIDER=vllm".to_string())
    })?;

    let model = lookup("LLM_MODEL").ok_or_else(|| {
        PipelineError::LlmProvider("LLM_MODEL is required when LLM_PROVIDER=vllm".to_string())
    })?;

    let api_key = lookup("VLLM_API_KEY");
    let max_tokens = parse_max_tokens(lookup)?;

    let provider = VllmProvider::new(base_url, model, api_key, max_tokens, None)?;
    Ok(Arc::new(provider))
}

fn parse_max_tokens(lookup: EnvLookup<'_>) -> Result<u32, PipelineError> {
    match lookup("LLM_MAX_TOKENS") {
        None => Ok(DEFAULT_LLM_MAX_TOKENS),
        Some(raw) => raw.parse::<u32>().map_err(|e| {
            PipelineError::LlmProvider(format!(
                "LLM_MAX_TOKENS must be a non-negative integer: got '{raw}': {e}"
            ))
        }),
    }
}

// =========================================================================
// Embedding provider factory
// =========================================================================

/// Construct an [`EmbeddingProvider`] trait object from process environment.
///
/// Reads `std::env` directly. Tests and alternative config sources should use
/// [`embedding_provider_from_lookup`] instead.
///
/// # Errors
///
/// Returns `PipelineError::LlmProvider` with a descriptive message pointing
/// at the relevant env var on any missing or invalid configuration.
pub fn embedding_provider_from_env() -> Result<Arc<dyn EmbeddingProvider>, PipelineError> {
    embedding_provider_from_lookup(&std_env_lookup)
}

/// Testable core of [`embedding_provider_from_env`].
///
/// Recognized `EMBEDDING_PROVIDER` values: `"fastembed"` (default), `"vllm"`.
pub fn embedding_provider_from_lookup(
    lookup: EnvLookup<'_>,
) -> Result<Arc<dyn EmbeddingProvider>, PipelineError> {
    let provider =
        lookup("EMBEDDING_PROVIDER").unwrap_or_else(|| DEFAULT_EMBEDDING_PROVIDER.to_string());

    match provider.as_str() {
        "fastembed" => build_fastembed(lookup),
        "vllm" => build_vllm_embedding(lookup),
        other => Err(PipelineError::LlmProvider(format!(
            "Unknown EMBEDDING_PROVIDER: '{other}'. Valid values: fastembed, vllm"
        ))),
    }
}

fn build_fastembed(lookup: EnvLookup<'_>) -> Result<Arc<dyn EmbeddingProvider>, PipelineError> {
    let mode = lookup("FASTEMBED_MODE").unwrap_or_else(|| DEFAULT_FASTEMBED_MODE.to_string());

    match mode.as_str() {
        "huggingface" => build_fastembed_huggingface(lookup),
        "local" => build_fastembed_local(lookup),
        other => Err(PipelineError::LlmProvider(format!(
            "Unknown FASTEMBED_MODE: '{other}'. Valid values: huggingface, local"
        ))),
    }
}

fn build_fastembed_huggingface(
    lookup: EnvLookup<'_>,
) -> Result<Arc<dyn EmbeddingProvider>, PipelineError> {
    let model_name = lookup("FASTEMBED_MODEL").ok_or_else(|| {
        PipelineError::LlmProvider(
            "FASTEMBED_MODEL is required when EMBEDDING_PROVIDER=fastembed".to_string(),
        )
    })?;

    let model = parse_fastembed_model(&model_name)?;
    let cache_dir = lookup("FASTEMBED_CACHE_DIR").map(PathBuf::from);

    let provider = FastembedProvider::new_from_huggingface(model, cache_dir)?;
    Ok(Arc::new(provider))
}

fn build_fastembed_local(
    lookup: EnvLookup<'_>,
) -> Result<Arc<dyn EmbeddingProvider>, PipelineError> {
    let model_name = lookup("FASTEMBED_MODEL").ok_or_else(|| {
        PipelineError::LlmProvider(
            "FASTEMBED_MODEL is required when EMBEDDING_PROVIDER=fastembed".to_string(),
        )
    })?;

    // Parse the whitelist even in local mode to ensure the operator has
    // selected a supported model identifier. The actual model bytes come from
    // disk, but the name must match the whitelist for downstream logging and
    // dimensional consistency.
    let _ = parse_fastembed_model(&model_name)?;

    let onnx = require_local_file(lookup, "FASTEMBED_LOCAL_ONNX_PATH")?;
    let tokenizer = require_local_file(lookup, "FASTEMBED_LOCAL_TOKENIZER_PATH")?;
    let config = require_local_file(lookup, "FASTEMBED_LOCAL_CONFIG_PATH")?;
    let special_tokens_map = require_local_file(lookup, "FASTEMBED_LOCAL_SPECIAL_TOKENS_PATH")?;
    let tokenizer_config = require_local_file(lookup, "FASTEMBED_LOCAL_TOKENIZER_CONFIG_PATH")?;

    let dimensions = require_u32(lookup, "FASTEMBED_LOCAL_DIMENSIONS")?;

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: tokenizer,
        config_file: config,
        special_tokens_map_file: special_tokens_map,
        tokenizer_config_file: tokenizer_config,
    };

    // UserDefinedEmbeddingModel is #[non_exhaustive] with 4 fields
    // (onnx_file, tokenizer_files, pooling, quantization). Use the crate's
    // provided constructor, which defaults pooling=None and quantization=None —
    // matching what FastembedProvider::new_from_local expects via
    // InitOptionsUserDefined::default().
    let model_files = UserDefinedEmbeddingModel::new(onnx, tokenizer_files);

    let provider = FastembedProvider::new_from_local(model_files, model_name, dimensions)?;
    Ok(Arc::new(provider))
}

fn build_vllm_embedding(
    lookup: EnvLookup<'_>,
) -> Result<Arc<dyn EmbeddingProvider>, PipelineError> {
    let base_url = lookup("VLLM_BASE_URL").ok_or_else(|| {
        PipelineError::LlmProvider(
            "VLLM_BASE_URL is required when EMBEDDING_PROVIDER=vllm".to_string(),
        )
    })?;

    let model = lookup("EMBEDDING_MODEL").ok_or_else(|| {
        PipelineError::LlmProvider(
            "EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=vllm".to_string(),
        )
    })?;

    let dimensions = require_u32(lookup, "EMBEDDING_DIMENSIONS")?;
    let api_key = lookup("VLLM_API_KEY");

    let provider = VllmEmbeddingProvider::new(base_url, model, api_key, dimensions, None)?;
    Ok(Arc::new(provider))
}

fn require_local_file(lookup: EnvLookup<'_>, var: &str) -> Result<Vec<u8>, PipelineError> {
    let path = lookup(var).ok_or_else(|| {
        PipelineError::LlmProvider(format!("{var} is required when FASTEMBED_MODE=local"))
    })?;
    std::fs::read(&path)
        .map_err(|e| PipelineError::LlmProvider(format!("Failed to read {var} from '{path}': {e}")))
}

fn require_u32(lookup: EnvLookup<'_>, var: &str) -> Result<u32, PipelineError> {
    let raw =
        lookup(var).ok_or_else(|| PipelineError::LlmProvider(format!("{var} is required")))?;
    raw.parse::<u32>().map_err(|e| {
        PipelineError::LlmProvider(format!(
            "{var} must be a non-negative integer: got '{raw}': {e}"
        ))
    })
}

/// Parse a `FASTEMBED_MODEL` env string into a fastembed [`EmbeddingModel`]
/// variant.
///
/// The accepted set is deliberately a curated subset of fastembed's 44
/// variants. See module-level documentation for the curation rationale.
fn parse_fastembed_model(name: &str) -> Result<EmbeddingModel, PipelineError> {
    // Curated whitelist: text-embedding variants suitable for Colossus.
    // Adding a variant requires: (1) a match arm here, (2) confirmation that
    // its dimension matches your Qdrant collection, (3) a unit test.
    match name {
        "NomicEmbedTextV15" => Ok(EmbeddingModel::NomicEmbedTextV15),
        "NomicEmbedTextV15Q" => Ok(EmbeddingModel::NomicEmbedTextV15Q),
        "AllMiniLML6V2" => Ok(EmbeddingModel::AllMiniLML6V2),
        "AllMiniLML6V2Q" => Ok(EmbeddingModel::AllMiniLML6V2Q),
        "BGESmallENV15" => Ok(EmbeddingModel::BGESmallENV15),
        "BGESmallENV15Q" => Ok(EmbeddingModel::BGESmallENV15Q),
        "BGEBaseENV15" => Ok(EmbeddingModel::BGEBaseENV15),
        "BGELargeENV15" => Ok(EmbeddingModel::BGELargeENV15),
        "MultilingualE5Small" => Ok(EmbeddingModel::MultilingualE5Small),
        "MultilingualE5Base" => Ok(EmbeddingModel::MultilingualE5Base),
        "GTEBaseENV15" => Ok(EmbeddingModel::GTEBaseENV15),
        other => Err(PipelineError::LlmProvider(format!(
            "FASTEMBED_MODEL '{other}' is not in the curated whitelist. Supported: \
             NomicEmbedTextV15, NomicEmbedTextV15Q, AllMiniLML6V2, AllMiniLML6V2Q, \
             BGESmallENV15, BGESmallENV15Q, BGEBaseENV15, BGELargeENV15, \
             MultilingualE5Small, MultilingualE5Base, GTEBaseENV15. To add a \
             variant, edit parse_fastembed_model in providers/factory.rs and verify \
             the model dimension matches your Qdrant collection."
        ))),
    }
}

// =========================================================================
// Unit tests — pure-function paths only
// =========================================================================
//
// Tests that exercise provider construction end-to-end (HTTP client builds,
// HuggingFace downloads, file I/O) live in tests/factory_tests.rs as
// integration tests. This module contains only pure-function tests that need
// no external resources.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fastembed_model_accepts_all_whitelisted_variants() {
        // If this test fails, the whitelist and parse fn are out of sync.
        let cases = [
            ("NomicEmbedTextV15", EmbeddingModel::NomicEmbedTextV15),
            ("NomicEmbedTextV15Q", EmbeddingModel::NomicEmbedTextV15Q),
            ("AllMiniLML6V2", EmbeddingModel::AllMiniLML6V2),
            ("AllMiniLML6V2Q", EmbeddingModel::AllMiniLML6V2Q),
            ("BGESmallENV15", EmbeddingModel::BGESmallENV15),
            ("BGESmallENV15Q", EmbeddingModel::BGESmallENV15Q),
            ("BGEBaseENV15", EmbeddingModel::BGEBaseENV15),
            ("BGELargeENV15", EmbeddingModel::BGELargeENV15),
            ("MultilingualE5Small", EmbeddingModel::MultilingualE5Small),
            ("MultilingualE5Base", EmbeddingModel::MultilingualE5Base),
            ("GTEBaseENV15", EmbeddingModel::GTEBaseENV15),
        ];
        for (input, expected) in cases {
            let result = parse_fastembed_model(input).expect("whitelist should parse");
            assert_eq!(result, expected, "input {input:?} produced wrong variant");
        }
    }

    #[test]
    fn parse_fastembed_model_rejects_unknown() {
        let err = parse_fastembed_model("NotAModel").unwrap_err();
        match err {
            PipelineError::LlmProvider(msg) => {
                assert!(msg.contains("NotAModel"));
                assert!(msg.contains("curated whitelist"));
                assert!(msg.contains("parse_fastembed_model"));
            }
            other => panic!("expected LlmProvider variant, got {other:?}"),
        }
    }

    #[test]
    fn parse_llm_temperature_cases() {
        // (env_value, expected) — None means LLM_TEMPERATURE unset.
        // Invalid values fall back to None (do not block startup) with a
        // warning log; the API default takes over.
        let cases: &[(Option<&str>, Option<f64>)] = &[
            (None, None),                 // unset
            (Some("0"), Some(0.0)),       // explicit zero (deterministic extraction)
            (Some("0.7"), Some(0.7)),     // explicit float
            (Some("not-a-number"), None), // unparseable → fallback
        ];

        for (env_value, expected) in cases {
            let lookup = |key: &str| -> Option<String> {
                if key == "LLM_TEMPERATURE" {
                    env_value.map(|s| s.to_string())
                } else {
                    None
                }
            };
            assert_eq!(
                parse_llm_temperature(&lookup),
                *expected,
                "case: LLM_TEMPERATURE={env_value:?}",
            );
        }
    }

    #[test]
    fn parse_fastembed_model_rejects_nontext_variants() {
        // Sanity check: ClipVitB32 is a real fastembed variant but an image
        // encoder. It was deliberately excluded. This test locks in that
        // exclusion so a future "just add everything" edit fails here first.
        let err = parse_fastembed_model("ClipVitB32").unwrap_err();
        match err {
            PipelineError::LlmProvider(msg) => assert!(msg.contains("ClipVitB32")),
            other => panic!("expected LlmProvider variant, got {other:?}"),
        }
    }
}
