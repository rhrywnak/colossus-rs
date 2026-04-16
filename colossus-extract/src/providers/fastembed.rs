//! Local embedding provider via fastembed-rs (ONNX Runtime).
//!
//! ## Why local fastembed vs remote embedding API
//!
//! Embedding calls are high-volume (one per chunk, thousands per document).
//! Running them remotely adds latency and dollar cost. Local ONNX inference
//! is free (hardware amortization only) and fast.
//!
//! ## Why `tokio::task::spawn_blocking`
//!
//! ONNX inference is CPU-bound (typically 10-200ms per batch). Running it
//! directly in an async fn would block the Tokio executor and stall all other
//! async tasks (HTTP requests, DB queries, etc.) for the duration of the
//! embedding. `spawn_blocking` moves the work to Tokio's blocking thread pool,
//! keeping the executor responsive.
//!
//! ## Two constructors — `new_from_huggingface` and `new_from_local`
//!
//! `new_from_huggingface` loads models via fastembed's default HuggingFace Hub
//! path — flexible for development and future projects (colossus-ai).
//! `new_from_local` uses fastembed's `try_new_from_user_defined` for air-gapped
//! production deployments (colossus-legal). The P2-6 factory function chooses
//! based on env vars.
//!
//! ## Why `embed_batch` override
//!
//! fastembed uses rayon internally for parallel batch inference. The trait's
//! default `embed_batch` does serial single-text `embed()` calls, defeating
//! that parallelism. The override calls fastembed's batch API once.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel,
};
use tokio::task;

use crate::error::PipelineError;
use crate::traits::EmbeddingProvider;

/// Local embedding provider using fastembed-rs (ONNX Runtime).
///
/// ## Rust Learning: `Arc<TextEmbedding>` for `spawn_blocking`
///
/// The `EmbeddingProvider` trait has `async fn embed(&self, ...)`, but ONNX
/// inference is CPU-bound and must run on `tokio::task::spawn_blocking`.
/// `spawn_blocking` requires a `'static` closure, so we cannot borrow `&self`
/// into the closure. `Arc<TextEmbedding>` lets us clone a cheap reference-counted
/// pointer into the closure, satisfying the `'static` requirement.
///
/// fastembed's `TextEmbedding::embed()` takes `&self` (immutable borrow),
/// so no `Mutex` is needed — multiple concurrent `spawn_blocking` calls can
/// safely share the `Arc<TextEmbedding>` via `&*model`.
pub struct FastembedProvider {
    /// The fastembed model, shared via Arc for spawn_blocking closures.
    model: Arc<TextEmbedding>,
    /// Human-readable model name for logging (e.g., "AllMiniLML6V2").
    model_name: String,
    /// Embedding vector dimension, captured at construction time.
    dimensions: u32,
}

impl FastembedProvider {
    /// Create a provider that downloads/caches models from HuggingFace Hub.
    ///
    /// - `model`: fastembed `EmbeddingModel` enum variant (e.g.,
    ///   `EmbeddingModel::AllMiniLML6V2`).
    /// - `cache_dir`: Directory for cached model files. `None` uses fastembed's
    ///   default (system temp dir). **Warning:** temp dir is ephemeral in
    ///   containers — production should always pass `Some(path)` pointing at
    ///   persistent storage (e.g., `/opt/colossus/data/fastembed`).
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` if model download or ONNX
    /// initialization fails.
    pub fn new_from_huggingface(
        model: EmbeddingModel,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self, PipelineError> {
        let model_name = model.to_string();

        let dimensions = TextEmbedding::get_model_info(&model)
            .map(|info| info.dim)
            .map_err(|e| PipelineError::LlmProvider(format!("fastembed model info: {e:#}")))?;
        let dimensions = u32::try_from(dimensions).map_err(|e| {
            PipelineError::LlmProvider(format!(
                "model dimensions {dimensions} out of u32 range: {e}"
            ))
        })?;

        let mut init_options = InitOptions::new(model).with_show_download_progress(false);
        if let Some(path) = cache_dir {
            init_options = init_options.with_cache_dir(path);
        }

        let text_embedding = TextEmbedding::try_new(init_options)
            .map_err(|e| PipelineError::LlmProvider(format!("fastembed init: {e:#}")))?;

        Ok(Self {
            model: Arc::new(text_embedding),
            model_name,
            dimensions,
        })
    }

    /// Create a provider from local ONNX model files (air-gapped deployments).
    ///
    /// - `model_files`: fastembed `UserDefinedEmbeddingModel` with ONNX file bytes
    ///   and tokenizer files.
    /// - `model_name`: Human-readable identifier for logging (caller-supplied
    ///   because fastembed cannot introspect arbitrary ONNX files).
    /// - `dimensions`: Embedding vector dimension (caller-supplied for the same
    ///   reason — must match the Qdrant collection configuration).
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` if ONNX model loading fails.
    pub fn new_from_local(
        model_files: UserDefinedEmbeddingModel,
        model_name: String,
        dimensions: u32,
    ) -> Result<Self, PipelineError> {
        let options = InitOptionsUserDefined::default();
        let text_embedding = TextEmbedding::try_new_from_user_defined(model_files, options)
            .map_err(|e| PipelineError::LlmProvider(format!("fastembed local init: {e:#}")))?;

        Ok(Self {
            model: Arc::new(text_embedding),
            model_name,
            dimensions,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for FastembedProvider {
    /// Embed a single text string via local ONNX inference.
    ///
    /// ONNX inference runs on `tokio::task::spawn_blocking` to avoid blocking
    /// the async executor. `Arc<TextEmbedding>` is cloned into the closure to
    /// satisfy the `'static` lifetime requirement.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` wrapping fastembed/ONNX errors
    /// or spawn_blocking join errors (rare — thread panic).
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        let text = text.to_string();
        let model = Arc::clone(&self.model);
        let model_name = self.model_name.clone();

        task::spawn_blocking(move || {
            let batch = model.embed(vec![text.as_str()], None).map_err(|e| {
                PipelineError::LlmProvider(format!(
                    "fastembed embed failed for {model_name}: {e:#}"
                ))
            })?;
            batch.into_iter().next().ok_or_else(|| {
                PipelineError::LlmProvider(
                    "fastembed returned empty batch for single input".to_string(),
                )
            })
        })
        .await
        .map_err(|e| PipelineError::LlmProvider(format!("fastembed spawn_blocking failed: {e}")))?
    }

    /// Embed multiple texts in a single batch via local ONNX inference.
    ///
    /// Overrides the trait's default serial implementation to use fastembed's
    /// native batch API, which leverages rayon for parallel inference.
    /// `None` batch_size uses fastembed's default of 256.
    ///
    /// The `&[&str]` → `Vec<String>` → `Vec<&str>` conversion is necessary
    /// because `&[&str]` has a non-`'static` lifetime and cannot cross into
    /// `spawn_blocking`. The owned `Vec<String>` is moved into the closure,
    /// then re-borrowed as `Vec<&str>` for fastembed's API.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::LlmProvider` wrapping fastembed/ONNX errors.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, PipelineError> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model = Arc::clone(&self.model);
        let model_name = self.model_name.clone();

        task::spawn_blocking(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            model.embed(text_refs, None).map_err(|e| {
                PipelineError::LlmProvider(format!(
                    "fastembed batch embed failed for {model_name}: {e:#}"
                ))
            })
        })
        .await
        .map_err(|e| PipelineError::LlmProvider(format!("fastembed spawn_blocking failed: {e}")))?
    }

    /// Returns the embedding vector dimension, captured at construction time.
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Returns the model name for logging and cost tracking.
    fn model_name(&self) -> &str {
        &self.model_name
    }
}
