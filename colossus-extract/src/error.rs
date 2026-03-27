//! Error types for the extraction pipeline.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Template error: {0}")]
    Template(String),

    #[error("LLM provider error: {0}")]
    LlmProvider(String),

    #[error("Extraction failed: {0}")]
    Extraction(String),

    #[error("Verification error: {0}")]
    Verification(String),

    #[error("Entity resolution error: {0}")]
    EntityResolution(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}
