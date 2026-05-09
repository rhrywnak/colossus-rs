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

    /// The API has rate-limited this request.
    ///
    /// ## Why a typed variant instead of LlmProvider("429...")
    ///
    /// When a rate limit error is a string buried in LlmProvider, the calling
    /// code has two bad options: pattern-match on error message strings (fragile,
    /// breaks silently when error text changes) or treat all LlmProvider errors
    /// the same (forces the same retry policy for rate limits and auth failures).
    ///
    /// A typed variant lets the orchestrator match precisely:
    ///   PipelineError::RateLimited { retry_after_secs } => {
    ///       // wait exactly this long, then retry this chunk
    ///   }
    ///
    /// ## The retry-after header
    ///
    /// Anthropic includes a retry-after header in every 429 response containing
    /// the exact number of seconds to wait before retrying. This is not a guess
    /// or a heuristic — it is the authoritative answer from the API about when
    /// the rate limit bucket will have enough capacity to accept the next request.
    ///
    /// Token bucket algorithm: Anthropic replenishes capacity continuously, not
    /// at fixed minute boundaries. The retry-after value reflects how long until
    /// YOUR specific request will fit in the current bucket state. Waiting less
    /// causes the next attempt to fail immediately. Waiting more wastes time.
    /// Waiting exactly retry_after_secs is correct.
    ///
    /// ## What to do with this error
    ///
    /// The extractor returns this error immediately — it does NOT retry internally.
    /// The orchestrator (chunk_orchestration.rs) owns the retry loop and has access
    /// to the database pool to update the user-visible progress label during the wait.
    /// This separation of concerns is intentional: the extractor knows HOW to detect
    /// a rate limit, the orchestrator knows WHAT to tell the user and WHEN to retry.
    #[error("Rate limited by API — retry after {retry_after_secs}s")]
    RateLimited {
        /// Exact number of seconds to wait before retrying, from the retry-after
        /// response header. If the header was absent (rare), defaults to 60.
        retry_after_secs: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limited_error_display() {
        // Verify the Display impl (from thiserror) formats correctly.
        // This test documents the exact string format that log parsers and
        // error_suggestion() in process.rs pattern-match against.
        let err = PipelineError::RateLimited {
            retry_after_secs: 45,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("45"),
            "Display must include retry_after_secs value"
        );
        assert!(
            msg.contains("rate limit") || msg.contains("Rate limit"),
            "Display must mention rate limit for log searchability"
        );
    }
}
