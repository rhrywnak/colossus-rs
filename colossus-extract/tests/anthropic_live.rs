//! Live integration test for AnthropicProvider.
//!
//! This test actually calls api.anthropic.com and verifies the request
//! completes within a reasonable time. It exists to catch HTTP/2 hang
//! regressions — a unit test with a mock server cannot catch the hang
//! because mocks don't exercise real TLS/HTTP/2 negotiation.
//!
//! GATED on the ANTHROPIC_API_KEY env var. CI that doesn't set the key
//! will see the test as "ignored." Run locally with:
//!
//!     export ANTHROPIC_API_KEY=sk-ant-...
//!     cargo test --test anthropic_live -- --ignored --nocapture

use std::time::{Duration, Instant};

use colossus_extract::providers::anthropic::AnthropicProvider;
use colossus_extract::traits::LlmProvider;

/// Maximum acceptable time for a small live call.
const MAX_CALL_SECS: u64 = 30;

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and real network access"]
async fn live_anthropic_call_returns_within_30s() {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set for this test");

    let provider = AnthropicProvider::new(
        api_key,
        "claude-sonnet-4-6".to_string(),
        1024,
        None,
        None,
    )
    .expect("provider should build");

    let prompt = "Reply with exactly one word: OK";

    let start = Instant::now();
    let response = tokio::time::timeout(
        Duration::from_secs(MAX_CALL_SECS),
        provider.invoke(prompt, 100),
    )
    .await
    .expect("call timed out — HTTP/2 hang regression?");

    let elapsed = start.elapsed();
    let resp = response.expect("Anthropic call failed");

    assert!(
        !resp.text.is_empty(),
        "response text should not be empty"
    );
    assert!(
        elapsed < Duration::from_secs(MAX_CALL_SECS),
        "call took {}s — HTTP/2 hang regression?",
        elapsed.as_secs()
    );

    println!("Live call succeeded in {}ms. Response: {:?}", elapsed.as_millis(), resp.text);
}
