//! All AnthropicProvider trait-passthrough tests deleted in the test audit
//! (Batch 8): object-safety is compile-only (compiler enforces wherever
//! `Arc<dyn LlmProvider>` is used in AppContext); model_name/max_tokens_default
//! getters return what the constructor set (rule 7); cost_methods/
//! supports_structured_output/provider_name return literal constants matched
//! at compile time by Rust consumers (rule 8); constructor happy-path Ok
//! check is rule 2. Body-shape tests live in `src/providers/anthropic.rs`
//! tests; live API behavior is exercised by `tests/anthropic_live.rs`.
