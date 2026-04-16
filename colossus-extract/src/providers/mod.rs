//! LLM and embedding provider implementations.

pub mod anthropic;
pub mod fastembed;
pub mod vllm;
pub mod vllm_embed;

pub use self::fastembed::FastembedProvider;
pub use anthropic::AnthropicProvider;
pub use vllm::VllmProvider;
pub use vllm_embed::VllmEmbeddingProvider;
