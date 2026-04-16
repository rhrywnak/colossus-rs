//! LLM and embedding provider implementations.

pub mod anthropic;
pub mod factory;
pub mod fastembed;
pub mod vllm;
pub mod vllm_embed;

pub use self::fastembed::FastembedProvider;
pub use anthropic::AnthropicProvider;
pub use factory::embedding_provider_from_env;
pub use factory::embedding_provider_from_lookup;
pub use factory::llm_provider_from_env;
pub use factory::llm_provider_from_lookup;
pub use factory::EnvLookup;
pub use vllm::VllmProvider;
pub use vllm_embed::VllmEmbeddingProvider;
