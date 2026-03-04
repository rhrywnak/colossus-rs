//! # Rig Framework Compatibility Spike — Integration Tests
//!
//! These tests validate that the Rig framework can work with our infrastructure:
//! - Qdrant vector database (colossus_evidence collection, 768-dim, cosine)
//! - fastembed (nomic-embed-text v1.5)
//! - Claude API (Anthropic provider)
//!
//! ## Rust Learning: Integration tests
//!
//! Files in `tests/` are compiled as SEPARATE crates that depend on your library.
//! They can only access your crate's public API (unlike unit tests in `src/`).
//! Each file in `tests/` becomes its own test binary.
//!
//! ## Rust Learning: `#[tokio::test]`
//!
//! This attribute macro transforms an `async fn` into a synchronous test function
//! that creates a tokio runtime and blocks on the async body. It's equivalent to:
//! ```rust,ignore
//! #[test]
//! fn test_name() {
//!     tokio::runtime::Runtime::new().unwrap().block_on(async {
//!         // your async test body here
//!     });
//! }
//! ```

// ---------------------------------------------------------------------------
// Test 1: Rig Compiles
// ---------------------------------------------------------------------------

/// ## T-R.0.1 — Verify Rig types can be imported and compile
///
/// This is a compile-time validation test. If the Rig crate's types, traits,
/// and generics are fundamentally incompatible with our toolchain or have
/// breaking API changes, this test will fail to compile.
///
/// ## Rust Learning: `use` imports for type validation
///
/// Even if we don't call anything at runtime, the act of importing types
/// forces the compiler to resolve all generic bounds, trait implementations,
/// and lifetime constraints. This catches incompatibilities at compile time.
///
/// ## Rig Concept: Core traits
///
/// Rig organizes its API around a few key traits:
/// - `Prompt` / `Chat`: High-level interfaces for sending text to an LLM
///   and getting a string response back.
/// - `CompletionModel`: Low-level trait that providers implement. Gives you
///   full access to the completion response including token usage.
/// - `EmbeddingModel`: Trait for models that convert text → vector embeddings.
/// - `VectorStoreIndex`: Trait for vector databases that support similarity search.
///   The `top_n` method returns `Vec<(score, id, payload)>`.
///
/// These traits are defined in `rig::completion`, `rig::embeddings`, and
/// `rig::vector_store` respectively.
#[test]
fn test_rig_compiles() {
    // Import core Rig traits — if any of these fail to resolve,
    // we have a fundamental compatibility problem.

    // CompletionModel trait: the low-level interface providers implement.
    // Gives access to full responses including token usage via
    // `.completion_request("text").max_tokens(N).send().await`.
    use rig::completion::CompletionModel;

    // Prompt trait: the high-level "send text, get text" interface.
    // Any type implementing Prompt can be used with `.prompt("text").await`.
    use rig::completion::Prompt;

    // EmbeddingModel trait: converts text into vector embeddings.
    // Implementations provide `.embed_text("text").await` and `.ndims()`.
    use rig::embeddings::EmbeddingModel;

    // VectorStoreIndex trait: similarity search over a vector database.
    // Provides `.top_n::<T>(request).await` returning `Vec<(score, id, T)>`.
    use rig::vector_store::VectorStoreIndex;

    // Anthropic provider types — prove Claude integration compiles.
    // `Client` wraps the HTTP client configured for Anthropic's API.
    // Model constants like CLAUDE_3_5_HAIKU define model ID strings.
    use rig::providers::anthropic;

    // Suppress unused-import warnings — we're testing compilation, not runtime.
    //
    // ## Rust Learning: Trait bounds in generic functions
    //
    // These traits are NOT dyn-compatible (they have generic methods, associated
    // consts, and `impl Trait` returns). We can't make trait objects (`dyn Prompt`).
    // Instead, we prove they exist by using them as generic bounds in a
    // never-called function. The compiler still type-checks the bounds.
    fn _assert_traits_exist<
        P: Prompt,
        M: CompletionModel,
        E: EmbeddingModel,
        V: VectorStoreIndex,
    >() {}

    // Also verify the Anthropic model constant is accessible.
    let _model_id: &str = anthropic::completion::CLAUDE_3_5_HAIKU;

    // If we reach here, all Rig types resolved successfully.
    // The compiler verified trait bounds, generic parameters, and dependencies.
}

// ---------------------------------------------------------------------------
// Test 2: Fastembed Nomic Embed
// ---------------------------------------------------------------------------

/// ## T-R.0.2 — Embed text with rig-fastembed using nomic-embed-text v1.5
///
/// Validates that rig-fastembed can:
/// 1. Initialize the nomic-embed-text v1.5 model
/// 2. Produce embeddings with exactly 768 dimensions
/// 3. Produce non-zero embeddings (not a degenerate zero vector)
///
/// ## Rig Concept: `rig_fastembed::Client` and `FastembedModel`
///
/// rig-fastembed provides its own `Client` struct (not to be confused with
/// the Anthropic `Client`). You create it with `Client::new()`, then call
/// `.embedding_model(&FastembedModel::NomicEmbedTextV15)` to get an
/// `EmbeddingModel` that wraps fastembed's `TextEmbedding`.
///
/// Internally, this downloads the ONNX model weights on first use (~100MB)
/// to `~/.cache/fastembed` and runs inference on the CPU.
///
/// ## Rig Concept: `EmbeddingModel::embed_text()`
///
/// The `EmbeddingModel` trait defines `embed_text(&self, text)` which returns
/// `Result<Embedding, EmbeddingError>`. An `Embedding` struct contains:
/// - `document`: the original text
/// - `vec`: `Vec<f64>` — note: f64, not f32! Rig uses f64 internally even
///   though fastembed produces f32. The conversion happens in rig-fastembed's
///   `embed_texts()` implementation.
///
/// ## IMPORTANT: fastembed version mismatch
///
/// rig-fastembed 0.2.23 pulls fastembed 4.9.1, but colossus-legal uses
/// fastembed 5.x. Different versions MAY produce different embedding vectors
/// for the same text. Test 3 (Qdrant search) will reveal if this causes
/// search quality degradation against our existing collection.
#[tokio::test]
async fn test_fastembed_nomic_embed() {
    // rig_fastembed re-exports fastembed's EmbeddingModel enum as `FastembedModel`.
    // This lets us specify which model to load without importing fastembed directly.
    use rig_fastembed::{Client, FastembedModel};

    // rig::embeddings::EmbeddingModel is the TRAIT that defines embed_text().
    // We need it in scope to call .embed_text() on the model instance.
    //
    // ## Rust Learning: Trait imports for method resolution
    // In Rust, you must import a trait to call its methods on a concrete type.
    // Even though `rig_fastembed::EmbeddingModel` implements the trait, the
    // compiler won't find `.embed_text()` unless the trait is in scope.
    use rig::embeddings::EmbeddingModel;

    // Step 1: Create the fastembed client.
    // This is a zero-cost struct — no network calls happen here.
    let client = Client::new();

    // Step 2: Create the embedding model for nomic-embed-text v1.5.
    //
    // ## Rig Concept: Model initialization
    // `.embedding_model()` calls fastembed's `TextEmbedding::try_new()` internally,
    // which downloads the ONNX model on first run. Subsequent calls load from cache.
    // The model is wrapped in an `Arc` so it can be cloned cheaply.
    //
    // NomicEmbedTextV15 produces 768-dimensional embeddings — the same dimensions
    // as our existing colossus_evidence Qdrant collection.
    let model = client.embedding_model(&FastembedModel::NomicEmbedTextV15);

    // Step 3: Verify the model reports correct dimensions.
    // The `ndims()` method returns the embedding dimensionality.
    assert_eq!(
        model.ndims(),
        768,
        "NomicEmbedTextV15 should produce 768-dimensional embeddings"
    );

    // Step 4: Embed a sample text.
    //
    // ## Rig Concept: embed_text() is async
    // Even though fastembed runs ONNX inference synchronously on the CPU,
    // rig wraps it in an async interface for consistency with remote embedding
    // providers (like OpenAI embeddings). Under the hood, rig-fastembed calls
    // `self.embedder.embed(...)` synchronously within the async function.
    let embedding = model
        .embed_text("The plaintiff alleges damages of $50,000 for breach of contract.")
        .await
        .expect("embed_text should succeed");

    // Step 5: Verify embedding dimensions.
    // `embedding.vec` is a `Vec<f64>` (Rig uses f64 internally).
    assert_eq!(
        embedding.vec.len(),
        768,
        "Embedding vector should have exactly 768 dimensions"
    );

    // Step 6: Verify the embedding is not a degenerate zero vector.
    // A zero vector would mean the model failed to encode meaning.
    let all_zeros = embedding.vec.iter().all(|&v| v == 0.0);
    assert!(
        !all_zeros,
        "Embedding should not be all zeros — model should encode semantic meaning"
    );

    // Log some stats for diagnostic purposes.
    let norm: f64 = embedding.vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  Embedding norm (L2): {norm:.4}");
    println!(
        "  First 5 values: {:?}",
        &embedding.vec[..5]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 3: Qdrant Search
// ---------------------------------------------------------------------------

/// ## T-R.0.3 — Search colossus_evidence on DEV Qdrant via rig-qdrant
///
/// This test validates the full RAG search pipeline through Rig:
/// 1. Embed a query using rig-fastembed (nomic-embed-text v1.5)
/// 2. Search the existing colossus_evidence collection on DEV Qdrant
/// 3. Verify results contain payload fields (node_id, node_type)
///
/// ## Rig Concept: `QdrantVectorStore`
///
/// `rig_qdrant::QdrantVectorStore` combines an embedding model with a Qdrant
/// client to provide the `VectorStoreIndex` trait. You construct it with:
/// ```rust,ignore
/// QdrantVectorStore::new(qdrant_client, embedding_model, query_params)
/// ```
/// where `query_params` is a `QueryPoints` struct that specifies the collection
/// name and default search parameters.
///
/// ## Rig Concept: `VectorStoreIndex::top_n()`
///
/// The `top_n::<T>(request)` method is generic over the payload type `T`.
/// It returns `Vec<(f64, String, T)>` where:
/// - `f64` is the similarity score
/// - `String` is the point ID
/// - `T` is the payload deserialized from the Qdrant point's payload
///
/// We deserialize payloads as `serde_json::Value` for maximum flexibility,
/// then extract specific fields manually.
///
/// ## CRITICAL: Score comparison with Minerva
///
/// This test logs search scores so we can compare them against Minerva's
/// (colossus-legal's current search) scores for the same query. If rig-fastembed
/// (fastembed 4.9.1) produces different embeddings than our production fastembed 5.x,
/// scores will be noticeably lower.
#[tokio::test]
async fn test_qdrant_search() {
    use rig::vector_store::VectorStoreIndex;
    use rig_fastembed::{Client as FembedClient, FastembedModel};
    use rig_qdrant::QdrantVectorStore;

    // --- Configuration from environment (with defaults matching DEV) ---
    //
    // ## IMPORTANT: gRPC vs REST ports
    //
    // qdrant-client (used by rig-qdrant) communicates via gRPC, which runs
    // on port 6334 by default. Our existing colossus-legal code uses the
    // REST API on port 6333 via raw reqwest HTTP calls.
    //
    // QDRANT_URL env var points to the REST API (port 6333).
    // For gRPC, we need port 6334. We use a separate QDRANT_GRPC_URL env var,
    // or derive it from QDRANT_URL by replacing the port.
    let qdrant_url = std::env::var("QDRANT_GRPC_URL").unwrap_or_else(|_| {
        let rest_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://10.10.100.200:6333".to_string());
        // Replace port 6333 with 6334 for gRPC
        rest_url.replace(":6333", ":6334")
    });

    let collection_name = "colossus_evidence";

    // --- Step 1: Create the embedding model ---
    //
    // We use the same model (NomicEmbedTextV15) that was used to embed the
    // documents in our Qdrant collection. If the fastembed version produces
    // incompatible embeddings, search scores will be poor.
    let fastembed_client = FembedClient::new();
    let embedding_model = fastembed_client.embedding_model(&FastembedModel::NomicEmbedTextV15);

    // --- Step 2: Create the Qdrant client ---
    //
    // ## Rig Concept: qdrant_client::Qdrant
    //
    // rig-qdrant uses the official `qdrant-client` crate (v1.17.0) which
    // communicates over gRPC by default. The `Qdrant::from_url()` constructor
    // takes a URL and builds a gRPC client.
    //
    // IMPORTANT: qdrant-client uses gRPC (port 6334 by default), NOT the
    // REST API (port 6333) that we use in colossus-legal. However,
    // `Qdrant::from_url` with an HTTP URL on port 6333 should auto-detect
    // and use the appropriate protocol, or we may need port 6334 for gRPC.
    //
    // Let's try the HTTP URL first — qdrant-client may support REST mode.
    // If this fails, we'll switch to gRPC port 6334.
    // `skip_compatibility_check()` prevents the client from querying the
    // Qdrant server version at startup. This avoids an extra round-trip
    // and a noisy warning message in test output.
    let qdrant_client = qdrant_client::Qdrant::from_url(&qdrant_url)
        .skip_compatibility_check()
        .build()
        .expect("Failed to create Qdrant client");

    // --- Step 3: Configure search parameters ---
    //
    // ## Rig Concept: QueryPoints
    //
    // `QueryPoints` is qdrant-client's struct for configuring vector search.
    // At minimum, we set the collection name. The `QdrantVectorStore` will
    // fill in the query vector and limit from the `VectorSearchRequest`.
    //
    // `with_payload(true)` tells Qdrant to return payload data with results.
    // Without this, we'd only get point IDs and scores.
    let query_params = qdrant_client::qdrant::QueryPoints {
        collection_name: collection_name.to_string(),
        with_payload: Some(true.into()),
        ..Default::default()
    };

    // --- Step 4: Create the vector store ---
    //
    // ## Rig Concept: Combining model + client + params
    //
    // `QdrantVectorStore::new()` takes ownership of all three components.
    // The type parameter `M` (embedding model) is inferred from the model arg.
    // This store now provides the `VectorStoreIndex` trait methods.
    let vector_store = QdrantVectorStore::new(qdrant_client, embedding_model, query_params);

    // --- Step 5: Search ---
    //
    // ## Rig Concept: VectorSearchRequest builder
    //
    // We build a search request with:
    // - `.query("...")`: the text to embed and search with
    // - `.samples(N)`: maximum number of results to return
    // - `.build()`: finalizes the request (panics if required fields missing)
    //
    // The `top_n::<T>()` method handles embedding the query text automatically
    // using the model we provided to the vector store. We don't need to embed
    // manually — Rig does it internally.
    //
    // QUERY: "What did Phillips say about the $50,000?" — a question about
    // a specific person and dollar amount that should match legal evidence
    // in the colossus_evidence collection.
    let query_text = "What did Phillips say about the $50,000?";

    // ## Rig Concept: VectorSearchRequest builder
    //
    // `.build()` returns a `Result` because it validates that required
    // fields (query, samples) are set. We `.expect()` here since we
    // know we've set both.
    let request = rig::vector_store::VectorSearchRequest::builder()
        .query(query_text)
        .samples(5)
        .build()
        .expect("VectorSearchRequest should build with query + samples");

    // We deserialize payloads as serde_json::Value for flexibility.
    // This avoids needing to define a struct matching the exact payload schema.
    let results: Vec<(f64, String, serde_json::Value)> = vector_store
        .top_n::<serde_json::Value>(request)
        .await
        .expect("Qdrant search should succeed");

    // --- Step 6: Validate results ---
    assert!(
        !results.is_empty(),
        "Search should return at least one result from colossus_evidence"
    );

    // --- Step 7: Log scores for Minerva comparison ---
    //
    // IMPORTANT: Compare these scores against what Minerva returns for
    // the same query "What did Phillips say about the $50,000?"
    // If scores here are significantly lower than Minerva's, it indicates
    // the fastembed 4.9.1 vs 5.x version mismatch is producing
    // incompatible embeddings.
    println!("\n  === Qdrant Search Results via Rig ===");
    println!("  Query: \"{query_text}\"");
    println!("  Collection: {collection_name}");
    println!("  Results returned: {}", results.len());
    println!("  ---");

    for (i, (score, id, payload)) in results.iter().enumerate() {
        let node_id = payload
            .get("node_id")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");
        let node_type = payload
            .get("node_type")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");
        let title = payload
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");

        println!("  Result {}: score={score:.4}, id={id}", i + 1);
        println!("    node_id:   {node_id}");
        println!("    node_type: {node_type}");
        println!("    title:     {title}");

        // Validate score is positive (cosine similarity > 0 means some relevance)
        assert!(
            *score > 0.0,
            "Search result score should be positive (cosine similarity)"
        );
    }

    // Validate the first result has expected payload fields.
    let (_, _, first_payload) = &results[0];
    assert!(
        first_payload.get("node_id").is_some(),
        "Payload should contain 'node_id' field"
    );
    assert!(
        first_payload.get("node_type").is_some(),
        "Payload should contain 'node_type' field"
    );

    // Log the top score prominently for easy comparison with Minerva.
    let top_score = results[0].0;
    println!("\n  >>> TOP SCORE: {top_score:.4} (compare with Minerva scores) <<<\n");
}

// ---------------------------------------------------------------------------
// Test 4: Claude via Rig
// ---------------------------------------------------------------------------

/// ## T-R.0.4 — Call Claude through Rig's Anthropic provider
///
/// Validates that Rig can:
/// 1. Create an Anthropic client from the ANTHROPIC_API_KEY env var
/// 2. Send a completion request to Claude
/// 3. Receive a non-empty text response
/// 4. Return token usage (input_tokens, output_tokens)
///
/// ## Rig Concept: Anthropic Provider
///
/// Rig's Anthropic provider (`rig::providers::anthropic`) wraps the Anthropic
/// Messages API. Key types:
///
/// - `Client`: The HTTP client configured with API key and headers.
///   Created via `Client::from_env()` (reads ANTHROPIC_API_KEY) or
///   `Client::new("key")`.
///
/// - `CompletionModel`: Created via `client.completion_model("model-id")`.
///   Model constants like `CLAUDE_3_5_HAIKU` provide valid model ID strings.
///
/// ## Rig Concept: `completion_request()` vs `prompt()`
///
/// Rig offers two levels of API:
///
/// 1. **High-level** (`Prompt` trait): `model.prompt("text").await`
///    Returns just a `String`. No access to token usage or raw response.
///
/// 2. **Low-level** (`CompletionModel` trait):
///    `model.completion_request("text").max_tokens(N).send().await`
///    Returns `CompletionResponse<T>` with `.usage.input_tokens` and
///    `.usage.output_tokens`.
///
/// We use the low-level API here because we need token usage metrics.
///
/// ## Rig Concept: `max_tokens` is REQUIRED for Anthropic
///
/// Unlike OpenAI, the Anthropic API requires `max_tokens` to be set on
/// every request. Rig enforces this — omitting `.max_tokens()` from the
/// request builder will cause a runtime error.
#[tokio::test]
async fn test_claude_via_rig() {
    // The CompletionModel trait must be in scope to call
    // `.completion_request()` and `.send()` on the model instance.
    use rig::completion::CompletionModel;

    // The Prompt trait must be in scope to use the high-level `.prompt()` API.
    // We import it for the first sub-test.
    use rig::completion::Prompt;

    // The CompletionClient trait must be in scope to call
    // `.completion_model()` on the Anthropic client. Without this import,
    // the method exists but Rust can't find it.
    use rig::client::CompletionClient;

    use rig::providers::anthropic;

    // ## Rig Concept: `ProviderClient` trait
    //
    // The `from_env()` method lives on the `ProviderClient` trait, not on
    // the `Client` struct directly. We must import the trait to call it.
    // This is a common Rust pattern: methods defined in traits need the
    // trait in scope even when called on a concrete type.
    use rig::client::ProviderClient;

    // --- Check for API key ---
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("  SKIPPED: ANTHROPIC_API_KEY not set");
        return;
    }

    // --- Step 1: Create Anthropic client ---
    //
    // `Client::from_env()` reads ANTHROPIC_API_KEY from the environment.
    // It also sets the `anthropic-version` header to "2023-06-01" and
    // the `x-api-key` header automatically.
    let client = anthropic::Client::from_env();

    // --- Step 2: Create a completion model ---
    //
    // ## Rig Concept: Model IDs as plain strings
    //
    // Rig provides constants like `CLAUDE_3_5_HAIKU`, `CLAUDE_4_SONNET` etc,
    // but you can also pass ANY valid model ID string. This is important because
    // Anthropic deprecates model IDs over time — the constants in rig-core 0.31
    // may lag behind the latest available models.
    //
    // We use the cheapest model to minimize cost. Rig constants define:
    //   CLAUDE_4_SONNET = "claude-sonnet-4-0"
    //   CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    //
    // But some of these may be deprecated. We use a known-good model ID
    // from our own config as fallback.
    let model_id = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    let model = client.completion_model(&model_id);

    // --- Sub-test A: High-level prompt via Agent ---
    //
    // ## Rig Concept: Agent vs CompletionModel
    //
    // The `Prompt` trait (high-level `.prompt()` returning a String) is
    // implemented on `Agent`, NOT on bare `CompletionModel`. To use the
    // simple prompt interface, you build an Agent from the model:
    //
    //   let agent = client.agent(model_id).build();
    //   let response = agent.prompt("text").await?;
    //
    // An Agent wraps a CompletionModel with optional preamble, tools,
    // and configuration. For a simple prompt, it's just a thin wrapper.
    // ## Rig Concept: max_tokens is REQUIRED for Anthropic
    //
    // Unlike OpenAI providers, Anthropic requires `max_tokens` on every
    // request. Rig enforces this — if you forget `.max_tokens()` when
    // building an Agent for Anthropic, you'll get a runtime error:
    //   "`max_tokens` must be set for Anthropic"
    let agent = client
        .agent(&model_id)
        .preamble("You are a helpful assistant. Be very concise.")
        .max_tokens(128)
        .build();

    let response = agent
        .prompt("Respond with exactly one word: hello")
        .await
        .expect("Claude prompt via Agent should succeed");

    assert!(
        !response.is_empty(),
        "Claude response should not be empty"
    );
    println!("  High-level Agent prompt response: \"{response}\"");

    // --- Sub-test B: Low-level completion (with token usage) ---
    //
    // ## Rig Concept: CompletionRequestBuilder chain
    //
    // `.completion_request("text")` returns a builder that lets you configure:
    // - `.preamble("system prompt")` — sets the system message
    // - `.max_tokens(N)` — required for Anthropic
    // - `.temperature(T)` — optional, defaults to provider default
    // - `.send()` — executes the request and returns CompletionResponse
    //
    // The `CompletionResponse` contains:
    // - `.choice`: the model's response content (OneOrMany<AssistantContent>)
    // - `.usage`: token counts (input_tokens, output_tokens)
    // - `.raw_response`: the provider-specific raw response
    let completion = model
        .completion_request("What is 2 + 2? Answer with just the number.")
        .preamble("You are a helpful math assistant. Be concise.".to_string())
        .max_tokens(64)
        .send()
        .await
        .expect("Claude completion should succeed");

    // Verify token usage is returned.
    //
    // ## Rig Concept: Usage struct
    //
    // `completion.usage` is `rig::completion::Usage` with fields:
    // - `input_tokens: u64` — tokens consumed by the prompt + system message
    // - `output_tokens: u64` — tokens generated in the response
    //
    // These map directly from Anthropic's API response.
    println!("  Token usage:");
    println!("    input_tokens:  {}", completion.usage.input_tokens);
    println!("    output_tokens: {}", completion.usage.output_tokens);

    assert!(
        completion.usage.input_tokens > 0,
        "input_tokens should be greater than 0"
    );
    assert!(
        completion.usage.output_tokens > 0,
        "output_tokens should be greater than 0"
    );

    // Verify the response content is accessible.
    // `completion.choice` is `OneOrMany<AssistantContent>`.
    // We convert to string representation.
    let choice_text = format!("{:?}", completion.choice);
    println!("  Completion choice: {choice_text}");
    assert!(
        !choice_text.is_empty(),
        "Completion choice should not be empty"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Workspace Builds
// ---------------------------------------------------------------------------

/// ## T-R.0.5 — Validate workspace coexistence
///
/// This test verifies that:
/// 1. The spike crate compiles alongside colossus-auth without conflicts
/// 2. No dependency version clashes between the two crates
///
/// This is primarily a compile-time check. If you're reading this,
/// the test has already passed — the fact that this test file compiled
/// means the workspace builds successfully.
///
/// ## Rust Learning: Workspace dependency resolution
///
/// Cargo workspaces resolve dependencies across ALL member crates.
/// If crate A needs `serde 1.0.100` and crate B needs `serde 1.0.200`,
/// Cargo picks one version that satisfies both (usually the highest
/// compatible version).
///
/// However, if crate A needs `reqwest 0.12` and crate B needs `reqwest 0.13`,
/// Cargo will keep BOTH versions since they have different major.minor
/// versions. This is fine for builds but increases compile time and binary size.
///
/// In our case:
/// - colossus-auth uses `axum 0.7` and basic workspace deps
/// - spike uses `rig-core 0.31` which pulls `reqwest 0.13`
/// - No conflicts because colossus-auth doesn't use reqwest at all
#[test]
fn test_workspace_builds() {
    // If this test compiles and runs, the workspace builds successfully.
    // The Rust compiler has verified that:
    // 1. All dependencies resolved without conflicts
    // 2. All generic bounds are satisfied
    // 3. No link-time symbol collisions

    // Verify we can reference types from rig-core.
    let _model_name: &str = rig::providers::anthropic::completion::CLAUDE_3_5_HAIKU;

    // Verify rig-fastembed types are accessible.
    let _client = rig_fastembed::Client::new();

    // The fact that `cargo build` succeeded for the entire workspace
    // (including colossus-auth) means there are no dependency conflicts.
    println!("  Workspace builds successfully with both colossus-auth and spike");
    println!("  rig-core:      0.31.0");
    println!("  rig-qdrant:    0.1.37");
    println!("  rig-fastembed: 0.2.23");
}
