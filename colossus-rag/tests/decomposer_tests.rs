//! Unit tests for `LlmDecomposer` (P3-2 rewrite).
//!
//! These tests exercise `LlmDecomposer` through the `LlmProvider` abstraction
//! by injecting a tiny in-memory `TestLlmProvider` stub. No network, no API
//! keys, no feature flags.
//!
//! ## What changed from the old test story
//!
//! The prior constructor took `(api_key, model_id, ...)` and built an
//! Anthropic HTTP client internally, which made isolated unit tests
//! impossible. The new constructor takes `Arc<dyn LlmProvider>`, so tests
//! build a stub provider and verify:
//!
//! - Construction is infallible
//! - `decompose()` calls `invoke_with_system`, not plain `invoke`
//! - System prompt carries docs + persons; user prompt carries question +
//!   strategy (no leakage across the boundary)
//! - JSON parsing, markdown-fence stripping, and fallback behavior on
//!   provider error / parse failure
//! - Custom system templates override the default
//!
//! ## Stub duplication
//!
//! The `TestLlmProvider` stub below duplicates the one in
//! `synthesizer_tests.rs`. This duplication is deliberate — extracting a
//! `tests/common/` module is explicitly deferred. See the prior decision
//! about the 5-file `TestEmbeddingProvider` duplication across colossus-rs
//! tests.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use colossus_extract::{LlmProvider, LlmResponse, PipelineError};
use colossus_rag::{LlmDecomposer, QueryDecomposer, RetrievalStrategy, SubQuery};

// ===========================================================================
// TestLlmProvider — minimal stub implementing the LlmProvider trait
// ===========================================================================

/// A tiny in-memory provider for unit tests. Each call to `invoke()` or
/// `invoke_with_system()` returns a clone of the configured result and records
/// the call so tests can assert on interaction (which method was called, with
/// what arguments).
///
/// Interior mutability via `std::sync::Mutex` is required because the trait
/// methods take `&self`. Locks are held briefly (push + drop), so a blocking
/// mutex is correct here — no need for `tokio::sync::Mutex`.
struct TestLlmProvider {
    result: Result<LlmResponse, String>,
    invoke_calls: Mutex<Vec<String>>,
    invoke_with_system_calls: Mutex<Vec<(String, String)>>,
}

impl TestLlmProvider {
    fn new_ok(response: LlmResponse) -> Self {
        Self {
            result: Ok(response),
            invoke_calls: Mutex::new(Vec::new()),
            invoke_with_system_calls: Mutex::new(Vec::new()),
        }
    }

    fn new_err(msg: &str) -> Self {
        Self {
            result: Err(msg.to_string()),
            invoke_calls: Mutex::new(Vec::new()),
            invoke_with_system_calls: Mutex::new(Vec::new()),
        }
    }

    fn invoke_call_count(&self) -> usize {
        self.invoke_calls.lock().unwrap().len()
    }

    fn invoke_with_system_call_count(&self) -> usize {
        self.invoke_with_system_calls.lock().unwrap().len()
    }

    fn last_invoke_with_system_args(&self) -> Option<(String, String)> {
        self.invoke_with_system_calls
            .lock()
            .unwrap()
            .last()
            .cloned()
    }
}

#[async_trait]
impl LlmProvider for TestLlmProvider {
    async fn invoke(&self, prompt: &str, _max_tokens: u32) -> Result<LlmResponse, PipelineError> {
        self.invoke_calls.lock().unwrap().push(prompt.to_string());
        match &self.result {
            Ok(r) => Ok(r.clone()),
            Err(msg) => Err(PipelineError::LlmProvider(msg.clone())),
        }
    }

    // Explicit override of the default trait method. The default impl
    // concatenates system+prompt and delegates to `invoke()`, which would
    // hide whether `decompose()` is calling the preferred `invoke_with_system`
    // path or falling back to `invoke`. By recording calls here separately,
    // the test can distinguish the two paths.
    async fn invoke_with_system(
        &self,
        system: &str,
        prompt: &str,
        _max_tokens: u32,
    ) -> Result<LlmResponse, PipelineError> {
        self.invoke_with_system_calls
            .lock()
            .unwrap()
            .push((system.to_string(), prompt.to_string()));
        match &self.result {
            Ok(r) => Ok(r.clone()),
            Err(msg) => Err(PipelineError::LlmProvider(msg.clone())),
        }
    }

    fn model_name(&self) -> &str {
        "test-model"
    }

    fn provider_name(&self) -> &str {
        "test-provider"
    }

    fn cost_per_input_token(&self) -> Option<f64> {
        None
    }

    fn cost_per_output_token(&self) -> Option<f64> {
        None
    }

    fn supports_structured_output(&self) -> bool {
        false
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn test_document_aliases() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "phillips coa response".to_string(),
        "doc-phillips-coa-response-300891".to_string(),
    );
    m
}

fn test_person_names() -> Vec<String> {
    vec!["George Phillips".to_string(), "Marie Awad".to_string()]
}

fn make_decomposer(
    provider: Arc<TestLlmProvider>,
    system_template: Option<String>,
    user_template: Option<String>,
) -> LlmDecomposer {
    let provider_dyn: Arc<dyn LlmProvider> = provider;
    LlmDecomposer::new(
        provider_dyn,
        &test_document_aliases(),
        &test_person_names(),
        system_template,
        user_template,
    )
}

fn ok_provider(text: &str) -> Arc<TestLlmProvider> {
    Arc::new(TestLlmProvider::new_ok(LlmResponse {
        text: text.to_string(),
        input_tokens: None,
        output_tokens: None,
    }))
}

fn broad_strategy() -> RetrievalStrategy {
    RetrievalStrategy::Broad { node_types: None }
}

// ===========================================================================
// Unit tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: decompose() uses invoke_with_system, not invoke
// ---------------------------------------------------------------------------

/// Regression guard. If someone reverts `decompose()` to call `invoke()` with
/// a single concatenated prompt, this fails immediately with a clear reason.
#[tokio::test]
async fn test_decompose_uses_invoke_with_system_not_invoke() {
    let provider = ok_provider(
        r#"{"needs_decomposition": false, "sub_queries": [{"type": "vector_search", "query": "q"}]}"#,
    );
    let decomposer = make_decomposer(provider.clone(), None, None);

    let _ = decomposer
        .decompose("What is the test?", &broad_strategy())
        .await
        .expect("decompose should succeed");

    assert_eq!(
        provider.invoke_call_count(),
        0,
        "invoke() should not be called — decompose() must go through invoke_with_system()"
    );
    assert_eq!(
        provider.invoke_with_system_call_count(),
        1,
        "invoke_with_system() must be called exactly once per decompose()"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Docs and persons go in the system prompt
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_sends_docs_and_persons_in_system_prompt() {
    let provider = ok_provider(
        r#"{"needs_decomposition": false, "sub_queries": [{"type": "vector_search", "query": "q"}]}"#,
    );
    let decomposer = make_decomposer(provider.clone(), None, None);

    let _ = decomposer
        .decompose("any question", &broad_strategy())
        .await
        .expect("decompose should succeed");

    let (captured_system, _captured_user) = provider
        .last_invoke_with_system_args()
        .expect("invoke_with_system must have been called");

    assert!(
        captured_system.contains("\"phillips coa response\" -> doc-phillips-coa-response-300891"),
        "system prompt must contain the formatted document alias line; got:\n{captured_system}"
    );
    assert!(
        captured_system.contains("George Phillips"),
        "system prompt must contain known person 'George Phillips'; got:\n{captured_system}"
    );
    assert!(
        captured_system.contains("Marie Awad"),
        "system prompt must contain known person 'Marie Awad'; got:\n{captured_system}"
    );
    // And the persona/rules scaffolding from the default template:
    assert!(
        captured_system.contains("legal research query planner"),
        "system prompt must include the persona line from DEFAULT_SYSTEM_TEMPLATE"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Question and strategy go in the user prompt
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_sends_question_and_strategy_in_user_prompt() {
    let provider = ok_provider(
        r#"{"needs_decomposition": false, "sub_queries": [{"type": "vector_search", "query": "q"}]}"#,
    );
    let decomposer = make_decomposer(provider.clone(), None, None);

    let question = "What did Phillips state in his CoA response?";
    let strategy = broad_strategy();
    let _ = decomposer
        .decompose(question, &strategy)
        .await
        .expect("decompose should succeed");

    let (captured_system, captured_user) = provider
        .last_invoke_with_system_args()
        .expect("invoke_with_system must have been called");

    assert!(
        captured_user.contains(question),
        "user prompt must include the question verbatim; got:\n{captured_user}"
    );
    assert!(
        captured_user.contains(&format!("{strategy:?}")),
        "user prompt must include the strategy Debug output; got:\n{captured_user}"
    );
    // And the inverse: question/strategy must NOT leak into the system prompt.
    assert!(
        !captured_system.contains(question),
        "system prompt must NOT contain the question; got:\n{captured_system}"
    );
    assert!(
        !captured_user.contains("AVAILABLE DOCUMENTS"),
        "user prompt must NOT contain the system-level persona scaffolding; got:\n{captured_user}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Valid JSON response parses into DecompositionResult
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_parses_valid_json_response() {
    let json = r#"{
        "needs_decomposition": true,
        "sub_queries": [
            {"type": "vector_search", "query": "refined question"},
            {"type": "graph_document_content", "document_id": "doc-123", "description": "fetch doc"}
        ]
    }"#;
    let provider = ok_provider(json);
    let decomposer = make_decomposer(provider, None, None);

    let result = decomposer
        .decompose("orig question", &broad_strategy())
        .await
        .expect("decompose should succeed");

    assert!(result.needs_decomposition);
    assert_eq!(result.sub_queries.len(), 2);
    assert_eq!(result.original_question, "orig question");

    match &result.sub_queries[0] {
        SubQuery::VectorSearch { query } => assert_eq!(query, "refined question"),
        other => panic!("expected VectorSearch, got {other:?}"),
    }
    match &result.sub_queries[1] {
        SubQuery::GraphDocumentContent {
            document_id,
            description,
        } => {
            assert_eq!(document_id, "doc-123");
            assert_eq!(description, "fetch doc");
        }
        other => panic!("expected GraphDocumentContent, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 7: Provider error falls back to no-decomposition VectorSearch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_falls_back_on_provider_error() {
    let provider = Arc::new(TestLlmProvider::new_err("boom"));
    let decomposer = make_decomposer(provider, None, None);

    let result = decomposer
        .decompose("broken?", &broad_strategy())
        .await
        .expect("decompose must return Ok(fallback) on provider error");

    assert!(!result.needs_decomposition);
    assert_eq!(result.sub_queries.len(), 1);
    match &result.sub_queries[0] {
        SubQuery::VectorSearch { query } => assert_eq!(query, "broken?"),
        other => panic!("fallback must be VectorSearch of the original question, got {other:?}"),
    }
    assert_eq!(result.original_question, "broken?");
}

// ---------------------------------------------------------------------------
// Test 8: Invalid JSON falls back to no-decomposition VectorSearch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_falls_back_on_invalid_json() {
    let provider = ok_provider("this is not json at all");
    let decomposer = make_decomposer(provider, None, None);

    let result = decomposer
        .decompose("nonjson?", &broad_strategy())
        .await
        .expect("decompose must return Ok(fallback) on parse error");

    assert!(!result.needs_decomposition);
    assert_eq!(result.sub_queries.len(), 1);
    match &result.sub_queries[0] {
        SubQuery::VectorSearch { query } => assert_eq!(query, "nonjson?"),
        other => panic!("fallback must be VectorSearch of the original question, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 9: Markdown code fences are stripped before parsing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_strips_markdown_fences() {
    let fenced = "```json\n{\"needs_decomposition\": false, \"sub_queries\": [{\"type\": \"vector_search\", \"query\": \"inner\"}]}\n```";
    let provider = ok_provider(fenced);
    let decomposer = make_decomposer(provider, None, None);

    let result = decomposer
        .decompose("outer", &broad_strategy())
        .await
        .expect("decompose should parse through fenced JSON");

    assert!(!result.needs_decomposition);
    assert_eq!(result.sub_queries.len(), 1);
    match &result.sub_queries[0] {
        SubQuery::VectorSearch { query } => assert_eq!(query, "inner"),
        other => panic!("expected VectorSearch with query 'inner', got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 10: Custom system template overrides the default
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decompose_uses_custom_system_template_when_provided() {
    let custom = "CUSTOM_PERSONA_TAG docs={docs} persons={persons}".to_string();
    let provider = ok_provider(
        r#"{"needs_decomposition": false, "sub_queries": [{"type": "vector_search", "query": "q"}]}"#,
    );
    let decomposer = make_decomposer(provider.clone(), Some(custom), None);

    let _ = decomposer
        .decompose("anything", &broad_strategy())
        .await
        .expect("decompose should succeed");

    let (captured_system, _captured_user) = provider
        .last_invoke_with_system_args()
        .expect("invoke_with_system must have been called");

    assert!(
        captured_system.starts_with("CUSTOM_PERSONA_TAG"),
        "custom system template must be used verbatim (with placeholders replaced); got:\n{captured_system}"
    );
    assert!(
        captured_system.contains("doc-phillips-coa-response-300891"),
        "{{docs}} placeholder must be replaced with document list; got:\n{captured_system}"
    );
    assert!(
        captured_system.contains("George Phillips"),
        "{{persons}} placeholder must be replaced with person list; got:\n{captured_system}"
    );
    // And the default persona must NOT appear when a custom template is provided.
    assert!(
        !captured_system.contains("legal research query planner"),
        "default persona must not appear when a custom template is supplied; got:\n{captured_system}"
    );
}
