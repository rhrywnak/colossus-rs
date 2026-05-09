//! Tests for Neo4jExpander — unit tests and integration tests against DEV Neo4j.
//!
//! ## Test organization
//!
//! - **Unit tests** (test 1): No Neo4j needed. Test construction.
//! - **Integration tests** (tests 2–5): Require NEO4J_URI, NEO4J_USER,
//!   NEO4J_PASSWORD env vars. Marked with `#[ignore]` — run with:
//!   ```bash
//!   NEO4J_URI=bolt://10.10.100.200:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=<pwd> \
//!     cargo test -p colossus-rag --features neo4j --test expander_tests -- --ignored --nocapture
//!   ```
//!
//! ## Note on feature gates
//!
//! Neo4jExpander requires the `neo4j` feature. These tests need `--features neo4j`
//! (or `--features full`) to compile.

#[cfg(feature = "neo4j")]
mod neo4j_tests {
    use colossus_rag::{GraphExpander, Neo4jExpander, RetrievalStrategy, ScopeFilter, ScopeFilterType};
    use std::sync::Arc;

    // ===========================================================================
    // Unit Tests — no Neo4j needed
    // ===========================================================================

    // ===========================================================================
    // Integration Tests — require NEO4J_URI + credentials
    // ===========================================================================

    /// Helper: create a Neo4j connection from environment variables.
    ///
    /// Uses:
    /// - NEO4J_URI (default: bolt://10.10.100.200:7687)
    /// - NEO4J_USER (default: neo4j)
    /// - NEO4J_PASSWORD (required)
    async fn create_test_graph() -> Arc<neo4rs::Graph> {
        let uri = std::env::var("NEO4J_URI")
            .unwrap_or_else(|_| "bolt://10.10.100.200:7687".to_string());
        let user = std::env::var("NEO4J_USER")
            .unwrap_or_else(|_| "neo4j".to_string());
        let password = std::env::var("NEO4J_PASSWORD")
            .expect("NEO4J_PASSWORD must be set for integration tests");

        let config = neo4rs::ConfigBuilder::default()
            .uri(&uri)
            .user(&user)
            .password(&password)
            .build()
            .expect("Failed to build Neo4j config");

        let graph = neo4rs::Graph::connect(config)
            .await
            .expect("Failed to connect to Neo4j");

        Arc::new(graph)
    }

    // -----------------------------------------------------------------------
    // Test 2: Expand an Evidence node
    // -----------------------------------------------------------------------

    /// Expand "evidence-phillips-q74" — a well-connected Evidence node.
    ///
    /// Expected neighbors: Person (Phillips), Document (Phillips Deposition),
    /// and possibly ComplaintAllegation and rebuttal Evidence.
    #[tokio::test]
    #[ignore]
    async fn test_expand_evidence_node() {
        let graph = create_test_graph().await;
        let expander = Neo4jExpander::new(graph);

        let seeds = vec!["evidence-phillips-q74".to_string()];
        let chunks = expander.expand(&seeds, 1, &RetrievalStrategy::Broad { node_types: None }).await
            .expect("Expansion should succeed");

        // Should have at least the seed + some neighbors.
        assert!(
            !chunks.is_empty(),
            "Should return at least one chunk for a known Evidence node"
        );

        // Print results for verification.
        println!("\n  === Evidence Expansion: evidence-phillips-q74 ===");
        println!("  Total chunks: {}", chunks.len());
        for chunk in &chunks {
            println!(
                "  - {} ({}): {} [rels: {}]",
                chunk.node_id,
                chunk.node_type,
                chunk.title,
                chunk.relationships.len()
            );
            for rel in &chunk.relationships {
                let arrow = match rel.direction {
                    colossus_rag::RelationDirection::Outbound => "→",
                    colossus_rag::RelationDirection::Inbound => "←",
                };
                println!(
                    "      {} {} {} ({})",
                    arrow, rel.relationship, rel.node_id, rel.node_type
                );
            }
        }

        // The seed itself should appear in the results.
        assert!(
            chunks.iter().any(|c| c.node_id == "evidence-phillips-q74"),
            "Seed node should appear in results"
        );

        // Should find at least one neighbor node.
        assert!(
            chunks.len() > 1,
            "Should find at least one neighbor (Person, Document, etc.)"
        );

        // Graph-expanded nodes have score 0.0.
        for chunk in &chunks {
            assert_eq!(
                chunk.score, 0.0,
                "Graph-expanded chunks should have score 0.0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: Expand multiple seeds with deduplication
    // -----------------------------------------------------------------------

    /// Expand two Evidence nodes that likely share neighbors (e.g., same
    /// speaker or document). Verify deduplication — each unique node
    /// should appear at most once.
    #[tokio::test]
    #[ignore]
    async fn test_expand_multiple_seeds_deduplication() {
        let graph = create_test_graph().await;
        let expander = Neo4jExpander::new(graph);

        let seeds = vec![
            "evidence-phillips-q74".to_string(),
            "evidence-phillips-q91".to_string(),
        ];
        let chunks = expander.expand(&seeds, 1, &RetrievalStrategy::Broad { node_types: None }).await
            .expect("Expansion should succeed");

        println!("\n  === Multi-seed Expansion ===");
        println!("  Total chunks: {}", chunks.len());

        // Verify uniqueness — no duplicate node IDs.
        let mut seen_ids = std::collections::HashSet::new();
        for chunk in &chunks {
            assert!(
                seen_ids.insert(&chunk.node_id),
                "Duplicate node_id found: {} — deduplication failed",
                chunk.node_id
            );
            println!("  - {} ({})", chunk.node_id, chunk.node_type);
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Expand a non-existent node returns empty
    // -----------------------------------------------------------------------

    /// Expanding a node ID that doesn't exist in the graph should return
    /// an empty vec (not an error). The type resolution query simply
    /// returns no results for unknown IDs.
    #[tokio::test]
    #[ignore]
    async fn test_expand_nonexistent_id() {
        let graph = create_test_graph().await;
        let expander = Neo4jExpander::new(graph);

        let seeds = vec!["this-id-does-not-exist-12345".to_string()];
        let chunks = expander.expand(&seeds, 1, &RetrievalStrategy::Broad { node_types: None }).await
            .expect("Expansion should succeed (empty, not error)");

        assert!(
            chunks.is_empty(),
            "Non-existent ID should return empty vec, got {} chunks",
            chunks.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Expand a ComplaintAllegation node
    // -----------------------------------------------------------------------

    /// Expand an allegation node to verify the allegation-specific Cypher
    /// query works (different relationships: PROVES, RELIES_ON, SUPPORTS).
    #[tokio::test]
    #[ignore]
    async fn test_expand_allegation_node() {
        let graph = create_test_graph().await;
        let expander = Neo4jExpander::new(graph.clone());

        // First, find an allegation ID by querying Neo4j directly.
        let cypher = "MATCH (a:ComplaintAllegation) RETURN a.id AS id LIMIT 1";
        let mut result = graph.execute(neo4rs::query(cypher)).await
            .expect("Should query Neo4j");

        let allegation_id: String = if let Some(row) = result.next().await
            .expect("Should get result") {
            row.get("id").unwrap_or_default()
        } else {
            println!("  No ComplaintAllegation nodes found — skipping test");
            return;
        };

        if allegation_id.is_empty() {
            println!("  Empty allegation ID — skipping test");
            return;
        }

        let seeds = vec![allegation_id.clone()];
        let chunks = expander.expand(&seeds, 1, &RetrievalStrategy::Broad { node_types: None }).await
            .expect("Expansion should succeed");

        println!("\n  === Allegation Expansion: {allegation_id} ===");
        println!("  Total chunks: {}", chunks.len());
        for chunk in &chunks {
            println!(
                "  - {} ({}): {}",
                chunk.node_id, chunk.node_type, chunk.title
            );
        }

        assert!(
            !chunks.is_empty(),
            "Should return at least the seed allegation node"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Person-focused expansion returns fewer nodes than Broad
    // -----------------------------------------------------------------------

    /// Test that Person-focused expansion returns fewer nodes than Broad.
    /// This is a behavioral test — we compare node counts between strategies.
    #[tokio::test]
    #[ignore] // Requires Neo4j
    async fn test_person_focused_returns_fewer_nodes() {
        let graph = create_test_graph().await;
        let expander = Neo4jExpander::new(graph);

        // Use a known Evidence seed that has STATED_BY + CONTAINED_IN + CHARACTERIZES.
        let seed_ids = vec!["evidence-phillips-q74".to_string()];

        // Broad should return all neighbors.
        let broad_result = expander
            .expand(
                &seed_ids,
                1,
                &RetrievalStrategy::Broad { node_types: None },
            )
            .await
            .expect("Broad expansion should succeed");

        // Person-focused should return only STATED_BY and ABOUT neighbors.
        let person_result = expander
            .expand(
                &seed_ids,
                1,
                &RetrievalStrategy::Focused {
                    scope: vec![ScopeFilter {
                        filter_type: ScopeFilterType::Person,
                        value: "Phillips".to_string(),
                    }],
                },
            )
            .await
            .expect("Person-focused expansion should succeed");

        println!("\n  === Person-focused vs Broad ===");
        println!("  Broad: {} nodes", broad_result.len());
        println!("  Person-focused: {} nodes", person_result.len());

        // Person result should have fewer or equal nodes.
        assert!(
            person_result.len() <= broad_result.len(),
            "Person-focused ({}) should return <= Broad ({}) nodes",
            person_result.len(),
            broad_result.len(),
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: GraphDirectRetriever — fetch document evidence
    // -----------------------------------------------------------------------

    /// Test that fetch_document_evidence returns Evidence nodes from a known document.
    #[tokio::test]
    #[ignore] // Requires Neo4j
    async fn test_fetch_document_evidence() {
        use colossus_rag::GraphDirectRetriever;

        let graph = create_test_graph().await;
        let retriever = GraphDirectRetriever::new(graph);

        let chunks = retriever
            .fetch_document_evidence("doc-phillips-coa-response-300891")
            .await
            .expect("Should succeed");

        assert!(
            !chunks.is_empty(),
            "Should find evidence in Phillips CoA response"
        );

        // All results should be Evidence type with score 0.0.
        for chunk in &chunks {
            assert_eq!(chunk.node_type, "Evidence");
            assert_eq!(chunk.score, 0.0);
        }

        println!(
            "Found {} evidence nodes from Phillips CoA response",
            chunks.len()
        );
    }
}
