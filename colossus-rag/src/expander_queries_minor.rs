//! Minor Neo4j expansion queries: Harm, Document, Person, Organization.
//!
//! Split from `expander_queries.rs` to keep each module under the 300-line
//! code limit. These are the simpler expansion functions with fewer
//! OPTIONAL MATCH clauses.
//!
//! ## CRITICAL: Cypher queries are IDENTICAL to the original
//!
//! Migrated verbatim from `colossus-legal/backend/src/services/graph_expansion_minor.rs`.
//! DO NOT modify the Cypher queries without also updating the original.

use neo4rs::{query, Graph};
use std::collections::HashSet;

use crate::error::RagError;
use crate::expander::{get_str, try_extract_node, ExpandedNode, ExpandedRel};
use crate::expander_queries::is_rel_allowed;

/// Map a `neo4rs::Error` to `RagError::ExpandError`.
fn map_neo4j_err(e: neo4rs::Error) -> RagError {
    RagError::ExpandError(e.to_string())
}

// ---------------------------------------------------------------------------
// Harm expansion
// ---------------------------------------------------------------------------

/// Expand a Harm seed: allegation, evidence, documents, legal count.
pub(crate) async fn expand_harm(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
    allowed_rels: &Option<HashSet<&str>>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (h:Harm {id: $id})
        OPTIONAL MATCH (h)-[:CAUSED_BY]->(allegation:ComplaintAllegation)
        OPTIONAL MATCH (h)-[:EVIDENCED_BY]->(evidence:Evidence)
        OPTIONAL MATCH (evidence)-[:CONTAINED_IN]->(doc:Document)
        OPTIONAL MATCH (h)-[:DAMAGES_FOR]->(count:LegalCount)
        RETURN h.id AS hid, h.title AS htitle, h.description AS hdesc,
               h.amount AS hamount,
               allegation.id AS aid, allegation.title AS atitle,
               evidence.id AS eid, evidence.title AS etitle,
               evidence.verbatim_quote AS equote,
               doc.id AS did, doc.title AS dtitle,
               count.id AS cid, count.title AS ctitle";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        // Seed node — always extracted.
        if let Some(n) = try_extract_node(
            &row, "hid", "Harm",
            &[("htitle", "title"), ("hdesc", "description"), ("hamount", "amount")],
            seen,
        ) { nodes.push(n); }

        // Allegation this harm is CAUSED_BY.
        if is_rel_allowed("CAUSED_BY", allowed_rels) {
            let aid = get_str(&row, "aid");
            if let Some(n) = try_extract_node(&row, "aid", "ComplaintAllegation", &[("atitle", "title")], seen) {
                rels.push(ExpandedRel::new(id, &aid, "CAUSED_BY"));
                nodes.push(n);
            }
        }

        // Evidence this harm is EVIDENCED_BY.
        if is_rel_allowed("EVIDENCED_BY", allowed_rels) {
            let eid = get_str(&row, "eid");
            if let Some(n) = try_extract_node(
                &row, "eid", "Evidence", &[("etitle", "title"), ("equote", "verbatim_quote")], seen,
            ) {
                rels.push(ExpandedRel::new(id, &eid, "EVIDENCED_BY"));
                nodes.push(n);
            }
        }

        // Document the evidence is CONTAINED_IN.
        if is_rel_allowed("CONTAINED_IN", allowed_rels) {
            let eid = get_str(&row, "eid");
            let did = get_str(&row, "did");
            if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
                if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
                nodes.push(n);
            }
        }

        // Legal count this harm DAMAGES_FOR.
        if is_rel_allowed("DAMAGES_FOR", allowed_rels) {
            let cid = get_str(&row, "cid");
            if let Some(n) = try_extract_node(&row, "cid", "LegalCount", &[("ctitle", "title")], seen) {
                rels.push(ExpandedRel::new(id, &cid, "DAMAGES_FOR"));
                nodes.push(n);
            }
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// Document expansion
// ---------------------------------------------------------------------------

/// Expand a Document seed: contained evidence, speakers.
///
/// Uses LIMIT 20 to prevent returning too many evidence nodes from
/// large documents (e.g., a deposition with 50+ evidence excerpts).
pub(crate) async fn expand_document(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
    allowed_rels: &Option<HashSet<&str>>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (d:Document {id: $id})
        OPTIONAL MATCH (evidence:Evidence)-[:CONTAINED_IN]->(d)
        OPTIONAL MATCH (evidence)-[:STATED_BY]->(speaker)
        RETURN d.id AS did, d.title AS dtitle, d.document_type AS dtype,
               evidence.id AS eid, evidence.title AS etitle,
               speaker.id AS sid, speaker.name AS sname
        LIMIT 20";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        // Seed node — always extracted.
        if let Some(n) = try_extract_node(
            &row, "did", "Document", &[("dtitle", "title"), ("dtype", "document_type")], seen,
        ) { nodes.push(n); }

        // Evidence CONTAINED_IN this document (inbound).
        if is_rel_allowed("CONTAINED_IN", allowed_rels) {
            let eid = get_str(&row, "eid");
            if let Some(n) = try_extract_node(&row, "eid", "Evidence", &[("etitle", "title")], seen) {
                rels.push(ExpandedRel::new(&eid, id, "CONTAINED_IN"));
                nodes.push(n);
            }
        }

        // Speaker — evidence STATED_BY person.
        if is_rel_allowed("STATED_BY", allowed_rels) {
            if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
                nodes.push(n);
            }
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// Person expansion
// ---------------------------------------------------------------------------

/// Expand a Person seed: evidence stated by them, documents.
///
/// Uses LIMIT 15 to cap results for people with many evidence excerpts.
pub(crate) async fn expand_person(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
    allowed_rels: &Option<HashSet<&str>>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (p:Person {id: $id})
        OPTIONAL MATCH (evidence:Evidence)-[:STATED_BY]->(p)
        OPTIONAL MATCH (evidence)-[:CONTAINED_IN]->(doc:Document)
        RETURN p.id AS pid, p.name AS pname, p.role AS prole,
               p.description AS pdesc,
               evidence.id AS eid, evidence.title AS etitle,
               doc.id AS did, doc.title AS dtitle
        LIMIT 15";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        // Seed node — always extracted.
        if let Some(n) = try_extract_node(
            &row, "pid", "Person",
            &[("pname", "name"), ("prole", "role"), ("pdesc", "description")],
            seen,
        ) { nodes.push(n); }

        // Evidence STATED_BY this person (inbound).
        if is_rel_allowed("STATED_BY", allowed_rels) {
            let eid = get_str(&row, "eid");
            if let Some(n) = try_extract_node(&row, "eid", "Evidence", &[("etitle", "title")], seen) {
                rels.push(ExpandedRel::new(&eid, id, "STATED_BY"));
                nodes.push(n);
            }
        }

        // Document the evidence is CONTAINED_IN.
        if is_rel_allowed("CONTAINED_IN", allowed_rels) {
            let eid = get_str(&row, "eid");
            let did = get_str(&row, "did");
            if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
                if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
                nodes.push(n);
            }
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// Organization expansion
// ---------------------------------------------------------------------------

/// Expand an Organization seed: same pattern as Person.
pub(crate) async fn expand_organization(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
    allowed_rels: &Option<HashSet<&str>>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (o:Organization {id: $id})
        OPTIONAL MATCH (evidence:Evidence)-[:STATED_BY]->(o)
        OPTIONAL MATCH (evidence)-[:CONTAINED_IN]->(doc:Document)
        RETURN o.id AS oid, o.name AS oname, o.role AS orole,
               evidence.id AS eid, evidence.title AS etitle,
               doc.id AS did, doc.title AS dtitle
        LIMIT 15";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        // Seed node — always extracted.
        if let Some(n) = try_extract_node(
            &row, "oid", "Organization",
            &[("oname", "name"), ("orole", "role")],
            seen,
        ) { nodes.push(n); }

        // Evidence STATED_BY this organization (inbound).
        if is_rel_allowed("STATED_BY", allowed_rels) {
            let eid = get_str(&row, "eid");
            if let Some(n) = try_extract_node(&row, "eid", "Evidence", &[("etitle", "title")], seen) {
                rels.push(ExpandedRel::new(&eid, id, "STATED_BY"));
                nodes.push(n);
            }
        }

        // Document the evidence is CONTAINED_IN.
        if is_rel_allowed("CONTAINED_IN", allowed_rels) {
            let eid = get_str(&row, "eid");
            let did = get_str(&row, "did");
            if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
                if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
                nodes.push(n);
            }
        }
    }

    Ok((nodes, rels))
}
