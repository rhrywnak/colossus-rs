//! Per-type Neo4j expansion queries for the graph expander.
//!
//! Contains 7 expansion functions — one per node type in the legal
//! knowledge graph. Each function runs a Cypher query with OPTIONAL MATCH
//! to find neighbors, then extracts nodes and relationships.
//!
//! ## CRITICAL: Cypher queries are IDENTICAL to the original
//!
//! These queries are migrated verbatim from:
//! - `colossus-legal/backend/src/services/graph_expansion_queries.rs`
//!   (Evidence, ComplaintAllegation, MotionClaim)
//! - `colossus-legal/backend/src/services/graph_expansion_minor.rs`
//!   (Harm, Document, Person, Organization)
//!
//! DO NOT modify the Cypher queries without also updating the original.
//!
//! ## Neo4j relationship types in the Awad v. CFS knowledge graph
//!
//! | Relationship | Meaning |
//! |-------------|---------|
//! | STATED_BY | Evidence → Person (who said it) |
//! | ABOUT | Evidence → Person (who it's about) |
//! | CONTAINED_IN | Evidence → Document (source document) |
//! | CHARACTERIZES | Evidence → ComplaintAllegation (evidence supports allegation) |
//! | REBUTS | Evidence → Evidence (one piece rebuts another) |
//! | CONTRADICTS | Evidence ↔ Evidence (mutual contradiction) |
//! | PROVES | MotionClaim → ComplaintAllegation |
//! | RELIES_ON | MotionClaim → Evidence |
//! | SUPPORTS | ComplaintAllegation → LegalCount |
//! | CAUSED_BY | Harm → ComplaintAllegation |
//! | EVIDENCED_BY | Harm → Evidence |
//! | DAMAGES_FOR | Harm → LegalCount |
//! | APPEARS_IN | MotionClaim → Document |
//!
//! ## Rust Learning: Cypher OPTIONAL MATCH
//!
//! `OPTIONAL MATCH` is like a LEFT JOIN in SQL — if the pattern doesn't
//! match, the variables are bound to `null` rather than eliminating the row.
//! This is essential because a seed node might not have all relationship
//! types (e.g., an Evidence node with no rebuttals).
//!
//! The downside: multiple OPTIONAL MATCHes create a cartesian product of
//! results. If an Evidence has 3 speakers and 2 documents, you get 6 rows.
//! Our deduplication (`try_extract_node` + `seen` HashSet) handles this —
//! each node is extracted only once.

use neo4rs::{query, Graph};
use std::collections::HashSet;

use crate::error::RagError;
use crate::expander::{get_str, try_extract_node, ExpandedNode, ExpandedRel};

/// Map a `neo4rs::Error` to `RagError::ExpandError`.
///
/// Used for both `graph.execute()` and `result.next()` calls — both
/// return `neo4rs::Error` in neo4rs 0.8.
fn map_neo4j_err(e: neo4rs::Error) -> RagError {
    RagError::ExpandError(e.to_string())
}

// ---------------------------------------------------------------------------
// Evidence expansion
// ---------------------------------------------------------------------------

/// Expand an Evidence seed: speaker, subject, document, allegation,
/// rebuttals, contradictions.
///
/// This is the most complex expansion — Evidence nodes are the richest
/// in the legal knowledge graph, with connections to people, documents,
/// allegations, and other evidence that rebuts or contradicts them.
pub(crate) async fn expand_evidence(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (e:Evidence {id: $id})
        OPTIONAL MATCH (e)-[:STATED_BY]->(speaker)
        OPTIONAL MATCH (e)-[:ABOUT]->(subject)
        OPTIONAL MATCH (e)-[:CONTAINED_IN]->(doc:Document)
        OPTIONAL MATCH (e)-[:CHARACTERIZES]->(allegation:ComplaintAllegation)
        OPTIONAL MATCH (e)<-[:REBUTS]-(rebuttal:Evidence)
        OPTIONAL MATCH (e)-[:CONTRADICTS]-(contradiction:Evidence)
        RETURN e.id AS eid, e.title AS etitle, e.verbatim_quote AS equote,
               e.significance AS esig, e.page_number AS epage,
               speaker.id AS sid, speaker.name AS sname,
               subject.id AS subid, subject.name AS subname,
               doc.id AS did, doc.title AS dtitle, doc.document_type AS dtype,
               allegation.id AS aid, allegation.title AS atitle,
               allegation.evidence_status AS astatus,
               rebuttal.id AS rid, rebuttal.title AS rtitle,
               contradiction.id AS cid, contradiction.title AS ctitle";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        // Seed node itself.
        if let Some(n) = try_extract_node(
            &row, "eid", "Evidence",
            &[("etitle", "title"), ("equote", "verbatim_quote"),
              ("esig", "significance"), ("epage", "page_number")],
            seen,
        ) { nodes.push(n); }

        // Speaker (who stated this evidence).
        let sid = get_str(&row, "sid");
        if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
            rels.push(ExpandedRel::new(id, &sid, "STATED_BY"));
            nodes.push(n);
        }

        // Subject (who the evidence is about).
        let subid = get_str(&row, "subid");
        if let Some(n) = try_extract_node(&row, "subid", "Person", &[("subname", "name")], seen) {
            rels.push(ExpandedRel::new(id, &subid, "ABOUT"));
            nodes.push(n);
        }

        // Source document.
        let did = get_str(&row, "did");
        if let Some(n) = try_extract_node(
            &row, "did", "Document", &[("dtitle", "title"), ("dtype", "document_type")], seen,
        ) {
            rels.push(ExpandedRel::new(id, &did, "CONTAINED_IN"));
            nodes.push(n);
        }

        // Allegation this evidence characterizes.
        let aid = get_str(&row, "aid");
        if let Some(n) = try_extract_node(
            &row, "aid", "ComplaintAllegation",
            &[("atitle", "title"), ("astatus", "evidence_status")], seen,
        ) {
            rels.push(ExpandedRel::new(id, &aid, "CHARACTERIZES"));
            nodes.push(n);
        }

        // Evidence that rebuts this evidence.
        let rid = get_str(&row, "rid");
        if let Some(n) = try_extract_node(&row, "rid", "Evidence", &[("rtitle", "title")], seen) {
            rels.push(ExpandedRel::new(&rid, id, "REBUTS"));
            nodes.push(n);
        }

        // Evidence that contradicts this evidence.
        let cid_val = get_str(&row, "cid");
        if let Some(n) = try_extract_node(&row, "cid", "Evidence", &[("ctitle", "title")], seen) {
            rels.push(ExpandedRel::new(id, &cid_val, "CONTRADICTS"));
            nodes.push(n);
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// ComplaintAllegation expansion
// ---------------------------------------------------------------------------

/// Expand a ComplaintAllegation seed: claims, evidence, documents,
/// speakers, legal counts, harms.
pub(crate) async fn expand_allegation(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (a:ComplaintAllegation {id: $id})
        OPTIONAL MATCH (claim:MotionClaim)-[:PROVES]->(a)
        OPTIONAL MATCH (claim)-[:RELIES_ON]->(evidence:Evidence)
        OPTIONAL MATCH (evidence)-[:CONTAINED_IN]->(doc:Document)
        OPTIONAL MATCH (evidence)-[:STATED_BY]->(speaker)
        OPTIONAL MATCH (a)-[:SUPPORTS]->(count:LegalCount)
        OPTIONAL MATCH (harm:Harm)-[:CAUSED_BY]->(a)
        RETURN a.id AS aid, a.title AS atitle, a.evidence_status AS astatus,
               a.allegation AS aalleg,
               claim.id AS cid, claim.title AS ctitle,
               evidence.id AS eid, evidence.title AS etitle,
               evidence.verbatim_quote AS equote,
               doc.id AS did, doc.title AS dtitle,
               speaker.id AS sid, speaker.name AS sname,
               count.id AS lcid, count.title AS lctitle,
               harm.id AS hid, harm.title AS htitle, harm.amount AS hamount";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        if let Some(n) = try_extract_node(
            &row, "aid", "ComplaintAllegation",
            &[("atitle", "title"), ("astatus", "evidence_status"), ("aalleg", "allegation")],
            seen,
        ) { nodes.push(n); }

        let cid = get_str(&row, "cid");
        if let Some(n) = try_extract_node(&row, "cid", "MotionClaim", &[("ctitle", "title")], seen) {
            rels.push(ExpandedRel::new(&cid, id, "PROVES"));
            nodes.push(n);
        }

        let eid = get_str(&row, "eid");
        if let Some(n) = try_extract_node(
            &row, "eid", "Evidence", &[("etitle", "title"), ("equote", "verbatim_quote")], seen,
        ) {
            if !cid.is_empty() { rels.push(ExpandedRel::new(&cid, &eid, "RELIES_ON")); }
            nodes.push(n);
        }

        let did = get_str(&row, "did");
        if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
            if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
            nodes.push(n);
        }

        if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
            nodes.push(n);
        }

        let lcid = get_str(&row, "lcid");
        if let Some(n) = try_extract_node(&row, "lcid", "LegalCount", &[("lctitle", "title")], seen) {
            rels.push(ExpandedRel::new(id, &lcid, "SUPPORTS"));
            nodes.push(n);
        }

        let hid = get_str(&row, "hid");
        if let Some(n) = try_extract_node(
            &row, "hid", "Harm", &[("htitle", "title"), ("hamount", "amount")], seen,
        ) {
            rels.push(ExpandedRel::new(&hid, id, "CAUSED_BY"));
            nodes.push(n);
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// MotionClaim expansion
// ---------------------------------------------------------------------------

/// Expand a MotionClaim seed: evidence, documents, speakers, allegation,
/// motion documents.
pub(crate) async fn expand_motion_claim(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
) -> Result<(Vec<ExpandedNode>, Vec<ExpandedRel>), RagError> {
    let mut nodes = Vec::new();
    let mut rels = Vec::new();

    let cypher = "MATCH (m:MotionClaim {id: $id})
        OPTIONAL MATCH (m)-[:RELIES_ON]->(evidence:Evidence)
        OPTIONAL MATCH (evidence)-[:CONTAINED_IN]->(doc:Document)
        OPTIONAL MATCH (evidence)-[:STATED_BY]->(speaker)
        OPTIONAL MATCH (m)-[:PROVES]->(allegation:ComplaintAllegation)
        OPTIONAL MATCH (m)-[:APPEARS_IN]->(motion_doc:Document)
        RETURN m.id AS mid, m.title AS mtitle, m.claim_text AS mtext,
               m.significance AS msig,
               evidence.id AS eid, evidence.title AS etitle,
               evidence.verbatim_quote AS equote,
               doc.id AS did, doc.title AS dtitle,
               speaker.id AS sid, speaker.name AS sname,
               allegation.id AS aid, allegation.title AS atitle,
               motion_doc.id AS mdid, motion_doc.title AS mdtitle";

    let mut result = graph.execute(query(cypher).param("id", id)).await.map_err(map_neo4j_err)?;

    while let Some(row) = result.next().await.map_err(map_neo4j_err)? {
        if let Some(n) = try_extract_node(
            &row, "mid", "MotionClaim",
            &[("mtitle", "title"), ("mtext", "claim_text"), ("msig", "significance")],
            seen,
        ) { nodes.push(n); }

        let eid = get_str(&row, "eid");
        if let Some(n) = try_extract_node(
            &row, "eid", "Evidence", &[("etitle", "title"), ("equote", "verbatim_quote")], seen,
        ) {
            rels.push(ExpandedRel::new(id, &eid, "RELIES_ON"));
            nodes.push(n);
        }

        let did = get_str(&row, "did");
        if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
            if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
            nodes.push(n);
        }

        if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
            nodes.push(n);
        }

        let aid = get_str(&row, "aid");
        if let Some(n) = try_extract_node(
            &row, "aid", "ComplaintAllegation", &[("atitle", "title")], seen,
        ) {
            rels.push(ExpandedRel::new(id, &aid, "PROVES"));
            nodes.push(n);
        }

        let mdid = get_str(&row, "mdid");
        if let Some(n) = try_extract_node(&row, "mdid", "Document", &[("mdtitle", "title")], seen) {
            rels.push(ExpandedRel::new(id, &mdid, "APPEARS_IN"));
            nodes.push(n);
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// Harm expansion
// ---------------------------------------------------------------------------

/// Expand a Harm seed: allegation, evidence, documents, legal count.
pub(crate) async fn expand_harm(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
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
        if let Some(n) = try_extract_node(
            &row, "hid", "Harm",
            &[("htitle", "title"), ("hdesc", "description"), ("hamount", "amount")],
            seen,
        ) { nodes.push(n); }

        let aid = get_str(&row, "aid");
        if let Some(n) = try_extract_node(&row, "aid", "ComplaintAllegation", &[("atitle", "title")], seen) {
            rels.push(ExpandedRel::new(id, &aid, "CAUSED_BY"));
            nodes.push(n);
        }

        let eid = get_str(&row, "eid");
        if let Some(n) = try_extract_node(
            &row, "eid", "Evidence", &[("etitle", "title"), ("equote", "verbatim_quote")], seen,
        ) {
            rels.push(ExpandedRel::new(id, &eid, "EVIDENCED_BY"));
            nodes.push(n);
        }

        let did = get_str(&row, "did");
        if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
            if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
            nodes.push(n);
        }

        let cid = get_str(&row, "cid");
        if let Some(n) = try_extract_node(&row, "cid", "LegalCount", &[("ctitle", "title")], seen) {
            rels.push(ExpandedRel::new(id, &cid, "DAMAGES_FOR"));
            nodes.push(n);
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
        if let Some(n) = try_extract_node(
            &row, "did", "Document", &[("dtitle", "title"), ("dtype", "document_type")], seen,
        ) { nodes.push(n); }

        let eid = get_str(&row, "eid");
        if let Some(n) = try_extract_node(&row, "eid", "Evidence", &[("etitle", "title")], seen) {
            rels.push(ExpandedRel::new(&eid, id, "CONTAINED_IN"));
            nodes.push(n);
        }

        if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
            nodes.push(n);
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
        if let Some(n) = try_extract_node(
            &row, "pid", "Person",
            &[("pname", "name"), ("prole", "role"), ("pdesc", "description")],
            seen,
        ) { nodes.push(n); }

        let eid = get_str(&row, "eid");
        if let Some(n) = try_extract_node(&row, "eid", "Evidence", &[("etitle", "title")], seen) {
            rels.push(ExpandedRel::new(&eid, id, "STATED_BY"));
            nodes.push(n);
        }

        let did = get_str(&row, "did");
        if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
            if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
            nodes.push(n);
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
        if let Some(n) = try_extract_node(
            &row, "oid", "Organization",
            &[("oname", "name"), ("orole", "role")],
            seen,
        ) { nodes.push(n); }

        let eid = get_str(&row, "eid");
        if let Some(n) = try_extract_node(&row, "eid", "Evidence", &[("etitle", "title")], seen) {
            rels.push(ExpandedRel::new(&eid, id, "STATED_BY"));
            nodes.push(n);
        }

        let did = get_str(&row, "did");
        if let Some(n) = try_extract_node(&row, "did", "Document", &[("dtitle", "title")], seen) {
            if !eid.is_empty() { rels.push(ExpandedRel::new(&eid, &did, "CONTAINED_IN")); }
            nodes.push(n);
        }
    }

    Ok((nodes, rels))
}
