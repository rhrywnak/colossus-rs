//! Major Neo4j expansion queries: Evidence, ComplaintAllegation, MotionClaim.
//!
//! These are the three most complex expansion functions — each has multiple
//! OPTIONAL MATCH clauses following different relationship types.
//!
//! ## Relationship filtering (WP-2)
//!
//! Each function now accepts `allowed_rels: &Option<HashSet<&str>>`.
//! When `Some(set)`, only relationships in the set are followed (neighbor
//! nodes from non-allowed relationships are skipped). When `None` (Broad
//! mode), all relationships are followed — preserving the original behavior.
//!
//! The Cypher queries are NOT modified — we still run the full OPTIONAL MATCH
//! query and filter in Rust. This is safer: if the category is wrong, we just
//! include extra nodes rather than missing the seed node entirely.
//!
//! ## CRITICAL: Cypher queries are IDENTICAL to the original
//!
//! Migrated verbatim from `colossus-legal/backend/src/services/graph_expansion_queries.rs`.
//! DO NOT modify the Cypher queries without also updating the original.
//!
//! ## Neo4j relationship types in the Awad v. CFS knowledge graph
//!
//! | Relationship | Meaning |
//! |-------------|---------|
//! | STATED_BY | Evidence -> Person (who said it) |
//! | ABOUT | Evidence -> Person (who it's about) |
//! | CONTAINED_IN | Evidence -> Document (source document) |
//! | CHARACTERIZES | Evidence -> ComplaintAllegation (evidence supports allegation) |
//! | REBUTS | Evidence -> Evidence (one piece rebuts another) |
//! | CONTRADICTS | Evidence <-> Evidence (mutual contradiction) |
//! | PROVES | MotionClaim -> ComplaintAllegation |
//! | RELIES_ON | MotionClaim -> Evidence |
//! | SUPPORTS | ComplaintAllegation -> LegalCount |
//! | CAUSED_BY | Harm -> ComplaintAllegation |
//! | APPEARS_IN | MotionClaim -> Document |

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

/// Check if a relationship type should be followed during expansion.
///
/// When `allowed_rels` is `None` (Broad mode), everything is allowed.
/// When `Some(set)`, only relationships in the set are followed.
///
/// ## Rust Learning: Option<HashSet> as "allow all or allow some"
///
/// Using `Option<HashSet<&str>>` instead of just `HashSet<&str>` lets us
/// distinguish "no filter" (None = follow everything) from "empty filter"
/// (Some(empty set) = follow nothing). This is more expressive than a
/// HashSet alone, where an empty set would ambiguously mean either
/// "allow nothing" or "no filtering specified."
pub(crate) fn is_rel_allowed(rel_type: &str, allowed_rels: &Option<HashSet<&str>>) -> bool {
    match allowed_rels {
        None => true, // Broad mode: everything allowed
        Some(set) => set.contains(rel_type),
    }
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
    allowed_rels: &Option<HashSet<&str>>,
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
        // Seed node itself — always extracted regardless of allowed_rels.
        if let Some(n) = try_extract_node(
            &row, "eid", "Evidence",
            &[("etitle", "title"), ("equote", "verbatim_quote"),
              ("esig", "significance"), ("epage", "page_number")],
            seen,
        ) { nodes.push(n); }

        // Speaker (who stated this evidence) — only if STATED_BY is allowed.
        if is_rel_allowed("STATED_BY", allowed_rels) {
            let sid = get_str(&row, "sid");
            if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
                rels.push(ExpandedRel::new(id, &sid, "STATED_BY"));
                nodes.push(n);
            }
        }

        // Subject (who the evidence is about) — only if ABOUT is allowed.
        if is_rel_allowed("ABOUT", allowed_rels) {
            let subid = get_str(&row, "subid");
            if let Some(n) = try_extract_node(&row, "subid", "Person", &[("subname", "name")], seen) {
                rels.push(ExpandedRel::new(id, &subid, "ABOUT"));
                nodes.push(n);
            }
        }

        // Source document — only if CONTAINED_IN is allowed.
        if is_rel_allowed("CONTAINED_IN", allowed_rels) {
            let did = get_str(&row, "did");
            if let Some(n) = try_extract_node(
                &row, "did", "Document", &[("dtitle", "title"), ("dtype", "document_type")], seen,
            ) {
                rels.push(ExpandedRel::new(id, &did, "CONTAINED_IN"));
                nodes.push(n);
            }
        }

        // Allegation this evidence characterizes — only if CHARACTERIZES is allowed.
        if is_rel_allowed("CHARACTERIZES", allowed_rels) {
            let aid = get_str(&row, "aid");
            if let Some(n) = try_extract_node(
                &row, "aid", "ComplaintAllegation",
                &[("atitle", "title"), ("astatus", "evidence_status")], seen,
            ) {
                rels.push(ExpandedRel::new(id, &aid, "CHARACTERIZES"));
                nodes.push(n);
            }
        }

        // Evidence that rebuts this evidence — only if REBUTS is allowed.
        if is_rel_allowed("REBUTS", allowed_rels) {
            let rid = get_str(&row, "rid");
            if let Some(n) = try_extract_node(&row, "rid", "Evidence", &[("rtitle", "title")], seen) {
                rels.push(ExpandedRel::new(&rid, id, "REBUTS"));
                nodes.push(n);
            }
        }

        // Evidence that contradicts this evidence — only if CONTRADICTS is allowed.
        if is_rel_allowed("CONTRADICTS", allowed_rels) {
            let cid_val = get_str(&row, "cid");
            if let Some(n) = try_extract_node(&row, "cid", "Evidence", &[("ctitle", "title")], seen) {
                rels.push(ExpandedRel::new(id, &cid_val, "CONTRADICTS"));
                nodes.push(n);
            }
        }
    }

    Ok((nodes, rels))
}

// ---------------------------------------------------------------------------
// ComplaintAllegation expansion
// ---------------------------------------------------------------------------

/// Expand a ComplaintAllegation seed: claims, evidence, documents,
/// speakers, legal counts, harms.
///
/// ## Chained relationships
///
/// The Cypher has a chain: claim -> RELIES_ON -> evidence -> CONTAINED_IN -> doc
/// and evidence -> STATED_BY -> speaker. Filtering is applied per-relationship:
/// - If PROVES is not allowed, skip claim extraction
/// - If RELIES_ON is not allowed, skip evidence extraction
/// - If CONTAINED_IN is not allowed, skip doc extraction
/// - If STATED_BY is not allowed, skip speaker extraction
/// - SUPPORTS and CAUSED_BY are checked independently
pub(crate) async fn expand_allegation(
    graph: &Graph,
    id: &str,
    seen: &mut HashSet<String>,
    allowed_rels: &Option<HashSet<&str>>,
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
        // Seed node — always extracted.
        if let Some(n) = try_extract_node(
            &row, "aid", "ComplaintAllegation",
            &[("atitle", "title"), ("astatus", "evidence_status"), ("aalleg", "allegation")],
            seen,
        ) { nodes.push(n); }

        // Claim that PROVES this allegation.
        if is_rel_allowed("PROVES", allowed_rels) {
            let cid = get_str(&row, "cid");
            if let Some(n) = try_extract_node(&row, "cid", "MotionClaim", &[("ctitle", "title")], seen) {
                rels.push(ExpandedRel::new(&cid, id, "PROVES"));
                nodes.push(n);
            }
        }

        // Evidence that claim RELIES_ON.
        if is_rel_allowed("RELIES_ON", allowed_rels) {
            let cid = get_str(&row, "cid");
            let eid = get_str(&row, "eid");
            if let Some(n) = try_extract_node(
                &row, "eid", "Evidence", &[("etitle", "title"), ("equote", "verbatim_quote")], seen,
            ) {
                if !cid.is_empty() { rels.push(ExpandedRel::new(&cid, &eid, "RELIES_ON")); }
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

        // Speaker — evidence STATED_BY person.
        if is_rel_allowed("STATED_BY", allowed_rels) {
            if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
                nodes.push(n);
            }
        }

        // Legal count — allegation SUPPORTS count.
        if is_rel_allowed("SUPPORTS", allowed_rels) {
            let lcid = get_str(&row, "lcid");
            if let Some(n) = try_extract_node(&row, "lcid", "LegalCount", &[("lctitle", "title")], seen) {
                rels.push(ExpandedRel::new(id, &lcid, "SUPPORTS"));
                nodes.push(n);
            }
        }

        // Harm CAUSED_BY this allegation.
        if is_rel_allowed("CAUSED_BY", allowed_rels) {
            let hid = get_str(&row, "hid");
            if let Some(n) = try_extract_node(
                &row, "hid", "Harm", &[("htitle", "title"), ("hamount", "amount")], seen,
            ) {
                rels.push(ExpandedRel::new(&hid, id, "CAUSED_BY"));
                nodes.push(n);
            }
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
    allowed_rels: &Option<HashSet<&str>>,
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
        // Seed node — always extracted.
        if let Some(n) = try_extract_node(
            &row, "mid", "MotionClaim",
            &[("mtitle", "title"), ("mtext", "claim_text"), ("msig", "significance")],
            seen,
        ) { nodes.push(n); }

        // Evidence the claim RELIES_ON.
        if is_rel_allowed("RELIES_ON", allowed_rels) {
            let eid = get_str(&row, "eid");
            if let Some(n) = try_extract_node(
                &row, "eid", "Evidence", &[("etitle", "title"), ("equote", "verbatim_quote")], seen,
            ) {
                rels.push(ExpandedRel::new(id, &eid, "RELIES_ON"));
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

        // Speaker — evidence STATED_BY person.
        if is_rel_allowed("STATED_BY", allowed_rels) {
            if let Some(n) = try_extract_node(&row, "sid", "Person", &[("sname", "name")], seen) {
                nodes.push(n);
            }
        }

        // Allegation the claim PROVES.
        if is_rel_allowed("PROVES", allowed_rels) {
            let aid = get_str(&row, "aid");
            if let Some(n) = try_extract_node(
                &row, "aid", "ComplaintAllegation", &[("atitle", "title")], seen,
            ) {
                rels.push(ExpandedRel::new(id, &aid, "PROVES"));
                nodes.push(n);
            }
        }

        // Motion document the claim APPEARS_IN.
        if is_rel_allowed("APPEARS_IN", allowed_rels) {
            let mdid = get_str(&row, "mdid");
            if let Some(n) = try_extract_node(&row, "mdid", "Document", &[("mdtitle", "title")], seen) {
                rels.push(ExpandedRel::new(id, &mdid, "APPEARS_IN"));
                nodes.push(n);
            }
        }
    }

    Ok((nodes, rels))
}
