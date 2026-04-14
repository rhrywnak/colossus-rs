//! All SQL state transitions for pipeline_jobs.
//!
//! Each function corresponds to one FSM transition. All SQL binds use
//! JobStatus and JobControl enum values — no raw strings anywhere.
//! FOR UPDATE SKIP LOCKED on claim() prevents multiple workers from
//! claiming the same job under concurrent load.

// Stub — full implementation in P1-7.
