//! Knowledge-graph structured memory layer (Phase 4).
//!
//! Structured, verifiable recall — the "needle in a haystack" (NIAH)
//! substrate. Facts live as discrete [`KgTriple`](crate::types::KgTriple)
//! values keyed by `(subject, predicate)`. Retrieval via
//! [`KgStore::lookup`](crate::traits::KgStore::lookup) is exact: a stored
//! fact `(s, p, o)` is recalled as `[o, ...]`, with 100% precision and zero
//! hallucinated recall.
//!
//! The decode loop queries the KG for grounding; the KG does not decide
//! fluency (the draft model) or syntactic validity (the pruner). It owns
//! SEMANTIC GROUNDING — discrete, checkable facts that anchor generation.
//!
//! # Submodules
//!
//! - [`projection`] — shared deterministic BLAKE3 token→vector projection.
//!   Single source of truth for all KG embeddings (no learned weights).
//! - [`in_memory`] — `InMemoryKgStore`, a HashMap-indexed concrete
//!   [`KgStore`](crate::traits::KgStore) with role-scoped triple embeddings.
//! - [`schema_centroid`] — `SchemaCentroid`, per-class embedding centroids
//!   for structured entity initialization (grounded init, not random noise).

pub mod in_memory;
pub mod projection;
pub mod schema_centroid;

pub use in_memory::InMemoryKgStore;
pub use schema_centroid::SchemaCentroid;
