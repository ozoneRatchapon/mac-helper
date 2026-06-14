//! In-memory [`KgStore`](crate::traits::KgStore) backed by a hash index.
//!
//! Indexes triples by `(subject, predicate)` for O(1) average lookup and
//! stores the full triple set for projection and iteration. Embeddings use a
//! deterministic BLAKE3-seeded projection (no learned weights, no RNG): each
//! triple maps to a fixed vector with role-positioned thirds
//! `[subject | predicate | object]`, so distinct triples yield distinct
//! embeddings with overwhelming probability.
//!
//! # Why no learned weights?
//!
//! The modelless thesis (katopz): intelligence lives in the symbolic layer,
//! not in trained embeddings. A deterministic hash projection gives us
//! stable, reproducible embeddings for mid-layer K/V injection without
//! requiring a training step — the KG is the memory, the projection is just
//! a fixed coordinate rotation of the token space.
//!
//! # Concurrency
//!
//! `InMemoryKgStore` is `Send + Sync` via internal shared state (no
//! `UnsafeCell`). Writes (`insert`) require `&mut self`; reads (`lookup`,
//! `embed`) take `&self`. For concurrent read/write workloads, wrap in a
//! `Mutex` or use the bandit-style interior-mutability pattern from
//! [`HotSwapPruner`](crate::pruners::HotSwapPruner).

use std::collections::HashMap;

use crate::kg::projection::token_vector;
use crate::traits::KgStore;
use crate::types::{KgTriple, TokenId};

/// In-memory knowledge-graph store.
///
/// Stores `(subject, predicate, object)` triples and answers lookups by
/// `(subject, predicate)`. Embeddings are deterministic and role-aware.
///
/// # Example
///
/// ```
/// use ns_engine::kg::InMemoryKgStore;
/// use ns_engine::traits::KgStore;
/// use ns_engine::types::KgTriple;
///
/// let mut store = InMemoryKgStore::new(32); // 32 dims per role → 96 total
/// store.insert(KgTriple::new(1, 2, 3)); // (alice, knows, bob)
/// store.insert(KgTriple::new(1, 2, 4)); // (alice, knows, carol)
///
/// let objects = store.lookup(1, 2);
/// assert_eq!(objects, vec![3, 4]);
///
/// let v = store.embed(KgTriple::new(1, 2, 3));
/// assert_eq!(v.len(), 96);
/// ```
pub struct InMemoryKgStore {
    /// `(subject, predicate)` → ordered, deduped object tokens.
    ///
    /// Insertion order is preserved within each `(subject, predicate)` bucket
    /// so that `lookup` results are deterministic and reproducible.
    index: HashMap<(TokenId, TokenId), Vec<TokenId>>,
    /// All stored triples, deduped, in insertion order.
    ///
    /// Kept alongside `index` so `triples()` iteration is stable and
    /// `embed` has a single source of truth for triple identity.
    triples: Vec<KgTriple>,
    /// Per-role latent dimension.
    ///
    /// Each of (subject, predicate, object) contributes `role_dim`
    /// dimensions; total embedding dimension is `role_dim * 3`. A typical
    /// value is 32 → 96-dim embeddings.
    role_dim: usize,
}

impl InMemoryKgStore {
    /// Create an empty store with the given per-role embedding dimension.
    ///
    /// Total embedding dimension will be `role_dim * 3`. For example,
    /// `new(32)` produces 96-dimensional embeddings.
    ///
    /// Passing `role_dim = 0` yields a store whose `embed` always returns
    /// an empty vector; this is degenerate but not erroneous, and is
    /// permitted so callers can disable embeddings without restructuring.
    pub fn new(role_dim: usize) -> Self {
        Self {
            index: HashMap::new(),
            triples: Vec::new(),
            role_dim,
        }
    }

    /// Insert a triple.
    ///
    /// Dedupes by full triple equality: re-inserting a triple that is
    /// already stored (same subject, predicate, object) is a no-op and
    /// does not duplicate the object in any bucket.
    ///
    /// Returns `true` if the triple was newly inserted, `false` if it was
    /// already present.
    pub fn insert(&mut self, triple: KgTriple) -> bool {
        let key = (triple.subject, triple.predicate);
        let objects = self.index.entry(key).or_default();
        if objects.contains(&triple.object) {
            return false;
        }
        objects.push(triple.object);
        self.triples.push(triple);
        true
    }

    /// Number of distinct triples currently stored.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether the store holds no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Iterator over all stored triples in insertion order.
    pub fn triples(&self) -> impl Iterator<Item = &KgTriple> {
        self.triples.iter()
    }

    /// Per-role embedding dimension (one third of the total).
    pub fn role_dim(&self) -> usize {
        self.role_dim
    }
}

impl Default for InMemoryKgStore {
    fn default() -> Self {
        // 32 dims per role → 96-dim embeddings. A reasonable default that
        // keeps embeddings small enough for mid-layer injection while
        // giving each role ample coordinate space for hash separation.
        Self::new(32)
    }
}

impl KgStore for InMemoryKgStore {
    fn lookup(&self, subject: TokenId, predicate: TokenId) -> Vec<TokenId> {
        match self.index.get(&(subject, predicate)) {
            Some(objects) => objects.clone(),
            None => Vec::new(),
        }
    }

    fn embed(&self, triple: KgTriple) -> Vec<f32> {
        let total_dim = self.role_dim.saturating_mul(3);
        if total_dim == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(total_dim);
        out.extend(token_vector(
            triple.subject,
            self.role_dim,
            SUBJECT_SEED_TAG,
        ));
        out.extend(token_vector(
            triple.predicate,
            self.role_dim,
            PREDICATE_SEED_TAG,
        ));
        out.extend(token_vector(triple.object, self.role_dim, OBJECT_SEED_TAG));
        out
    }

    fn embed_dim(&self) -> usize {
        self.role_dim.saturating_mul(3)
    }
}

// ── Role seed tags ─────────────────────────────────────────────
//
// The shared `token_vector` projection (see `crate::kg::projection`) is
// namespaced by a seed tag. The three tags below keep the subject,
// predicate, and object thirds of a triple embedding in statistically
// independent coordinate subspaces, so role confusion is impossible even
// if a downstream consumer sums the thirds together.

/// Seed tag for the subject third of a triple embedding.
const SUBJECT_SEED_TAG: &[u8] = b"s";

/// Seed tag for the predicate third of a triple embedding.
const PREDICATE_SEED_TAG: &[u8] = b"p";

/// Seed tag for the object third of a triple embedding.
const OBJECT_SEED_TAG: &[u8] = b"o";

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::KgStore;

    // ── Construction & basic shape ──

    #[test]
    fn test_new_is_empty() {
        let store = InMemoryKgStore::new(32);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.role_dim(), 32);
        assert_eq!(store.embed_dim(), 96);
    }

    #[test]
    fn test_default_role_dim_is_32() {
        let store = InMemoryKgStore::default();
        assert_eq!(store.role_dim(), 32);
        assert_eq!(store.embed_dim(), 96);
    }

    #[test]
    fn test_zero_role_dim_yields_empty_embeds() {
        // Degenerate but legal: embeddings are empty vectors.
        let store = InMemoryKgStore::new(0);
        assert_eq!(store.embed_dim(), 0);
        let v = store.embed(KgTriple::new(1, 2, 3));
        assert!(v.is_empty());
    }

    // ── Insertion & dedup ──

    #[test]
    fn test_insert_single_triple() {
        let mut store = InMemoryKgStore::new(8);
        let inserted = store.insert(KgTriple::new(1, 2, 3));
        assert!(inserted, "first insert must report true");
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_insert_duplicate_triple_is_noop() {
        let mut store = InMemoryKgStore::new(8);
        store.insert(KgTriple::new(1, 2, 3));
        let second = store.insert(KgTriple::new(1, 2, 3));
        assert!(!second, "duplicate insert must report false");
        assert_eq!(store.len(), 1, "duplicate must not grow the store");
    }

    #[test]
    fn test_insert_same_sp_different_objects_grows_bucket() {
        let mut store = InMemoryKgStore::new(8);
        store.insert(KgTriple::new(1, 2, 3));
        store.insert(KgTriple::new(1, 2, 4));
        assert_eq!(store.len(), 2);
        let objects = store.lookup(1, 2);
        assert_eq!(objects, vec![3, 4], "both objects kept in insertion order");
    }

    #[test]
    fn test_insert_role_swap_is_distinct_triple() {
        // (a, p, b) and (b, p, a) are distinct triples.
        let mut store = InMemoryKgStore::new(8);
        store.insert(KgTriple::new(1, 5, 2));
        store.insert(KgTriple::new(2, 5, 1));
        assert_eq!(store.len(), 2);
        assert_eq!(store.lookup(1, 5), vec![2]);
        assert_eq!(store.lookup(2, 5), vec![1]);
    }

    // ── Lookup semantics ──

    #[test]
    fn test_lookup_returns_empty_for_unknown_subject() {
        let mut store = InMemoryKgStore::new(8);
        store.insert(KgTriple::new(1, 2, 3));
        assert!(store.lookup(99, 2).is_empty());
    }

    #[test]
    fn test_lookup_returns_empty_for_unknown_predicate() {
        let mut store = InMemoryKgStore::new(8);
        store.insert(KgTriple::new(1, 2, 3));
        assert!(store.lookup(1, 99).is_empty());
    }

    #[test]
    fn test_lookup_returns_empty_for_empty_store() {
        let store = InMemoryKgStore::new(8);
        assert!(store.lookup(1, 2).is_empty());
    }

    #[test]
    fn test_lookup_preserves_insertion_order() {
        let mut store = InMemoryKgStore::new(8);
        for obj in [10u32, 20, 30, 40, 50] {
            store.insert(KgTriple::new(1, 2, obj));
        }
        assert_eq!(store.lookup(1, 2), vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_lookup_result_is_owned_copy() {
        // Mutating the returned Vec must not affect the store.
        let mut store = InMemoryKgStore::new(8);
        store.insert(KgTriple::new(1, 2, 3));
        let mut out = store.lookup(1, 2);
        out.clear();
        assert_eq!(
            store.lookup(1, 2),
            vec![3],
            "store unaffected by caller mutation"
        );
    }

    // ── Embedding: shape & determinism ──

    #[test]
    fn test_embed_dim_matches_role_dim_times_three() {
        let store = InMemoryKgStore::new(16);
        assert_eq!(store.embed_dim(), 48);
        let v = store.embed(KgTriple::new(1, 2, 3));
        assert_eq!(v.len(), 48);
    }

    #[test]
    fn test_embed_is_deterministic_same_triple() {
        let store = InMemoryKgStore::new(32);
        let a = store.embed(KgTriple::new(7, 8, 9));
        let b = store.embed(KgTriple::new(7, 8, 9));
        assert_eq!(a, b, "same triple must produce identical embeddings");
    }

    #[test]
    fn test_embed_deterministic_across_store_instances() {
        // Embeddings depend only on (triple, role_dim), not on store contents.
        let mut s1 = InMemoryKgStore::new(16);
        s1.insert(KgTriple::new(100, 200, 300));
        let s2 = InMemoryKgStore::new(16);

        let v1 = s1.embed(KgTriple::new(7, 8, 9));
        let v2 = s2.embed(KgTriple::new(7, 8, 9));
        assert_eq!(v1, v2, "embed must not depend on store contents");
    }

    #[test]
    fn test_embed_bounded_to_unit_range() {
        let store = InMemoryKgStore::new(32);
        let v = store.embed(KgTriple::new(42, 13, 7));
        for x in &v {
            assert!(
                (-1.0..=1.0).contains(x),
                "embedding coordinate {x} outside [-1, 1]"
            );
        }
    }

    // ── Embedding: role distinction ──

    #[test]
    fn test_embed_role_swap_produces_different_vectors() {
        // (a, p, b) vs (b, p, a) — same tokens, different roles.
        let store = InMemoryKgStore::new(32);
        let ab = store.embed(KgTriple::new(1, 5, 2));
        let ba = store.embed(KgTriple::new(2, 5, 1));
        assert_ne!(ab, ba, "role-swapped triples must embed differently");
    }

    #[test]
    fn test_embed_subject_block_changes_with_subject() {
        let store = InMemoryKgStore::new(32);
        let role_dim = store.role_dim();
        let v1 = store.embed(KgTriple::new(1, 2, 3));
        let v2 = store.embed(KgTriple::new(99, 2, 3));
        // First third (subject block) must differ; last two thirds must match.
        let (s1, rest1) = v1.split_at(role_dim);
        let (s2, rest2) = v2.split_at(role_dim);
        assert_ne!(s1, s2, "subject block must change with subject");
        assert_eq!(rest1, rest2, "predicate+object blocks must be unchanged");
    }

    #[test]
    fn test_embed_predicate_block_changes_with_predicate() {
        let store = InMemoryKgStore::new(32);
        let role_dim = store.role_dim();
        let v1 = store.embed(KgTriple::new(1, 2, 3));
        let v2 = store.embed(KgTriple::new(1, 99, 3));
        let (s1, mid1_o1) = v1.split_at(role_dim);
        let (s2, mid2_o2) = v2.split_at(role_dim);
        let (p1, o1) = mid1_o1.split_at(role_dim);
        let (p2, o2) = mid2_o2.split_at(role_dim);
        assert_eq!(s1, s2, "subject block unchanged");
        assert_ne!(p1, p2, "predicate block must change with predicate");
        assert_eq!(o1, o2, "object block unchanged");
    }

    #[test]
    fn test_embed_object_block_changes_with_object() {
        let store = InMemoryKgStore::new(32);
        let role_dim = store.role_dim();
        let v1 = store.embed(KgTriple::new(1, 2, 3));
        let v2 = store.embed(KgTriple::new(1, 2, 99));
        let (s1, rest1) = v1.split_at(role_dim);
        let (s2, rest2) = v2.split_at(role_dim);
        // subject blocks match; the difference is in the last third.
        assert_eq!(s1, s2);
        assert_ne!(rest1, rest2, "object block must change with object");
    }

    // ── Triples iterator ──

    #[test]
    fn test_triples_iterator_empty() {
        let store = InMemoryKgStore::new(8);
        assert_eq!(store.triples().count(), 0);
    }

    #[test]
    fn test_triples_iterator_preserves_insertion_order_and_dedup() {
        let mut store = InMemoryKgStore::new(8);
        let triples = [
            KgTriple::new(1, 2, 3),
            KgTriple::new(4, 5, 6),
            KgTriple::new(1, 2, 3), // duplicate
            KgTriple::new(7, 8, 9),
        ];
        for t in triples {
            store.insert(t);
        }
        let collected: Vec<KgTriple> = store.triples().copied().collect();
        assert_eq!(
            collected,
            vec![
                KgTriple::new(1, 2, 3),
                KgTriple::new(4, 5, 6),
                KgTriple::new(7, 8, 9),
            ]
        );
    }

    // ── Trait-object dispatch ──

    #[test]
    fn test_kg_store_trait_object_lookup() {
        let mut store = InMemoryKgStore::new(16);
        store.insert(KgTriple::new(1, 2, 3));
        let dyn_store: &dyn KgStore = &store;
        assert_eq!(dyn_store.lookup(1, 2), vec![3]);
        assert!(dyn_store.lookup(1, 99).is_empty());
    }

    #[test]
    fn test_kg_store_trait_object_embed() {
        let store = InMemoryKgStore::new(16);
        let dyn_store: &dyn KgStore = &store;
        let v = dyn_store.embed(KgTriple::new(1, 2, 3));
        assert_eq!(v.len(), dyn_store.embed_dim());
        assert_eq!(v.len(), 48);
    }

    // ── Larger workload ──

    #[test]
    fn test_insert_and_lookup_many_triples() {
        let mut store = InMemoryKgStore::new(8);
        // 50 subjects × 10 predicates × 5 objects = 2500 triples.
        for s in 0..50u32 {
            for p in 0..10u32 {
                for o in 0..5u32 {
                    store.insert(KgTriple::new(s, p, s * 100 + p * 10 + o));
                }
            }
        }
        assert_eq!(store.len(), 2500);

        // Spot-check a bucket.
        let objects = store.lookup(7, 3);
        assert_eq!(
            objects,
            vec![
                7 * 100 + 3 * 10,
                7 * 100 + 3 * 10 + 1,
                7 * 100 + 3 * 10 + 2,
                7 * 100 + 3 * 10 + 3,
                7 * 100 + 3 * 10 + 4
            ],
        );

        // Lookup of unknown (s, p) is empty even in a populated store.
        assert!(store.lookup(7, 99).is_empty());
        assert!(store.lookup(99, 0).is_empty());
    }

    // ── Send + Sync ──

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryKgStore>();
    }
}
