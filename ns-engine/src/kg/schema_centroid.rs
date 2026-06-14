//! Per-class embedding centroids for structured entity initialization.
//!
//! When the KG initializes a new entity, its starting embedding should
//! reflect the entity's type — not random noise. [`SchemaCentroid`] provides
//! this: each class (a TokenId naming a type/category) has a centroid vector
//! computed as the element-wise mean of its members' deterministic token
//! embeddings. [`SchemaCentroid::init_embedding`] then returns the centroid
//! of an entity's assigned class (or the average across multiple classes),
//! grounding initialization in KG structure.
//!
//! # Cost
//!
//! Centroids are computed on demand in `O(members × dim)` with no caching.
//! This is fine for the entity-init use case (setup-time, called once per
//! entity), and avoids the invalidation bugs that a lazy cache would risk.
//! If a future checkbox needs hot-path centroid access, caching can be added
//! behind the same API.
//!
//! # Determinism
//!
//! Centroid vectors depend only on `(members, dim)` — not on insertion
//! order of *classes*, not on unrelated entities, not on process state.
//! Reordering `assign` calls for the same `(entity, class)` pairs produces
//! identical centroids because the member set is unchanged.
//!
//! # Concurrency
//!
//! `SchemaCentroid` is `Send + Sync`. Writes (`assign`) require `&mut self`;
//! reads (`centroid`, `init_embedding`, `members`, `classes_of`) take
//! `&self`.

use std::collections::HashMap;

use crate::kg::projection::token_vector;
use crate::types::TokenId;

/// Seed tag distinguishing entity embeddings produced here from the
/// role-scoped triple embeddings produced by
/// [`InMemoryKgStore`](super::InMemoryKgStore). Different seed tags yield
/// statistically independent vectors for the same token — see
/// [`token_vector`](crate::kg::projection::token_vector).
const ENTITY_SEED_TAG: &[u8] = b"entity";

/// Per-class embedding centroid store for structured entity initialization.
///
/// Each class (a TokenId naming a type/category) has a set of member
/// entities (also TokenIds). The [`centroid`](Self::centroid) of a class is
/// the element-wise mean of its members' deterministic token embeddings.
/// [`init_embedding`](Self::init_embedding) returns the centroid of an
/// entity's assigned class — or, for multi-class entities, the mean of those
/// centroids — grounding initialization in KG structure rather than random
/// noise.
///
/// # Example
///
/// ```
/// use ns_engine::kg::SchemaCentroid;
///
/// let mut sc = SchemaCentroid::new(16);
/// sc.assign(1, 100); // entity 1 ∈ class 100 (person)
/// sc.assign(2, 100); // entity 2 ∈ class 100 (person)
/// sc.assign(3, 200); // entity 3 ∈ class 200 (place)
///
/// // Class 100 centroid = mean of entity 1 and entity 2 vectors.
/// match sc.centroid(100) {
///     Some(c) => assert_eq!(c.len(), 16),
///     None => panic!("class 100 must have a centroid"),
/// }
///
/// // Init for a known entity returns its class centroid.
/// let init1 = sc.init_embedding(1);
/// assert_eq!(init1.len(), 16);
///
/// // Init for an unseen entity returns its raw token vector.
/// let init9 = sc.init_embedding(9);
/// assert_eq!(init9.len(), 16);
/// ```
pub struct SchemaCentroid {
    /// `class → member entities` (deduped, insertion-ordered).
    ///
    /// Drives [`centroid`](Self::centroid). Insertion order is preserved so
    /// that iteration over a class's members is stable and reproducible.
    class_members: HashMap<TokenId, Vec<TokenId>>,
    /// `entity → classes` (deduped, insertion-ordered). Reverse index.
    ///
    /// Drives [`init_embedding`](Self::init_embedding). Maintained in lockstep
    /// with `class_members` by [`assign`](Self::assign).
    entity_classes: HashMap<TokenId, Vec<TokenId>>,
    /// Per-token embedding dimension.
    ///
    /// Every centroid and entity vector has exactly this many coordinates.
    /// `dim = 0` yields empty vectors (degenerate but legal).
    dim: usize,
}

impl SchemaCentroid {
    /// Create an empty centroid store with the given per-token dimension.
    ///
    /// `dim = 0` is permitted: all centroids and entity vectors will be
    /// empty. This lets callers structurally disable embeddings without
    /// restructuring call sites.
    pub fn new(dim: usize) -> Self {
        Self {
            class_members: HashMap::new(),
            entity_classes: HashMap::new(),
            dim,
        }
    }

    /// Per-token embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Assign `entity` to `class`.
    ///
    /// Idempotent: re-assigning an existing `(entity, class)` pair is a
    /// no-op on both indices. Returns `true` if the pair was newly created.
    ///
    /// Both indices are updated atomically — `class_members[class]` and
    /// `entity_classes[entity]` are kept in lockstep so reads in either
    /// direction stay consistent.
    pub fn assign(&mut self, entity: TokenId, class: TokenId) -> bool {
        let added_to_class = push_dedup(&mut self.class_members, class, entity);
        let added_to_entity = push_dedup(&mut self.entity_classes, entity, class);
        // The two indices track the same relation, so a new pair appears in
        // both or neither. Defensive: OR them so a partial update is still
        // treated as a change.
        added_to_class || added_to_entity
    }

    /// Members of `class`, in insertion order.
    ///
    /// Returns an empty slice for an unknown class.
    pub fn members(&self, class: TokenId) -> &[TokenId] {
        match self.class_members.get(&class) {
            Some(members) => members.as_slice(),
            None => &[],
        }
    }

    /// Classes that `entity` belongs to, in insertion order.
    ///
    /// Returns an empty slice for an entity assigned to no class.
    pub fn classes_of(&self, entity: TokenId) -> &[TokenId] {
        match self.entity_classes.get(&entity) {
            Some(classes) => classes.as_slice(),
            None => &[],
        }
    }

    /// Number of distinct classes.
    pub fn class_count(&self) -> usize {
        self.class_members.len()
    }

    /// Number of distinct entities tracked across all classes.
    pub fn entity_count(&self) -> usize {
        self.entity_classes.len()
    }

    /// Deterministic per-token embedding for `entity`.
    ///
    /// This is the raw token projection (BLAKE3-seeded, no class info). It
    /// is the fallback used by [`init_embedding`](Self::init_embedding) when
    /// an entity belongs to no class.
    pub fn entity_vector(&self, entity: TokenId) -> Vec<f32> {
        token_vector(entity, self.dim, ENTITY_SEED_TAG)
    }

    /// Element-wise mean of the member embeddings of `class`.
    ///
    /// Returns `None` for an unknown class (not in the index) or a class
    /// with no members. Returns `Some(empty)` when `dim == 0`.
    ///
    /// Cost: `O(members × dim)`. No caching — see
    /// [module docs](self#cost) for rationale.
    pub fn centroid(&self, class: TokenId) -> Option<Vec<f32>> {
        let members = self.class_members.get(&class)?;
        // `assign` keeps members non-empty for any present class, but
        // defensive: an empty bucket has no defined centroid.
        if members.is_empty() {
            return None;
        }
        let mut acc = vec![0.0f32; self.dim];
        for entity in members {
            // Embed inline rather than via entity_vector() to avoid a
            // separate allocation per member; token_vector is the hot path.
            let v = token_vector(*entity, self.dim, ENTITY_SEED_TAG);
            for (a, x) in acc.iter_mut().zip(v.iter()) {
                *a += *x;
            }
        }
        let n = members.len() as f32;
        for a in acc.iter_mut() {
            *a /= n;
        }
        Some(acc)
    }

    /// Structured initialization vector for `entity`.
    ///
    /// Semantics:
    /// - **No classes** → the entity's raw
    ///   [`entity_vector`](Self::entity_vector) (no schema info available).
    /// - **Exactly one class** → that class's
    ///   [`centroid`](Self::centroid).
    /// - **Multiple classes** → the element-wise mean of those classes'
    ///   centroids (multi-type init, e.g. an entity that is both `person`
    ///   and `employee`).
    ///
    /// The returned vector always has length [`dim`](Self::dim).
    pub fn init_embedding(&self, entity: TokenId) -> Vec<f32> {
        let classes = match self.entity_classes.get(&entity) {
            Some(classes) => classes.as_slice(),
            None => return self.entity_vector(entity),
        };
        match classes.len() {
            0 => self.entity_vector(entity),
            1 => match self.centroid(classes[0]) {
                Some(c) => c,
                // Unreachable: classes[0] is non-empty by `assign` invariant,
                // but guard rather than unwrap.
                None => self.entity_vector(entity),
            },
            _ => {
                let mut acc = vec![0.0f32; self.dim];
                let mut count: f32 = 0.0;
                for class in classes {
                    match self.centroid(*class) {
                        Some(c) => {
                            for (a, x) in acc.iter_mut().zip(c.iter()) {
                                *a += *x;
                            }
                            count += 1.0;
                        }
                        None => continue,
                    }
                }
                match count > 0.0 {
                    true => {
                        for a in acc.iter_mut() {
                            *a /= count;
                        }
                        acc
                    }
                    // All centroids were None (defensive): fall back to raw.
                    false => self.entity_vector(entity),
                }
            }
        }
    }
}

impl Default for SchemaCentroid {
    fn default() -> Self {
        // 16 dims per entity: small enough for cheap init, large enough for
        // hash separation across a realistic schema. Centroid averaging
        // preserves the [-1, 1] range, so this stays well-scaled.
        Self::new(16)
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Push `value` into `index[key]`, deduped.
///
/// Creates the bucket if absent. Returns `true` if `value` was newly added,
/// `false` if it was already present. Centralizes the dedup-on-insert pattern
/// shared by both indices of [`SchemaCentroid`].
fn push_dedup(index: &mut HashMap<TokenId, Vec<TokenId>>, key: TokenId, value: TokenId) -> bool {
    let bucket = index.entry(key).or_default();
    if bucket.contains(&value) {
        return false;
    }
    bucket.push(value);
    true
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction & shape ──

    #[test]
    fn test_new_is_empty() {
        let sc = SchemaCentroid::new(16);
        assert_eq!(sc.dim(), 16);
        assert_eq!(sc.class_count(), 0);
        assert_eq!(sc.entity_count(), 0);
        assert!(sc.members(100).is_empty());
        assert!(sc.classes_of(1).is_empty());
    }

    #[test]
    fn test_default_dim_is_16() {
        let sc = SchemaCentroid::default();
        assert_eq!(sc.dim(), 16);
    }

    #[test]
    fn test_zero_dim_yields_empty_vectors() {
        // dim=0 is degenerate but legal: centroids and init vectors are empty.
        let mut sc = SchemaCentroid::new(0);
        sc.assign(1, 100);
        match sc.centroid(100) {
            Some(c) => assert!(c.is_empty(), "dim=0 centroid must be empty"),
            None => panic!("known class must return Some even at dim=0"),
        }
        assert!(sc.init_embedding(1).is_empty());
        assert!(sc.entity_vector(1).is_empty());
    }

    // ── assign / indices ──

    #[test]
    fn test_assign_single_pair() {
        let mut sc = SchemaCentroid::new(8);
        let added = sc.assign(1, 100);
        assert!(added, "first assign must report true");
        assert_eq!(sc.class_count(), 1);
        assert_eq!(sc.entity_count(), 1);
        assert_eq!(sc.members(100), &[1]);
        assert_eq!(sc.classes_of(1), &[100]);
    }

    #[test]
    fn test_assign_duplicate_is_noop() {
        let mut sc = SchemaCentroid::new(8);
        sc.assign(1, 100);
        let second = sc.assign(1, 100);
        assert!(!second, "duplicate assign must report false");
        assert_eq!(sc.members(100), &[1], "no duplicate member");
        assert_eq!(sc.classes_of(1), &[100], "no duplicate class");
    }

    #[test]
    fn test_assign_multiple_entities_to_same_class() {
        let mut sc = SchemaCentroid::new(8);
        sc.assign(1, 100);
        sc.assign(2, 100);
        sc.assign(3, 100);
        assert_eq!(sc.members(100), &[1, 2, 3]);
        assert_eq!(sc.class_count(), 1);
        assert_eq!(sc.entity_count(), 3);
    }

    #[test]
    fn test_assign_entity_to_multiple_classes() {
        let mut sc = SchemaCentroid::new(8);
        sc.assign(1, 100);
        sc.assign(1, 200);
        sc.assign(1, 300);
        assert_eq!(sc.classes_of(1), &[100, 200, 300]);
        assert_eq!(sc.entity_count(), 1);
        assert_eq!(sc.class_count(), 3);
    }

    #[test]
    fn test_assign_preserves_insertion_order_in_both_indices() {
        let mut sc = SchemaCentroid::new(8);
        // Interleaved assigns to stress ordering.
        sc.assign(2, 200);
        sc.assign(1, 100);
        sc.assign(3, 200);
        sc.assign(1, 200);
        assert_eq!(sc.members(200), &[2, 3, 1], "class index order");
        assert_eq!(sc.classes_of(1), &[100, 200], "entity index order");
    }

    // ── centroid ──

    #[test]
    fn test_centroid_unknown_class_is_none() {
        let sc = SchemaCentroid::new(8);
        assert!(sc.centroid(999).is_none());
    }

    #[test]
    fn test_centroid_single_member_equals_entity_vector() {
        let mut sc = SchemaCentroid::new(8);
        sc.assign(7, 100);

        let centroid = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some"),
        };
        let raw = sc.entity_vector(7);
        assert_eq!(
            centroid, raw,
            "single-member centroid must equal that member's vector"
        );
    }

    #[test]
    fn test_centroid_multiple_members_is_mean() {
        let mut sc = SchemaCentroid::new(8);
        sc.assign(1, 100);
        sc.assign(2, 100);

        let v1 = sc.entity_vector(1);
        let v2 = sc.entity_vector(2);
        let expected: Vec<f32> = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();

        let centroid = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some"),
        };
        assert_eq!(centroid.len(), expected.len());
        for (got, want) in centroid.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "centroid coord {got} != mean {want}"
            );
        }
    }

    #[test]
    fn test_centroid_is_deterministic() {
        let mut sc = SchemaCentroid::new(16);
        sc.assign(1, 100);
        sc.assign(2, 100);
        sc.assign(3, 100);

        let a = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some"),
        };
        let b = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some"),
        };
        assert_eq!(a, b, "centroid must be stable across calls");
    }

    #[test]
    fn test_centroid_bounded_to_unit_range() {
        // Mean of vectors each in [-1, 1] stays in [-1, 1].
        let mut sc = SchemaCentroid::new(32);
        for entity in 0..20u32 {
            sc.assign(entity, 100);
        }
        let c = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some"),
        };
        for x in &c {
            assert!(
                (-1.0..=1.0).contains(x),
                "centroid coord {x} outside [-1, 1]"
            );
        }
    }

    #[test]
    fn test_centroid_distinct_classes_distinct_vectors() {
        let mut sc = SchemaCentroid::new(16);
        sc.assign(1, 100);
        sc.assign(2, 100);
        sc.assign(10, 200);
        sc.assign(11, 200);
        sc.assign(12, 200);

        let c100 = sc.centroid(100);
        let c200 = sc.centroid(200);
        match (c100, c200) {
            (Some(a), Some(b)) => assert_ne!(a, b, "distinct classes should differ"),
            _ => panic!("both classes should have centroids"),
        }
    }

    #[test]
    fn test_centroid_independent_of_unrelated_entities() {
        // Adding entities to a *different* class must not change class 100's
        // centroid — the centroid depends only on class 100's members.
        let mut sc = SchemaCentroid::new(16);
        sc.assign(1, 100);
        sc.assign(2, 100);
        let before = sc.centroid(100);

        sc.assign(99, 200);
        sc.assign(98, 200);

        let after = sc.centroid(100);
        assert_eq!(
            before, after,
            "centroid must be independent of other classes"
        );
    }

    // ── init_embedding ──

    #[test]
    fn test_init_no_class_returns_raw_token_vector() {
        let sc = SchemaCentroid::new(16);
        let init = sc.init_embedding(42);
        let raw = sc.entity_vector(42);
        assert_eq!(init, raw, "no-class init must equal raw token vector");
    }

    #[test]
    fn test_init_unknown_entity_returns_raw() {
        let mut sc = SchemaCentroid::new(16);
        // Populate so the store is non-empty; the queried entity is unseen.
        sc.assign(1, 100);
        let init = sc.init_embedding(99);
        let raw = sc.entity_vector(99);
        assert_eq!(init, raw);
    }

    #[test]
    fn test_init_single_class_equals_class_centroid() {
        let mut sc = SchemaCentroid::new(16);
        sc.assign(1, 100);
        sc.assign(2, 100);

        let centroid = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some"),
        };
        let init1 = sc.init_embedding(1);
        let init2 = sc.init_embedding(2);
        assert_eq!(init1, centroid, "entity 1 inits to its class centroid");
        assert_eq!(init2, centroid, "entity 2 inits to its class centroid");
    }

    #[test]
    fn test_init_multi_class_is_centroid_average() {
        let mut sc = SchemaCentroid::new(16);
        sc.assign(1, 100);
        sc.assign(2, 100);
        sc.assign(1, 200);
        sc.assign(3, 200);

        let c100 = match sc.centroid(100) {
            Some(c) => c,
            None => panic!("expected Some for 100"),
        };
        let c200 = match sc.centroid(200) {
            Some(c) => c,
            None => panic!("expected Some for 200"),
        };
        let expected: Vec<f32> = c100
            .iter()
            .zip(c200.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();

        let init = sc.init_embedding(1);
        assert_eq!(init.len(), expected.len());
        for (got, want) in init.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "multi-class init coord {got} != centroid avg {want}"
            );
        }
    }

    #[test]
    fn test_init_dim_is_already_correct() {
        let mut sc = SchemaCentroid::new(13);
        sc.assign(1, 100);
        assert_eq!(sc.init_embedding(1).len(), 13);
        assert_eq!(sc.init_embedding(99).len(), 13);
        assert_eq!(sc.entity_vector(7).len(), 13);
    }

    // ── entity_vector ──

    #[test]
    fn test_entity_vector_deterministic() {
        let sc = SchemaCentroid::new(16);
        assert_eq!(
            sc.entity_vector(5),
            sc.entity_vector(5),
            "same token → same vector"
        );
    }

    #[test]
    fn test_entity_vector_differs_for_distinct_tokens() {
        let sc = SchemaCentroid::new(16);
        assert_ne!(
            sc.entity_vector(5),
            sc.entity_vector(6),
            "distinct tokens should produce distinct vectors"
        );
    }

    #[test]
    fn test_entity_vector_bounded() {
        let sc = SchemaCentroid::new(16);
        let v = sc.entity_vector(123);
        for x in &v {
            assert!((-1.0..=1.0).contains(x), "coord {x} outside [-1,1]");
        }
    }

    #[test]
    fn test_entity_vector_independent_of_membership() {
        // Adding membership does not change the raw token vector.
        let mut sc = SchemaCentroid::new(16);
        let before = sc.entity_vector(1);
        sc.assign(1, 100);
        sc.assign(1, 200);
        let after = sc.entity_vector(1);
        assert_eq!(before, after, "raw token vector must not depend on schema");
    }

    // ── push_dedup helper ──

    #[test]
    fn test_push_dedup_first_insert() {
        let mut idx: HashMap<TokenId, Vec<TokenId>> = HashMap::new();
        assert!(push_dedup(&mut idx, 1, 10));
        assert!(push_dedup(&mut idx, 1, 20));
        assert_eq!(idx.get(&1), Some(&vec![10, 20]));
    }

    #[test]
    fn test_push_dedup_duplicate_is_noop() {
        let mut idx: HashMap<TokenId, Vec<TokenId>> = HashMap::new();
        push_dedup(&mut idx, 1, 10);
        let second = push_dedup(&mut idx, 1, 10);
        assert!(!second);
        assert_eq!(idx.get(&1), Some(&vec![10]));
    }

    #[test]
    fn test_push_dedup_creates_bucket_for_new_key() {
        let mut idx: HashMap<TokenId, Vec<TokenId>> = HashMap::new();
        push_dedup(&mut idx, 5, 99);
        assert!(idx.contains_key(&5));
        assert!(!idx.contains_key(&6));
    }

    // ── Send + Sync ──

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SchemaCentroid>();
    }

    // ── Larger workload ──

    #[test]
    fn test_many_classes_many_entities_consistency() {
        // 20 classes × 50 entities = 1000 assignments. Verify both indices
        // stay consistent and centroid dims are correct.
        let mut sc = SchemaCentroid::new(8);
        for class in 0..20u32 {
            for entity in (class * 50)..(class * 50 + 50) {
                sc.assign(entity, class);
            }
        }
        assert_eq!(sc.class_count(), 20);
        assert_eq!(sc.entity_count(), 1000);

        // Each class has exactly 50 members.
        for class in 0..20u32 {
            assert_eq!(sc.members(class).len(), 50, "class {class} member count");
        }

        // Each entity belongs to exactly one class.
        for entity in 0..1000u32 {
            assert_eq!(sc.classes_of(entity).len(), 1, "entity {entity} classes");
        }

        // Centroid of every class has dim 8.
        for class in 0..20u32 {
            match sc.centroid(class) {
                Some(c) => assert_eq!(c.len(), 8),
                None => panic!("class {class} should have a centroid"),
            }
        }
    }
}
