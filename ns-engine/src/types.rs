//! Decoupled type definitions for the neuro-symbolic engine.
//!
//! All structs and impls live here, separate from traits (`traits.rs`) and
//! logic (`decode.rs`, `pruners/`, `draft/`). This keeps the dependency graph
//! flat: traits depend on types, logic depends on both.

/// Token identifier in the vocabulary.
pub type TokenId = u32;

/// Log-probability distribution over the vocabulary (len = vocab_size).
pub type Logits = Vec<f32>;

/// Stable identifier for a bandit arm (pruner) within a [`BanditPruner`](crate::bandit::BanditPruner).
///
/// Assigned by arm insertion order. Used by the [`TrialLog`](crate::bandit::TrialLog) to
/// track per-arm statistics and by policies to report selections.
pub type ArmId = u32;

/// Configuration for the speculative decode loop.
#[derive(Clone, Debug)]
pub struct DecodeConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Number of top candidates to consider per step (draft beam width).
    pub top_k: usize,
    /// RNG seed for reproducible runs.
    pub seed: u64,
    /// Enable backtracking search on dead-ends (tree search mode).
    pub backtrack: bool,
    /// Maximum exploration attempts before giving up (backtrack mode only).
    pub max_attempts: u64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_tokens: 81,
            top_k: 9,
            seed: 42,
            backtrack: true,
            max_attempts: 100_000,
        }
    }
}

/// Result of a speculative decode run.
#[derive(Clone, Debug)]
pub struct DecodeResult {
    /// The generated token sequence.
    pub tokens: Vec<TokenId>,
    /// Whether the final sequence passed verification.
    pub verified: bool,
    /// Number of exploration attempts (forward steps + backtracks).
    pub attempts: u64,
    /// BLAKE3 hash of the token sequence for audit/reproducibility.
    pub hash: [u8; 32],
}

impl DecodeResult {
    /// Compute BLAKE3 hash from a token slice.
    ///
    /// Used for audit trails and verifying two decode runs produced
    /// identical output. Zero-allocation via streaming hasher.
    pub fn hash_tokens(tokens: &[TokenId]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for t in tokens {
            hasher.update(&t.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

// ── Knowledge-graph triple ─────────────────────────────────────

/// A knowledge-graph triple: `(subject, predicate, object)`.
///
/// All three slots are vocabulary token IDs, grounding the KG in the same
/// discrete space as the decode loop. Facts stored as triples can be
/// recalled verbatim (needle-in-haystack retrieval) and projected into a
/// latent vector via
/// [`KgStore::embed`](crate::traits::KgStore::embed) for mid-layer K/V
/// injection (Phase 4 `domain_latent`).
///
/// Canonical RDF form: `<subject> <predicate> <object>`.
/// Example: `(alice, knows, bob)` — entity, relation, entity.
///
/// The struct is `Copy` (12 bytes on 32-bit token IDs) and totally ordered,
/// so it can be stored in `BTreeSet`/`Vec` and compared cheaply without
/// allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KgTriple {
    /// Subject entity token ID.
    pub subject: TokenId,
    /// Predicate / relation token ID.
    pub predicate: TokenId,
    /// Object entity token ID.
    pub object: TokenId,
}

impl KgTriple {
    /// Create a new triple from its three token IDs.
    pub const fn new(subject: TokenId, predicate: TokenId, object: TokenId) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// BLAKE3 hash of the triple for deduplication and audit keys.
    ///
    /// Deterministic: same triple → same hash. The hash is order-sensitive
    /// across roles — `(a, p, b)` and `(b, p, a)` hash to different values,
    /// which is required for correct subject/object distinction.
    pub fn to_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.subject.to_le_bytes());
        hasher.update(&self.predicate.to_le_bytes());
        hasher.update(&self.object.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_config_default() {
        let config = DecodeConfig::default();
        assert_eq!(config.max_tokens, 81);
        assert_eq!(config.top_k, 9);
        assert!(config.backtrack);
    }

    #[test]
    fn test_hash_tokens_deterministic() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let hash_a = DecodeResult::hash_tokens(&tokens);
        let hash_b = DecodeResult::hash_tokens(&tokens);
        assert_eq!(hash_a, hash_b, "same tokens must produce same hash");
    }

    #[test]
    fn test_hash_tokens_differ_on_change() {
        let a = vec![1u32, 2, 3];
        let b = vec![1u32, 2, 4];
        let hash_a = DecodeResult::hash_tokens(&a);
        let hash_b = DecodeResult::hash_tokens(&b);
        assert_ne!(
            hash_a, hash_b,
            "different tokens must produce different hash"
        );
    }

    #[test]
    fn test_hash_tokens_empty() {
        let hash = DecodeResult::hash_tokens(&[]);
        // Empty input still produces a valid BLAKE3 hash (the zero-message hash)
        assert_ne!(hash, [0u8; 32], "BLAKE3 of empty input is not all zeros");
    }

    // ── KgTriple tests ──

    #[test]
    fn test_kg_triple_new_assigns_slots() {
        let t = KgTriple::new(1, 2, 3);
        assert_eq!(t.subject, 1);
        assert_eq!(t.predicate, 2);
        assert_eq!(t.object, 3);
    }

    #[test]
    fn test_kg_triple_field_constructor() {
        let t = KgTriple {
            subject: 7,
            predicate: 8,
            object: 9,
        };
        assert_eq!(t, KgTriple::new(7, 8, 9));
    }

    #[test]
    fn test_kg_triple_equality() {
        assert_eq!(KgTriple::new(1, 2, 3), KgTriple::new(1, 2, 3));
        assert_ne!(KgTriple::new(1, 2, 3), KgTriple::new(3, 2, 1));
    }

    #[test]
    fn test_kg_triple_copy() {
        let a = KgTriple::new(1, 2, 3);
        let b = a; // Copy — no move
        assert_eq!(a, b, "KgTriple is Copy, original must remain valid");
    }

    #[test]
    fn test_kg_triple_to_hash_deterministic() {
        let t = KgTriple::new(10, 20, 30);
        assert_eq!(t.to_hash(), t.to_hash(), "same triple → same hash");
    }

    #[test]
    fn test_kg_triple_to_hash_differs_on_role_swap() {
        // (a, p, b) and (b, p, a) must hash differently — role distinction.
        let ab = KgTriple::new(1, 5, 2);
        let ba = KgTriple::new(2, 5, 1);
        assert_ne!(
            ab.to_hash(),
            ba.to_hash(),
            "role-swapped triples must produce different hashes"
        );
    }

    #[test]
    fn test_kg_triple_to_hash_differs_on_predicate() {
        let a = KgTriple::new(1, 2, 3);
        let b = KgTriple::new(1, 9, 3);
        assert_ne!(
            a.to_hash(),
            b.to_hash(),
            "different predicates must produce different hashes"
        );
    }

    #[test]
    fn test_kg_triple_total_order() {
        // Required for BTreeSet<KgTriple> deterministic iteration.
        let mut v = vec![
            KgTriple::new(3, 1, 1),
            KgTriple::new(1, 1, 1),
            KgTriple::new(2, 2, 2),
            KgTriple::new(1, 1, 2),
        ];
        v.sort();
        assert_eq!(
            v,
            vec![
                KgTriple::new(1, 1, 1),
                KgTriple::new(1, 1, 2),
                KgTriple::new(2, 2, 2),
                KgTriple::new(3, 1, 1),
            ],
            "lexicographic subject→predicate→object ordering"
        );
    }

    #[test]
    fn test_kg_triple_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<KgTriple>();
    }
}
