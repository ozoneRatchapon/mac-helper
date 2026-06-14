//! KG-augmented decode step: inject KG-grounded candidates at a configured
//! decode midpoint.
//!
//! Interpretation of the Phase 4 "domain_latent mid-layer injection" checkbox:
//! the crate is modelless (no transformer layers, no attention weights), so
//! "mid-layer K/V injection" is realized as a KG-augmented decode step that
//! fires once at a configured decode step (the conceptual midpoint). No fake
//! transformer scaffold is introduced — the existing draft→prune→verify loop
//! is the integration target.
//!
//! # Mechanism
//!
//! When [`DomainLatent::fires_at`] returns true at the current decode step,
//! the decode loop calls [`ground`](DomainLatent::ground) with the context
//! generated so far. The method:
//!
//! 1. Takes the last context token as the subject entity.
//! 2. Looks up `kg.lookup(subject, predicate)` for the configured predicate.
//! 3. Computes a shard-space query embedding for the subject and key
//!    embeddings for each candidate object (via
//!    [`token_vector`](super::projection::token_vector) projected through
//!    [`ShardEmbedding`](super::ShardEmbedding)).
//! 4. Computes cosine similarity between query and each key, applies softmax,
//!    and returns `(object_token, weight)` pairs where weights sum to 1.0.
//!
//! The decode loop applies these weights as logit biases (via
//! [`apply_bias`](DomainLatent::apply_bias)) so the next token selection is
//! grounded in KG facts.
//!
//! # Modelless thesis
//!
//! Embedding similarity here is deterministic (BLAKE3-derived) and serves as
//! a reproducible tiebreaker among candidates — not as learned semantic
//! attention. The semantic content comes from the KG lookup (which objects
//! are known facts); the similarity ranking is a consistent, auditable
//! weighting that makes the same context always produce the same grounding.
//!
//! # Example
//!
//! ```
//! use ns_engine::kg::{DomainLatent, InMemoryKgStore, ShardEmbedding};
//! use ns_engine::types::KgTriple;
//!
//! let mut kg = InMemoryKgStore::new(32);
//! kg.insert(KgTriple::new(1, 10, 2)); // (alice, knows, bob)
//! kg.insert(KgTriple::new(1, 10, 3)); // (alice, knows, carol)
//!
//! let shard = ShardEmbedding::new(32, 16, 42);
//! let dl = DomainLatent::new(kg, shard, 4, 10);
//!
//! assert!(dl.fires_at(4));
//! assert!(!dl.fires_at(0));
//!
//! // With subject=1 (alice), predicate=10 (knows), get [bob, carol].
//! let groundings = dl.ground(&[1]);
//! assert_eq!(groundings.len(), 2);
//!
//! // Weights are non-negative and sum to ~1.0 (softmax).
//! let total: f32 = groundings.iter().map(|&(_, w)| w).sum();
//! assert!((total - 1.0).abs() < 1e-5);
//! ```

use crate::kg::projection::token_vector;
use crate::kg::ShardEmbedding;
use crate::traits::KgStore;
use crate::types::TokenId;

/// Seed tag for the query embedding (the subject entity being grounded).
///
/// Namespaces the subject's continuous representation apart from the
/// candidate objects (different seed tag -> statistically independent
/// vectors for the same token).
const QUERY_SEED_TAG: &[u8] = b"dl_query";

/// Seed tag for the key embeddings (the candidate object entities).
const OBJECT_SEED_TAG: &[u8] = b"dl_object";

// ── DomainLatent ───────────────────────────────────────────────

/// KG-augmented decode step that fires at a configured midpoint.
///
/// Constructed with a [`KgStore`], a [`ShardEmbedding`] (compact projection
/// space for similarity computation), an `inject_step` (the decode step at
/// which grounding fires), and a `predicate` (the KG relation to query).
///
/// At the configured step, the decode loop calls [`ground`](Self::ground)
/// with the current context. The last context token is treated as the
/// subject; KG facts `(subject, predicate, object)` are retrieved and their
/// objects returned as `(token, weight)` pairs. Weights come from softmax
/// over shard-space cosine similarities and sum to 1.0.
///
/// Generic over the [`KgStore`] implementation `S`. `DomainLatent<S>` is
/// `Send + Sync` whenever `S: Send + Sync` (which [`KgStore`] already
/// requires).
#[derive(Debug)]
pub struct DomainLatent<S: KgStore> {
    kg: S,
    shard: ShardEmbedding,
    inject_step: usize,
    predicate: TokenId,
}

/// Manual `Clone` with a conditional `S: Clone` bound.
///
/// `#[derive(Clone)]` would emit `impl<S: KgStore + Clone> Clone` implicitly,
/// but more importantly it would force the `Clone` bound to propagate through
/// every generic context that names `DomainLatent<S>` — an undue restriction
/// for callers that never clone. The manual impl makes cloning available
/// precisely when the underlying store is itself `Clone`.
impl<S: KgStore + Clone> Clone for DomainLatent<S> {
    fn clone(&self) -> Self {
        Self {
            kg: self.kg.clone(),
            shard: self.shard.clone(),
            inject_step: self.inject_step,
            predicate: self.predicate,
        }
    }
}

impl<S: KgStore> DomainLatent<S> {
    /// Construct a new KG-augmented decode step.
    ///
    /// - `kg`: the knowledge-graph store (owned).
    /// - `shard`: the compact projection used for query/key similarity.
    /// - `inject_step`: the decode step at which [`fires_at`](Self::fires_at)
    ///   returns true. Typically `max_tokens / 2` (see [`midpoint_step`](Self::midpoint_step)).
    /// - `predicate`: the KG relation to query for the subject.
    pub fn new(kg: S, shard: ShardEmbedding, inject_step: usize, predicate: TokenId) -> Self {
        Self {
            kg,
            shard,
            inject_step,
            predicate,
        }
    }

    /// Convenience: the midpoint decode step for a given `max_tokens`.
    ///
    /// Equals `max_tokens / 2`. Pass the result as `inject_step` in
    /// [`new`](Self::new) to fire grounding at the conceptual mid-layer
    /// (the modelless analog of "inject at `n_layer / 2`").
    pub fn midpoint_step(max_tokens: usize) -> usize {
        max_tokens / 2
    }

    /// Whether KG grounding should fire at `step`.
    ///
    /// Returns true only at the configured `inject_step`. The decode loop
    /// calls this at each step and invokes [`ground`](Self::ground) only when
    /// it returns true.
    pub fn fires_at(&self, step: usize) -> bool {
        step == self.inject_step
    }

    /// The configured injection step (for audit/diagnostics).
    pub fn inject_step(&self) -> usize {
        self.inject_step
    }

    /// The KG predicate queried for each subject (for audit/diagnostics).
    pub fn predicate(&self) -> TokenId {
        self.predicate
    }

    /// Ground the next-token prediction using KG facts.
    ///
    /// The last context token is the subject. Looks up all objects for
    /// `(subject, predicate)` in the KG, computes a shard-space cosine
    /// similarity between the subject's query embedding and each object's
    /// key embedding, applies softmax, and returns `(object_token, weight)`
    /// pairs.
    ///
    /// Returns an empty vec if:
    /// - the context is empty (no subject), or
    /// - the KG has no facts for `(subject, predicate)`.
    ///
    /// The returned weights are non-negative and sum to ~1.0 (softmax).
    /// Objects are returned in ascending token-ID order (deterministic).
    pub fn ground(&self, context: &[TokenId]) -> Vec<(TokenId, f32)> {
        let subject = match context.last() {
            Some(&s) => s,
            None => return Vec::new(),
        };

        let mut objects = self.kg.lookup(subject, self.predicate);
        if objects.is_empty() {
            return Vec::new();
        }
        objects.sort_unstable();
        objects.dedup();

        let shard_in = self.shard.input_dim();
        let query_raw = token_vector(subject, shard_in, QUERY_SEED_TAG);
        let query = self.shard.project(&query_raw);

        let sims: Vec<f32> = objects
            .iter()
            .map(|&obj| {
                let key_raw = token_vector(obj, shard_in, OBJECT_SEED_TAG);
                let key = self.shard.project(&key_raw);
                cosine_similarity(&query, &key)
            })
            .collect();

        let weights = softmax(&sims);

        objects.into_iter().zip(weights).collect()
    }

    /// Apply KG grounding as logit biases.
    ///
    /// For each `(token, weight)` from [`ground`](Self::ground), adds
    /// `weight * boost` to `logits[token]`. Tokens beyond `logits.len()`
    /// (out-of-vocab) are silently skipped — this is graceful degradation
    /// for vocab/KG size mismatches, not an error.
    ///
    /// The caller typically invokes this only when [`fires_at`](Self::fires_at)
    /// returns true, but calling it at any step is safe (it grounds on the
    /// current context regardless).
    pub fn apply_bias(&self, logits: &mut [f32], context: &[TokenId], boost: f32) {
        for (token, weight) in self.ground(context) {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] += weight * boost;
            }
        }
    }
}

// ── Private numerical helpers ──────────────────────────────────

/// Cosine similarity between two vectors.
///
/// Returns `a . b / (||a|| * ||b||)`, or `0.0` if either vector has zero
/// norm (degenerate — no meaningful direction). Output range: `[-1.0, 1.0]`.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Softmax over `values`: `exp(v_i - max) / sum(exp(v_j - max))`.
///
/// Returns a probability distribution (non-negative, sums to 1.0). The
/// max-subtraction trick prevents overflow. Returns empty for empty input.
/// Falls back to uniform if the denominator underflows to zero (measure-zero
/// for finite inputs).
fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / values.len() as f32; values.len()]
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::InMemoryKgStore;
    use crate::types::KgTriple;

    /// Build a test DomainLatent over an empty in-memory KG.
    fn build_fixture(
        shard_in: usize,
        shard_out: usize,
        inject_step: usize,
        predicate: TokenId,
    ) -> DomainLatent<InMemoryKgStore> {
        let kg = InMemoryKgStore::new(32);
        let shard = ShardEmbedding::new(shard_in, shard_out, 42);
        DomainLatent::new(kg, shard, inject_step, predicate)
    }

    // ── midpoint_step ──

    #[test]
    fn test_midpoint_step_halves_max_tokens() {
        assert_eq!(DomainLatent::<InMemoryKgStore>::midpoint_step(10), 5);
        assert_eq!(DomainLatent::<InMemoryKgStore>::midpoint_step(81), 40);
        assert_eq!(DomainLatent::<InMemoryKgStore>::midpoint_step(0), 0);
    }

    // ── fires_at ──

    #[test]
    fn test_fires_at_only_at_inject_step() {
        let dl = build_fixture(32, 16, 5, 10);
        assert!(!dl.fires_at(0));
        assert!(!dl.fires_at(4));
        assert!(dl.fires_at(5));
        assert!(!dl.fires_at(6));
    }

    #[test]
    fn test_fires_at_zero_step() {
        let dl = build_fixture(32, 16, 0, 10);
        assert!(dl.fires_at(0));
        assert!(!dl.fires_at(1));
    }

    #[test]
    fn test_accessors() {
        let dl = build_fixture(32, 16, 7, 42);
        assert_eq!(dl.inject_step(), 7);
        assert_eq!(dl.predicate(), 42);
    }

    // ── ground: empty / unknown ──

    #[test]
    fn test_ground_empty_context_returns_empty() {
        let dl = build_fixture(32, 16, 0, 10);
        let groundings = dl.ground(&[]);
        assert!(groundings.is_empty());
    }

    #[test]
    fn test_ground_unknown_subject_returns_empty() {
        let dl = build_fixture(32, 16, 0, 10);
        // Subject 999 is not in the empty KG.
        let groundings = dl.ground(&[999]);
        assert!(groundings.is_empty());
    }

    // ── ground: single fact ──

    #[test]
    fn test_ground_single_fact_single_object() {
        let mut kg = InMemoryKgStore::new(32);
        kg.insert(KgTriple::new(1, 10, 2)); // (alice, knows, bob)
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let groundings = dl.ground(&[1]);
        assert_eq!(groundings.len(), 1);
        match groundings.first() {
            Some(&(token, weight)) => {
                assert_eq!(token, 2);
                assert!(
                    (weight - 1.0).abs() < 1e-5,
                    "single object weight = {weight}, expected 1.0"
                );
            }
            None => panic!("expected one grounding"),
        }
    }

    // ── ground: multiple facts ──

    #[test]
    fn test_ground_multiple_facts_weights_sum_to_one() {
        let mut kg = InMemoryKgStore::new(32);
        kg.insert(KgTriple::new(1, 10, 2));
        kg.insert(KgTriple::new(1, 10, 3));
        kg.insert(KgTriple::new(1, 10, 4));
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let groundings = dl.ground(&[1]);
        assert_eq!(groundings.len(), 3);

        let total: f32 = groundings.iter().map(|&(_, w)| w).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "weights must sum to 1.0, got {total}"
        );

        for &(_, w) in &groundings {
            assert!(w >= 0.0, "weight {w} must be non-negative");
            assert!(w <= 1.0 + 1e-6, "weight {w} must not exceed 1.0");
        }
    }

    #[test]
    fn test_ground_returns_all_objects_in_token_order() {
        let mut kg = InMemoryKgStore::new(32);
        // Insert out of order — ground() should return sorted.
        kg.insert(KgTriple::new(1, 10, 5));
        kg.insert(KgTriple::new(1, 10, 2));
        kg.insert(KgTriple::new(1, 10, 8));
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let groundings = dl.ground(&[1]);
        let tokens: Vec<TokenId> = groundings.iter().map(|&(t, _)| t).collect();
        assert_eq!(tokens, vec![2, 5, 8], "objects must be in ascending order");
    }

    #[test]
    fn test_ground_dedups_duplicate_objects() {
        let mut kg = InMemoryKgStore::new(32);
        kg.insert(KgTriple::new(1, 10, 2));
        kg.insert(KgTriple::new(1, 10, 2)); // duplicate — store dedups
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let groundings = dl.ground(&[1]);
        assert_eq!(groundings.len(), 1, "duplicate inserts must be deduped");
    }

    // ── ground: determinism ──

    #[test]
    fn test_ground_is_deterministic() {
        let mut kg_a = InMemoryKgStore::new(32);
        kg_a.insert(KgTriple::new(1, 10, 2));
        kg_a.insert(KgTriple::new(1, 10, 3));
        let mut kg_b = InMemoryKgStore::new(32);
        kg_b.insert(KgTriple::new(1, 10, 2));
        kg_b.insert(KgTriple::new(1, 10, 3));

        let dl_a = DomainLatent::new(kg_a, ShardEmbedding::new(32, 16, 42), 0, 10);
        let dl_b = DomainLatent::new(kg_b, ShardEmbedding::new(32, 16, 42), 0, 10);

        let a = dl_a.ground(&[1]);
        let b = dl_b.ground(&[1]);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.0, pb.0, "token mismatch");
            let wa = pa.1;
            let wb = pb.1;
            assert!((wa - wb).abs() < 1e-6, "weight mismatch: {wa} vs {wb}");
        }
    }

    #[test]
    fn test_ground_uses_last_context_token_as_subject() {
        let mut kg = InMemoryKgStore::new(32);
        // Facts for subject 1 and subject 5.
        kg.insert(KgTriple::new(1, 10, 2));
        kg.insert(KgTriple::new(5, 10, 6));
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        // Last token is 5, so grounding should use subject=5, returning [6].
        let groundings = dl.ground(&[1, 2, 3, 4, 5]);
        assert_eq!(groundings.len(), 1);
        match groundings.first() {
            Some(&(token, _)) => assert_eq!(token, 6),
            None => panic!("expected grounding for subject 5"),
        }
    }

    // ── ground: predicate scoping ──

    #[test]
    fn test_ground_respects_predicate() {
        let mut kg = InMemoryKgStore::new(32);
        kg.insert(KgTriple::new(1, 10, 2)); // predicate 10
        kg.insert(KgTriple::new(1, 20, 3)); // predicate 20
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 20); // only predicate 20

        let groundings = dl.ground(&[1]);
        assert_eq!(groundings.len(), 1);
        match groundings.first() {
            Some(&(token, _)) => assert_eq!(token, 3),
            None => panic!("expected grounding for predicate 20"),
        }
    }

    // ── ground: edge case — degenerate shard (zero input or output) ──

    #[test]
    fn test_ground_with_degenerate_shard_is_uniform() {
        // Either zero input dim or zero output dim makes cosine similarities
        // all zero, softmax -> uniform. Grounding still returns all objects
        // with equal weight.
        for (shard_in, shard_out) in [(0usize, 16usize), (32, 0)] {
            let mut kg = InMemoryKgStore::new(32);
            kg.insert(KgTriple::new(1, 10, 2));
            kg.insert(KgTriple::new(1, 10, 3));
            kg.insert(KgTriple::new(1, 10, 4));
            let shard = ShardEmbedding::new(shard_in, shard_out, 42);
            let dl = DomainLatent::new(kg, shard, 0, 10);

            let groundings = dl.ground(&[1]);
            assert_eq!(groundings.len(), 3);
            let expected = 1.0 / 3.0;
            for &(token, weight) in &groundings {
                assert!(
                    (weight - expected).abs() < 1e-5,
                    "uniform weight for token {token} = {weight}, expected {expected} (shard {shard_in}->{shard_out})"
                );
            }
        }
    }

    // ── apply_bias ──

    #[test]
    fn test_apply_bias_boosts_grounded_tokens() {
        let mut kg = InMemoryKgStore::new(32);
        kg.insert(KgTriple::new(1, 10, 2));
        kg.insert(KgTriple::new(1, 10, 3));
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let mut logits = vec![0.0f32; 10];
        dl.apply_bias(&mut logits, &[1], 5.0);

        // Tokens 2 and 3 should have non-zero bias; everything else 0.
        let total_bias: f32 = logits.iter().sum();
        assert!(
            (total_bias - 5.0).abs() < 1e-4,
            "total bias = {total_bias}, expected ~5.0 (boost)"
        );
        for (i, &l) in logits.iter().enumerate() {
            match i {
                2 | 3 => assert!(l > 0.0, "grounded token {i} should have positive bias"),
                _ => assert!(
                    l == 0.0,
                    "non-grounded token {i} should have zero bias, got {l}"
                ),
            }
        }
    }

    #[test]
    fn test_apply_bias_single_object_gets_full_boost() {
        let mut kg = InMemoryKgStore::new(32);
        kg.insert(KgTriple::new(1, 10, 2));
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let mut logits = vec![0.0f32; 10];
        dl.apply_bias(&mut logits, &[1], 7.5);

        let bias_at_2 = logits[2];
        assert!(
            (bias_at_2 - 7.5).abs() < 1e-4,
            "single object bias = {bias_at_2}, expected 7.5"
        );
    }

    #[test]
    fn test_apply_bias_skips_out_of_vocab_tokens() {
        let mut kg = InMemoryKgStore::new(32);
        // Object token 100 is beyond the logits buffer (len=10).
        kg.insert(KgTriple::new(1, 10, 100));
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 0, 10);

        let mut logits = vec![0.0f32; 10];
        // Should not panic — token 100 is silently skipped.
        dl.apply_bias(&mut logits, &[1], 5.0);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l == 0.0, "no bias applied in vocab at {i}, got {l}");
        }
    }

    #[test]
    fn test_apply_bias_empty_context_is_noop() {
        let dl = build_fixture(32, 16, 0, 10);
        let mut logits = vec![0.5f32; 10];
        dl.apply_bias(&mut logits, &[], 5.0);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                (l - 0.5).abs() < 1e-6,
                "empty context must not modify logits[{i}] = {l}"
            );
        }
    }

    #[test]
    fn test_apply_bias_does_not_fire_on_unknown_subject() {
        let dl = build_fixture(32, 16, 0, 10);
        let mut logits = vec![0.5f32; 10];
        // Subject 999 is unknown; no facts -> no bias.
        dl.apply_bias(&mut logits, &[999], 5.0);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                (l - 0.5).abs() < 1e-6,
                "unknown subject must not modify logits[{i}] = {l}"
            );
        }
    }

    // ── cosine_similarity unit tests ──

    #[test]
    fn test_cosine_identical_vectors_is_one() {
        let v = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical vectors: cosine = {sim}"
        );
    }

    #[test]
    fn test_cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "orthogonal vectors: cosine = {sim}");
    }

    #[test]
    fn test_cosine_opposite_vectors_is_neg_one() {
        let a = vec![1.0f32, 1.0];
        let b = vec![-1.0f32, -1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-5,
            "opposite vectors: cosine = {sim}"
        );
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
    }

    // ── softmax unit tests ──

    #[test]
    fn test_softmax_empty_is_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn test_softmax_single_element_is_one() {
        let result = softmax(&[1.5f32]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let values = vec![1.0f32, 2.0, 3.0, -1.0, 0.5];
        let result = softmax(&values);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}, expected 1.0");
    }

    #[test]
    fn test_softmax_uniform_input_is_uniform() {
        let values = vec![5.0f32; 4];
        let result = softmax(&values);
        let expected = 0.25;
        for (i, &w) in result.iter().enumerate() {
            assert!(
                (w - expected).abs() < 1e-5,
                "uniform input weight {i} = {w}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_all_zero_input_is_uniform() {
        // All-zero input (e.g., from degenerate cosine similarities) -> uniform.
        let values = vec![0.0f32; 5];
        let result = softmax(&values);
        let expected = 0.2;
        for (i, &w) in result.iter().enumerate() {
            assert!(
                (w - expected).abs() < 1e-5,
                "all-zero input weight {i} = {w}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_higher_value_gets_higher_weight() {
        let values = vec![0.0f32, 10.0];
        let result = softmax(&values);
        let w0 = result[0];
        let w1 = result[1];
        assert!(
            w1 > w0,
            "larger value should get larger weight: {w0} vs {w1}"
        );
    }

    // ── Trait bounds ──

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DomainLatent<InMemoryKgStore>>();
    }

    /// Minimal Clone-able `KgStore` for testing `DomainLatent`'s `Clone` impl.
    ///
    /// `InMemoryKgStore` does not derive `Clone`, so this mock stands in to
    /// exercise the conditional `S: Clone` bound on `DomainLatent<S>`. It
    /// returns the same object set for any `(subject, predicate)` query,
    /// which is sufficient for verifying that cloning preserves grounding
    /// behavior.
    #[derive(Clone, Debug)]
    struct CloneKgStore {
        objects: Vec<TokenId>,
    }

    impl KgStore for CloneKgStore {
        fn lookup(&self, _subject: TokenId, _predicate: TokenId) -> Vec<TokenId> {
            self.objects.clone()
        }
        fn embed(&self, _triple: KgTriple) -> Vec<f32> {
            Vec::new()
        }
        fn embed_dim(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_clone_preserves_behavior() {
        let kg = CloneKgStore { objects: vec![2] };
        let shard = ShardEmbedding::new(32, 16, 42);
        let dl = DomainLatent::new(kg, shard, 3, 10);
        let dl_clone = dl.clone();

        let a = dl.ground(&[1]);
        let b = dl_clone.ground(&[1]);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.0, pb.0);
            let wa = pa.1;
            let wb = pb.1;
            assert!(
                (wa - wb).abs() < 1e-6,
                "cloned weight mismatch: {wa} vs {wb}"
            );
        }
    }
}
