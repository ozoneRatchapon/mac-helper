//! Core traits for the neuro-symbolic engine.
//!
//! Mirrors the constraint-pruner architecture from katopz/katgpt-rs:
//! the draft model provides fluency, the ConstraintPruner provides
//! correctness, and the speculative decode loop connects them.
//!
//! The "modelless" thesis: intelligence lives in the symbolic pruner
//! layer, not in model weights. The model only drafts; the pruner decides.

use crate::types::{ArmId, KgTriple, Logits, TokenId};

// ── ConstraintPruner ───────────────────────────────────────────

/// Hard structural validity pruner for speculative decoding branches.
///
/// The Deterministic Validator concept: before the target model verifies
/// drafted branches, a rules engine prunes invalid ones. This prevents the
/// decode tree from wasting budget on branches that can never be accepted.
///
/// Without pruner: decode explores ALL high-probability tokens.
/// With pruner:    decode explores only VALID high-probability tokens.
///
/// Ownership boundary: ConstraintPruner owns HARD structural validity
/// (syntax, rules, constraints). It returns `bool` — valid or not.
/// Graded semantic relevance belongs in a separate ScreeningPruner
/// (Phase 2).
pub trait ConstraintPruner: Send + Sync {
    /// Check if `token` at the given `depth` is valid, given the tokens
    /// placed at earlier depths in this path.
    ///
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    ///
    /// Returns `false` to prune (reject) this branch.
    fn is_valid(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool;

    /// Validate multiple token candidates at the same depth in a single call.
    ///
    /// Writes results into `results`:
    /// `results[i] = is_valid(depth, candidates[i], parent_tokens)`.
    ///
    /// Implementations can override this to amortize lock acquisition and
    /// setup costs across all candidates (e.g., single WASM fuel reset).
    ///
    /// Default implementation calls `is_valid` per-item.
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        for i in 0..len {
            results[i] = self.is_valid(depth, candidates[i], parent_tokens);
        }
    }

    /// Soft validity score: how close is this token to the constraint boundary?
    ///
    /// Returns `1.0` for valid, `0.0` for invalid by default.
    /// Override for soft scoring (e.g., ManifoldE point-to-manifold distance,
    /// graduated relevance). The score is blended into log-probability space:
    /// - `1.0` = perfect match, no penalty (`ln(1.0) = 0.0`)
    /// - `0.5` = mediocre match, soft penalty (`ln(0.5) ≈ -0.69`)
    /// - `0.0` = hard rejection / trim (`ln(0.0) = -∞`)
    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        match self.is_valid(depth, token, parent_tokens) {
            true => 1.0,
            false => 0.0,
        }
    }

    /// Propagate semantic state with a newly committed token.
    ///
    /// Called after a token is accepted into the sequence. Stateful pruners
    /// use this to update internal structures (e.g., incrementing a row
    /// bitset in Sudoku). Stateless pruners derive everything from
    /// `parent_tokens` and leave this as a no-op.
    ///
    /// Default: no-op (stateless derivation).
    fn propagate(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {}

    /// Called when the decode loop backtracks from `depth` (pops the token).
    ///
    /// `token` is the token being removed from the sequence. `parent_tokens`
    /// is the remaining prefix AFTER the pop (i.e., tokens `[0..depth)`).
    ///
    /// Stateful pruners use this to undo any state changes from the matching
    /// [`propagate`](Self::propagate) call. Bandit pruners use this to assign
    /// negative reward: the arm that fired at this depth led to a dead-end
    /// deeper in the search.
    ///
    /// Default: no-op.
    fn on_backtrack(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {}
}

// ── ScreeningPruner ────────────────────────────────────────────

/// Graded relevance screening pruner — subsumes [`ConstraintPruner`].
///
/// Where ConstraintPruner answers binary validity, ScreeningPruner adds a
/// graded `f32 ∈ [0.0, 1.0]` relevance score per candidate. This is the
/// "Screening Is Enough" layer from katopz/katgpt-rs: bandits assign reward
/// based on screen scores, learning which pruner arm fires best per context.
///
/// Every ScreeningPruner IS-A ConstraintPruner (via supertrait bound), so it
/// can be used anywhere a ConstraintPruner is expected — but the extra
/// metadata (`arm_id`, `arm_label`, `screen`) makes it eligible as a bandit
/// arm.
///
/// # Implementors
///
/// - [`SudokuPruner`](crate::pruners::SudokuPruner) — binary screen (1.0/0.0)
/// - [`NoPruner`](crate::pruners::NoPruner) — permissive baseline (1.0)
/// - [`NgramScreeningPruner`](crate::pruners::NgramScreeningPruner) — n-gram prob as score
/// - [`RegexPruner`](crate::pruners::RegexPruner) — regex prefix validity
/// - [`JsonSchemaPruner`](crate::pruners::JsonSchemaPruner) — JSON structural prefix validity
/// - [`BomberActionPruner`](crate::pruners::BomberActionPruner) — Bomberman action legality (graded heuristic)
pub trait ScreeningPruner: ConstraintPruner {
    /// Stable identifier for this arm within a [`BanditPruner`](crate::bandit::BanditPruner).
    ///
    /// Used for diagnostics and stable cross-pattern attribution. Not used as
    /// the trial-log key — the log keys arms by their Vec index in the bandit.
    fn arm_id(&self) -> ArmId;

    /// Human-readable label for diagnostics and logging.
    fn arm_label(&self) -> &str;

    /// Graded relevance score for `token` at this context.
    ///
    /// Returns `f32 ∈ [0.0, 1.0]`:
    /// - `1.0` = perfect match, no penalty
    /// - `0.5` = mediocre match, soft penalty
    /// - `0.0` = hard rejection (equivalent to `is_valid == false`)
    ///
    /// Default: delegates to [`manifold_score`](ConstraintPruner::manifold_score).
    /// Override for domain-specific graded scoring.
    fn screen(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        self.manifold_score(depth, token, parent_tokens)
    }

    /// Batch version of [`screen`](Self::screen).
    ///
    /// Writes `results[i] = screen(depth, candidates[i], parent_tokens)`.
    /// Implementations can override to amortize setup costs across candidates.
    ///
    /// Default: per-item delegation.
    fn batch_screen(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [f32],
    ) {
        let len = candidates.len().min(results.len());
        for i in 0..len {
            results[i] = self.screen(depth, candidates[i], parent_tokens);
        }
    }
}

// ── DraftModel ─────────────────────────────────────────────────

/// Draft model interface — produces log-probability distributions over
/// the vocabulary.
///
/// The draft model is responsible for FLUENCY and DIVERSITY (broad
/// candidate generation). Correctness is enforced by the ConstraintPruner.
/// This is the "modelless" thesis: a tiny or even uninformative draft
/// suffices when the pruner carries the domain logic.
///
/// Implementors:
/// - [`UniformDraftModel`](crate::draft::UniformDraftModel) — uninformative
///   prior, all tokens equal. The pruner alone decides.
/// - [`NgramDraftModel`](crate::draft::NgramDraftModel) — real statistical
///   n-gram model with Laplace smoothing.
pub trait DraftModel: Send + Sync {
    /// Size of the token vocabulary.
    fn vocab_size(&self) -> usize;

    /// Compute log-probabilities for the next token given the context.
    ///
    /// `context` = tokens generated so far (may be empty for the first token).
    /// Returns a vector of length `vocab_size()` where higher values = more
    /// likely. Values need not be normalized log-probabilities; any ordering
    /// signal works because the decode loop only uses relative ranking.
    fn log_probs(&self, context: &[TokenId]) -> Logits;
}

// ── KgStore ────────────────────────────────────────────────────

/// Knowledge-graph store: structured, verifiable memory layer.
///
/// Where [`ConstraintPruner`] owns syntactic validity and [`DraftModel`]
/// owns fluency, the `KgStore` owns SEMANTIC GROUNDING. Facts are stored as
/// discrete triples ([`KgTriple`]) and recalled verbatim — no hallucinated
/// recall, no embedding drift.
///
/// This is the "needle in a haystack" (NIAH) substrate: a stored fact
/// `(s, p, o)` is retrieved exactly by `lookup(s, p) = [o, ...]`, with
/// 100% precision. The continuous projection ([`embed`](Self::embed)) feeds
/// the mid-layer K/V injection planned for the `domain_latent` mid-layer
/// (Phase 4).
///
/// Ownership boundary: KgStore owns STRUCTURED FACTUAL RECALL. It does not
/// decide fluency (draft model) or syntactic validity (pruner); it grounds
/// generation in discrete, checkable facts.
pub trait KgStore: Send + Sync {
    /// Retrieve all object tokens matching `(subject, predicate)`.
    ///
    /// Returns every known `o` such that `(subject, predicate, o)` is a
    /// stored fact. An empty result means no stored fact for that
    /// `(subject, predicate)` pair — absent, not "unknown".
    ///
    /// Order is implementation-defined; the default in-memory store
    /// preserves insertion order. Callers requiring sorted output should
    /// sort the result themselves.
    fn lookup(&self, subject: TokenId, predicate: TokenId) -> Vec<TokenId>;

    /// Project a triple into a deterministic latent vector.
    ///
    /// The KG Latent projection maps the discrete triple into a continuous
    /// space of dimension [`embed_dim`](Self::embed_dim), suitable for
    /// mid-layer K/V injection. Deterministic: same triple → same vector,
    /// across calls and across runs (no learned weights, no RNG).
    fn embed(&self, triple: KgTriple) -> Vec<f32>;

    /// Dimensionality of vectors produced by [`embed`](Self::embed).
    ///
    /// Stable across calls for a given store instance. Consumers allocate
    /// target buffers of this size before calling `embed`.
    fn embed_dim(&self) -> usize;
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal pruner that rejects token 0 and accepts everything else.
    struct NonZeroPruner;

    impl ConstraintPruner for NonZeroPruner {
        fn is_valid(&self, _depth: usize, token: TokenId, _parent: &[TokenId]) -> bool {
            token != 0
        }
    }

    #[test]
    fn test_is_valid_basic() {
        let pruner = NonZeroPruner;
        assert!(!pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(5, 42, &[1, 2, 3]));
    }

    #[test]
    fn test_batch_is_valid_default() {
        let pruner = NonZeroPruner;
        let candidates = vec![0, 1, 2, 0, 3];
        let mut results = vec![false; candidates.len()];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        assert_eq!(results, vec![false, true, true, false, true]);
    }

    #[test]
    fn test_batch_is_valid_handles_mismatched_lengths() {
        let pruner = NonZeroPruner;
        let candidates = vec![1, 2, 3];
        let mut results = vec![false; 2]; // shorter than candidates
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        // Only first 2 are written
        assert_eq!(results, vec![true, true]);
    }

    #[test]
    fn test_manifold_score_default_binary() {
        let pruner = NonZeroPruner;

        let valid_score = pruner.manifold_score(0, 5, &[]);
        let invalid_score = pruner.manifold_score(0, 0, &[]);

        assert!((valid_score - 1.0).abs() < 1e-6);
        assert!((invalid_score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_propagate_default_noop() {
        let mut pruner = NonZeroPruner;
        // Should not panic and should not change validity behavior
        pruner.propagate(0, 1, &[]);
        // Token 0 is always rejected by NonZeroPruner
        assert!(!pruner.is_valid(1, 0, &[1]));
        // Token 2 is valid (non-zero)
        assert!(pruner.is_valid(1, 2, &[1]));
    }

    /// Minimal draft model returning uniform logits.
    struct StubModel {
        vocab: usize,
    }

    impl DraftModel for StubModel {
        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn log_probs(&self, _context: &[TokenId]) -> Logits {
            vec![0.0; self.vocab]
        }
    }

    #[test]
    fn test_draft_model_trait_object() {
        let model: Box<dyn DraftModel> = Box::new(StubModel { vocab: 10 });
        assert_eq!(model.vocab_size(), 10);

        let logits = model.log_probs(&[1, 2, 3]);
        assert_eq!(logits.len(), 10);
        assert!(logits.iter().all(|&l| (l - 0.0).abs() < 1e-6));
    }

    #[test]
    fn test_constraint_pruner_trait_object() {
        let pruner: Box<dyn ConstraintPruner> = Box::new(NonZeroPruner);
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(!pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NonZeroPruner>();
        assert_send_sync::<StubModel>();
    }
}
