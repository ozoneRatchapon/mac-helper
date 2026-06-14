//! N-gram screening pruner: wraps an [`NgramDraftModel`] as a graded
//! [`ScreeningPruner`] arm for the multi-armed bandit.
//!
//! # Scoring
//!
//! The screen score for a token is its n-gram probability under the wrapped
//! model:
//!
//! ```text
//! score(token | context) = exp(log_prob(token | context)) ∈ (0.0, 1.0]
//! ```
//!
//! The [`NgramDraftModel`] uses Laplace smoothing, so probabilities are always
//! positive — every token gets a non-zero score. Highly probable continuations
//! score near 1.0; rare or unseen tokens score close to (but above) zero.
//!
//! # Validity threshold
//!
//! A token is [`is_valid`](`crate::ConstraintPruner::is_valid`) when its
//! screen score meets or exceeds `validity_threshold`. The default threshold
//! is the uniform-prior probability `1 / vocab_size`: any token the model
//! considers at least as likely as random chance is accepted.
//!
//! - Untrained model → all tokens tie at `1/V` → all valid (no information)
//! - Trained model → observed tokens exceed the threshold; rare tokens
//!   may fall below it and be rejected
//!
//! # Use as a bandit arm
//!
//! The bandit reads the screen score as a reward signal. An n-gram arm that
//! consistently produces high-probability tokens gets high mean reward and
//! is selected more often. This is the "Screening Is Enough" thesis: graded
//! relevance, not binary validity, drives adaptation.
//!
//! # Stateless
//!
//! The pruner holds no mutable state. [`propagate`](`crate::ConstraintPruner::propagate`)
//! and [`on_backtrack`](`crate::ConstraintPruner::on_backtrack`) are no-ops.
//! All validity and scoring derives from the immutable model and the
//! `parent_tokens` prefix.

use crate::draft::NgramDraftModel;
use crate::traits::{ConstraintPruner, DraftModel, ScreeningPruner};
use crate::types::{ArmId, TokenId};

/// Stable arm identifier for [`NgramScreeningPruner`].
///
/// Used for diagnostics and cross-pattern attribution. The bandit's
/// [`TrialLog`](crate::bandit::TrialLog) keys arms by Vec index, not by this
/// ID, so the value just needs to be stable and unique within a deployment.
pub const NGRAM_SCREEN_ARM_ID: ArmId = 2;

/// Default arm label reported by [`NgramScreeningPruner::arm_label`].
pub const NGRAM_SCREEN_LABEL: &str = "ngram-screen";

/// N-gram screening pruner.
///
/// Wraps an [`NgramDraftModel`] and exposes its probabilities as graded
/// [`ScreeningPruner`] scores. See the [module docs](self) for scoring
/// semantics and validity thresholds.
pub struct NgramScreeningPruner {
    /// The underlying trained n-gram model. Immutable after construction.
    model: NgramDraftModel,
    /// Diagnostics identifier (see [`NGRAM_SCREEN_ARM_ID`]).
    arm_id: ArmId,
    /// Human-readable label for logs and arena reports.
    label: String,
    /// Minimum screen score for a token to be considered valid.
    validity_threshold: f32,
}

impl NgramScreeningPruner {
    /// Create a new n-gram screening pruner wrapping `model`.
    ///
    /// The validity threshold defaults to the uniform-prior probability
    /// `1 / vocab_size` — the boundary between "more likely than random"
    /// and "less likely than random".
    ///
    /// # Panics
    ///
    /// Inherited from [`NgramDraftModel::new`]: panics if order or vocab_size
    /// is zero. The wrapped model must already be constructed.
    pub fn new(model: NgramDraftModel) -> Self {
        let vocab = model.vocab_size();
        let threshold = Self::default_threshold(vocab);
        Self {
            model,
            arm_id: NGRAM_SCREEN_ARM_ID,
            label: NGRAM_SCREEN_LABEL.to_string(),
            validity_threshold: threshold,
        }
    }

    /// Override the default [`arm_id`](ScreeningPruner::arm_id).
    ///
    /// Useful when deploying multiple n-gram arms with different training
    /// corpora in the same bandit and you need stable, distinguishable IDs.
    pub fn with_arm_id(mut self, arm_id: ArmId) -> Self {
        self.arm_id = arm_id;
        self
    }

    /// Override the default arm label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Override the default validity threshold.
    ///
    /// Set lower than `1/vocab_size` to accept more tokens (lax screening)
    /// or higher to accept only the most probable tokens (strict screening).
    pub fn with_validity_threshold(mut self, threshold: f32) -> Self {
        self.validity_threshold = threshold;
        self
    }

    /// Read-only access to the underlying n-gram model.
    pub fn model(&self) -> &NgramDraftModel {
        &self.model
    }

    /// Vocabulary size of the underlying model.
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Current validity threshold.
    pub fn validity_threshold(&self) -> f32 {
        self.validity_threshold
    }

    /// Compute the default threshold for a vocabulary size.
    ///
    /// Returns `1/vocab_size` for positive vocabularies, `0.0` for the
    /// degenerate zero-vocabulary case.
    fn default_threshold(vocab: usize) -> f32 {
        match vocab {
            0 => 0.0,
            n => 1.0 / n as f32,
        }
    }

    /// Convert a log-probability to a screen score in `[0.0, 1.0]`.
    ///
    /// `exp(log_prob)` recovers the probability. The clamp guards against
    /// floating-point drift at the boundaries (e.g., `exp(0.0)` should
    /// produce exactly `1.0`, but rounding may give `1.0000001`).
    fn log_prob_to_score(log_prob: f32) -> f32 {
        log_prob.exp().clamp(0.0, 1.0)
    }
}

impl ConstraintPruner for NgramScreeningPruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        let logits = self.model.log_probs(parent_tokens);
        match logits.get(token as usize) {
            Some(&log_prob) => Self::log_prob_to_score(log_prob) >= self.validity_threshold,
            None => false,
        }
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        // Compute the distribution once and reuse for all candidates.
        let logits = self.model.log_probs(parent_tokens);
        let len = candidates.len().min(results.len());
        for i in 0..len {
            let token = candidates[i];
            results[i] = match logits.get(token as usize) {
                Some(&log_prob) => Self::log_prob_to_score(log_prob) >= self.validity_threshold,
                None => false,
            };
        }
    }

    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        // Delegate to screen() — the graded score IS the manifold score for
        // screening pruners. This unifies the two APIs.
        ScreeningPruner::screen(self, depth, token, parent_tokens)
    }

    fn propagate(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless — the wrapped NgramDraftModel is immutable after training.
    }

    fn on_backtrack(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless — nothing to undo.
    }
}

impl ScreeningPruner for NgramScreeningPruner {
    fn arm_id(&self) -> ArmId {
        self.arm_id
    }

    fn arm_label(&self) -> &str {
        &self.label
    }

    fn screen(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        let logits = self.model.log_probs(parent_tokens);
        match logits.get(token as usize) {
            Some(&log_prob) => Self::log_prob_to_score(log_prob),
            None => 0.0,
        }
    }

    fn batch_screen(
        &self,
        _depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [f32],
    ) {
        // Single log_probs call amortizes the HashMap lookups across all
        // candidates. Overrides the per-item default for efficiency.
        let logits = self.model.log_probs(parent_tokens);
        let len = candidates.len().min(results.len());
        for i in 0..len {
            let token = candidates[i];
            results[i] = match logits.get(token as usize) {
                Some(&log_prob) => Self::log_prob_to_score(log_prob),
                None => 0.0,
            };
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──

    #[test]
    fn test_new_sets_default_threshold_to_uniform_prior() {
        let model = NgramDraftModel::new(2, 8);
        let pruner = NgramScreeningPruner::new(model);

        // 1/8 = 0.125
        assert!(
            (pruner.validity_threshold() - 0.125).abs() < 1e-6,
            "default threshold should be 1/vocab_size"
        );
    }

    #[test]
    fn test_vocab_size_accessor() {
        let model = NgramDraftModel::new(2, 16);
        let pruner = NgramScreeningPruner::new(model);
        assert_eq!(pruner.vocab_size(), 16);
    }

    #[test]
    fn test_default_threshold_single_token_vocab() {
        let model = NgramDraftModel::new(1, 1);
        let pruner = NgramScreeningPruner::new(model);
        // 1/1 = 1.0 — only token 0 exists, must be valid.
        assert!((pruner.validity_threshold() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_with_arm_id_overrides_default() {
        let model = NgramDraftModel::new(1, 4);
        let pruner = NgramScreeningPruner::new(model).with_arm_id(99);
        assert_eq!(pruner.arm_id(), 99);
    }

    #[test]
    fn test_with_label_overrides_default() {
        let model = NgramDraftModel::new(1, 4);
        let pruner = NgramScreeningPruner::new(model).with_label("custom-ngram");
        assert_eq!(pruner.arm_label(), "custom-ngram");
    }

    #[test]
    fn test_with_validity_threshold_overrides_default() {
        let model = NgramDraftModel::new(1, 4);
        let pruner = NgramScreeningPruner::new(model).with_validity_threshold(0.42);
        assert!((pruner.validity_threshold() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_model_accessor_returns_underlying() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);
        assert_eq!(pruner.model().vocab_size(), 4);
    }

    // ── Screen scores on untrained model ──

    #[test]
    fn test_screen_untrained_returns_uniform_probability() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);

        // Untrained: P(any) = 1/4 → score = 0.25.
        let score = pruner.screen(0, 2, &[]);
        assert!(
            (score - 0.25).abs() < 1e-6,
            "untrained score should be 1/vocab_size, got {score}"
        );
    }

    #[test]
    fn test_screen_out_of_vocab_returns_zero() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);

        // Token 10 is out of vocab (vocab_size=4).
        let score = pruner.screen(0, 10, &[]);
        assert!((score - 0.0).abs() < 1e-6);
    }

    // ── Screen scores on trained model ──

    #[test]
    fn test_screen_reflects_training_counts() {
        let mut model = NgramDraftModel::new(2, 4);
        // Train: after 0, token 1 appears 10 times.
        for _ in 0..10 {
            model.train(&[0u32, 1]);
        }

        let pruner = NgramScreeningPruner::new(model);

        // P(1|0) = (10+1)/(10+4) = 11/14 ≈ 0.786
        let score_1 = pruner.screen(1, 1, &[0]);
        let expected_1 = 11.0f32 / 14.0;
        assert!(
            (score_1 - expected_1).abs() < 1e-5,
            "P(1|0) score mismatch: {score_1} vs {expected_1}"
        );

        // P(2|0) = (0+1)/(10+4) = 1/14 ≈ 0.071
        let score_2 = pruner.screen(1, 2, &[0]);
        let expected_2 = 1.0f32 / 14.0;
        assert!(
            (score_2 - expected_2).abs() < 1e-5,
            "P(2|0) score mismatch: {score_2} vs {expected_2}"
        );
    }

    #[test]
    fn test_screen_higher_for_observed_than_unobserved() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 0, 1, 0, 1]);

        let pruner = NgramScreeningPruner::new(model);
        let observed = pruner.screen(1, 1, &[0]);
        let unobserved = pruner.screen(1, 2, &[0]);

        assert!(
            observed > unobserved,
            "observed token should score higher: {observed} vs {unobserved}"
        );
    }

    // ── Validity threshold ──

    #[test]
    fn test_is_valid_untrained_accepts_all_at_uniform() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);

        // Untrained: score = 0.25, threshold = 0.25 → all valid (>=).
        for token in 0..4u32 {
            assert!(
                pruner.is_valid(0, token, &[]),
                "token {token} should be valid at uniform threshold"
            );
        }
    }

    #[test]
    fn test_is_valid_rejects_out_of_vocab() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);

        assert!(!pruner.is_valid(0, 99, &[]));
        assert!(!pruner.is_valid(0, 4, &[]));
    }

    #[test]
    fn test_is_valid_strict_threshold_rejects_rare() {
        let mut model = NgramDraftModel::new(2, 4);
        // Train so that token 1 after 0 is common, token 2 is rare.
        for _ in 0..10 {
            model.train(&[0u32, 1]);
        }
        model.train(&[0u32, 2]); // one occurrence of (0,2)

        // Strict threshold: only accept tokens with prob >= 0.5.
        let pruner = NgramScreeningPruner::new(model).with_validity_threshold(0.5);

        assert!(
            pruner.is_valid(1, 1, &[0]),
            "P(1|0) ≈ 0.786 should pass threshold 0.5"
        );
        assert!(
            !pruner.is_valid(1, 2, &[0]),
            "P(2|0) ≈ 0.071 should fail threshold 0.5"
        );
    }

    #[test]
    fn test_is_valid_lax_threshold_accepts_all() {
        let mut model = NgramDraftModel::new(2, 4);
        for _ in 0..10 {
            model.train(&[0u32, 1]);
        }

        // Lax threshold: accept everything with any probability.
        let pruner = NgramScreeningPruner::new(model).with_validity_threshold(0.0);

        for token in 0..4u32 {
            assert!(
                pruner.is_valid(1, token, &[0]),
                "lax threshold should accept token {token}"
            );
        }
    }

    // ── Batch operations ──

    #[test]
    fn test_batch_screen_matches_individual() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 0, 1, 0, 2]);

        let pruner = NgramScreeningPruner::new(model);
        let candidates = vec![0u32, 1, 2, 3];
        let parent = vec![0u32];

        let mut batch_scores = vec![0.0f32; candidates.len()];
        pruner.batch_screen(1, &candidates, &parent, &mut batch_scores);

        for (i, &tok) in candidates.iter().enumerate() {
            let individual = pruner.screen(1, tok, &parent);
            let batch_score = batch_scores[i];
            assert!(
                (batch_score - individual).abs() < 1e-6,
                "batch[{i}]={batch_score} != individual {individual}"
            );
        }
    }

    #[test]
    fn test_batch_is_valid_matches_individual() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 0, 1, 0, 2]);

        let pruner = NgramScreeningPruner::new(model);
        let candidates = vec![0u32, 1, 2, 3, 99];
        let parent = vec![0u32];

        let mut batch_results = vec![false; candidates.len()];
        pruner.batch_is_valid(1, &candidates, &parent, &mut batch_results);

        for (i, &tok) in candidates.iter().enumerate() {
            let individual = pruner.is_valid(1, tok, &parent);
            let batch_result = batch_results[i];
            assert_eq!(
                batch_result, individual,
                "batch[{i}]={batch_result} != individual {individual} for token {tok}"
            );
        }
    }

    #[test]
    fn test_batch_handles_shorter_results_slice() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);
        let candidates = vec![0u32, 1, 2, 3];
        let mut results = vec![false; 2]; // shorter than candidates

        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        // Only first 2 should be written.
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_batch_empty_candidates() {
        let model = NgramDraftModel::new(2, 4);
        let pruner = NgramScreeningPruner::new(model);
        let mut results: Vec<bool> = vec![];

        pruner.batch_is_valid(0, &[], &[], &mut results);
        assert!(results.is_empty());
    }

    // ── manifold_score delegates to screen ──

    #[test]
    fn test_manifold_score_equals_screen() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 0, 1]);

        let pruner = NgramScreeningPruner::new(model);
        let ms = pruner.manifold_score(1, 1, &[0]);
        let sc = ScreeningPruner::screen(&pruner, 1, 1, &[0]);

        assert!(
            (ms - sc).abs() < 1e-6,
            "manifold_score should equal screen: {ms} vs {sc}"
        );
    }

    // ── Stateless: propagate and on_backtrack ──

    #[test]
    fn test_propagate_is_noop() {
        let model = NgramDraftModel::new(2, 4);
        let mut pruner = NgramScreeningPruner::new(model);

        let before = pruner.screen(0, 1, &[]);
        pruner.propagate(0, 1, &[1]);
        let after = pruner.screen(0, 1, &[]);

        assert!(
            (before - after).abs() < 1e-9,
            "propagate must not change screen scores"
        );
    }

    #[test]
    fn test_on_backtrack_is_noop() {
        let model = NgramDraftModel::new(2, 4);
        let mut pruner = NgramScreeningPruner::new(model);

        let before = pruner.screen(0, 1, &[]);
        pruner.on_backtrack(0, 1, &[]);
        let after = pruner.screen(0, 1, &[]);

        assert!(
            (before - after).abs() < 1e-9,
            "on_backtrack must not change screen scores"
        );
    }

    // ── Edge cases ──

    #[test]
    fn test_screen_with_empty_parent_context() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 2, 3]);

        let pruner = NgramScreeningPruner::new(model);
        // Empty context → unigram-like evaluation for bigram model.
        let score = pruner.screen(0, 0, &[]);
        assert!(
            score > 0.0,
            "score should be positive even with empty context"
        );
    }

    #[test]
    fn test_screen_with_long_parent_context() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 2, 3, 0, 1]);

        let pruner = NgramScreeningPruner::new(model);
        // Long context — model uses only the last (order-1) = 1 token.
        let score = pruner.screen(5, 1, &[0, 1, 2, 3, 0]);
        assert!(score > 0.0);
    }

    #[test]
    fn test_log_prob_to_score_zero_for_negative_infinity() {
        // exp(-inf) = 0 — edge case for extremely negative log-probs.
        let score = NgramScreeningPruner::log_prob_to_score(f32::NEG_INFINITY);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_prob_to_score_one_for_zero() {
        // exp(0) = 1 — probability 1.0 (certain event).
        let score = NgramScreeningPruner::log_prob_to_score(0.0);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_prob_to_score_in_range() {
        // For log_prob in [-10, 0], score should be in (0, 1].
        for log_prob in [-10.0f32, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0] {
            let score = NgramScreeningPruner::log_prob_to_score(log_prob);
            assert!(
                (0.0..=1.0).contains(&score),
                "score {score} for log_prob {log_prob} out of [0, 1]"
            );
        }
    }

    // ── Trait object dispatch ──

    #[test]
    fn test_constraint_pruner_trait_object() {
        let model = NgramDraftModel::new(2, 4);
        let pruner: Box<dyn ConstraintPruner> = Box::new(NgramScreeningPruner::new(model));

        assert!(pruner.is_valid(0, 0, &[]));
        assert!(!pruner.is_valid(0, 99, &[]));
    }

    #[test]
    fn test_screening_pruner_trait_object() {
        let model = NgramDraftModel::new(2, 4);
        let pruner: Box<dyn ScreeningPruner> = Box::new(NgramScreeningPruner::new(model));

        assert_eq!(pruner.arm_id(), NGRAM_SCREEN_ARM_ID);
        assert_eq!(pruner.arm_label(), NGRAM_SCREEN_LABEL);

        let score = pruner.screen(0, 0, &[]);
        assert!(score > 0.0);
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NgramScreeningPruner>();
    }
}
