//! No-op pruner: allows all tokens (baseline / control group).
//!
//! Mirrors katopz/katgpt-rs `NoPruner`. Use this to measure the upper bound
//! of decode-tree expansion without any constraint filtering, or as a
//! fallback when no domain-specific pruner is available.

use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

/// Stable arm identifier for [`NoPruner`].
///
/// Used for diagnostics and cross-pattern attribution inside a
/// [`BanditPruner`](crate::bandit::BanditPruner). The trial log keys arms
/// by Vec index, not by this ID, so the value just needs to be stable and
/// unique within a deployment.
pub const NO_PRUNER_ARM_ID: ArmId = 4;

/// Default arm label reported by [`NoPruner::arm_label`].
pub const NO_PRUNER_LABEL: &str = "no-pruner";

/// Pruner that accepts every token at every depth.
///
/// The decode loop behaves as pure speculative drafting: all top-k
/// candidates are valid, no branches are pruned.
pub struct NoPruner;

impl NoPruner {
    /// Create a new no-op pruner.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoPruner {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintPruner for NoPruner {
    fn is_valid(&self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) -> bool {
        true
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[TokenId],
        _parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        results[..len].fill(true);
    }

    fn manifold_score(&self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) -> f32 {
        1.0
    }
}

impl ScreeningPruner for NoPruner {
    fn arm_id(&self) -> ArmId {
        NO_PRUNER_ARM_ID
    }

    fn arm_label(&self) -> &str {
        NO_PRUNER_LABEL
    }

    fn screen(&self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) -> f32 {
        // All tokens are equally "relevant" to a no-op pruner.
        1.0
    }

    fn batch_screen(
        &self,
        _depth: usize,
        candidates: &[TokenId],
        _parent_tokens: &[TokenId],
        results: &mut [f32],
    ) {
        let len = candidates.len().min(results.len());
        results[..len].fill(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_always_true() {
        let pruner = NoPruner::new();
        assert!(pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(42, 999, &[1, 2, 3]));
        assert!(pruner.is_valid(usize::MAX, TokenId::MAX, &[0, 0, 0]));
    }

    #[test]
    fn test_batch_is_valid_fills_true() {
        let pruner = NoPruner::new();
        let candidates = vec![0, 1, 2, 3, 4];
        let mut results = vec![false; candidates.len()];

        pruner.batch_is_valid(5, &candidates, &[1, 2], &mut results);

        assert_eq!(results, vec![true, true, true, true, true]);
    }

    #[test]
    fn test_batch_is_valid_handles_shorter_results() {
        let pruner = NoPruner::new();
        let candidates = vec![1, 2, 3, 4, 5];
        let mut results = vec![false; 3];

        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_batch_is_valid_empty_candidates() {
        let pruner = NoPruner::new();
        let mut results: Vec<bool> = vec![];

        pruner.batch_is_valid(0, &[], &[], &mut results);

        assert!(results.is_empty());
    }

    #[test]
    fn test_manifold_score_always_one() {
        let pruner = NoPruner::new();

        let score_a = pruner.manifold_score(0, 0, &[]);
        let score_b = pruner.manifold_score(10, 42, &[1, 2]);

        assert!((score_a - 1.0).abs() < 1e-6);
        assert!((score_b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_equals_new() {
        let a = NoPruner::new();
        let b = NoPruner;

        assert!(a.is_valid(0, 0, &[]));
        assert!(b.is_valid(0, 0, &[]));
    }

    // ── ScreeningPruner impl ──

    #[test]
    fn test_arm_id_matches_constant() {
        let pruner = NoPruner::new();
        assert_eq!(pruner.arm_id(), NO_PRUNER_ARM_ID);
    }

    #[test]
    fn test_arm_label_matches_constant() {
        let pruner = NoPruner::new();
        assert_eq!(pruner.arm_label(), NO_PRUNER_LABEL);
    }

    #[test]
    fn test_screen_always_one() {
        let pruner = NoPruner::new();
        assert!((pruner.screen(0, 0, &[]) - 1.0).abs() < 1e-6);
        assert!((pruner.screen(42, 99, &[1, 2]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_screen_fills_ones() {
        let pruner = NoPruner::new();
        let candidates = vec![0, 1, 2, 3];
        let mut results = vec![0.0f32; candidates.len()];

        pruner.batch_screen(0, &candidates, &[], &mut results);

        for r in &results {
            assert!((r - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_screen_handles_shorter_results() {
        let pruner = NoPruner::new();
        let candidates = vec![0, 1, 2, 3, 4];
        let mut results = vec![0.0f32; 3];

        pruner.batch_screen(0, &candidates, &[], &mut results);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!((r - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_screening_pruner_trait_object() {
        let pruner: Box<dyn ScreeningPruner> = Box::new(NoPruner::new());
        assert_eq!(pruner.arm_id(), NO_PRUNER_ARM_ID);
        assert_eq!(pruner.arm_label(), NO_PRUNER_LABEL);
        assert!(pruner.is_valid(0, 0, &[]));
        assert!((pruner.screen(0, 0, &[]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoPruner>();
    }
}
