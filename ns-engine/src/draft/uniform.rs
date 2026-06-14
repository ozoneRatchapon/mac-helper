//! Uniform draft model — uninformative prior.
//!
//! Returns equal log-probability for every token in the vocabulary. The
//! ConstraintPruner alone decides which tokens are valid. This is the purest
//! demonstration of the "modelless" thesis: with a uniform draft, intelligence
//! lives entirely in the symbolic pruner layer.
//!
//! # When to use
//!
//! - Constraint satisfaction problems (Sudoku, scheduling) where the pruner
//!   encodes the full domain logic.
//! - Baseline / control group when measuring how much work the pruner does.
//! - Fallback when no trained draft model is available.

use crate::traits::DraftModel;
use crate::types::{Logits, TokenId};

/// Uniform draft model: equal probability for all tokens.
#[derive(Clone, Debug)]
pub struct UniformDraftModel {
    vocab_size: usize,
}

impl UniformDraftModel {
    /// Create a new uniform draft model with the given vocabulary size.
    ///
    /// Tokens `0..vocab_size` are all equally likely.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Vocabulary size accessor.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl DraftModel for UniformDraftModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        // Equal logit (0.0) for all tokens. Since the decode loop only uses
        // relative ranking, the constant value is irrelevant — all tokens tie.
        // The stable sort in top_k preserves index order, so candidates are
        // enumerated 0, 1, 2, ..., vocab_size-1. The pruner then filters.
        vec![0.0; self.vocab_size]
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size_accessor() {
        let model = UniformDraftModel::new(27);
        assert_eq!(model.vocab_size(), 27);
        assert_eq!(model.vocab_size, 27);
    }

    #[test]
    fn test_trait_method_matches_accessor() {
        let model = UniformDraftModel::new(10);
        assert_eq!(model.vocab_size(), model.vocab_size);
    }

    #[test]
    fn test_log_probs_length_matches_vocab() {
        let model = UniformDraftModel::new(10);
        let logits = model.log_probs(&[]);
        assert_eq!(logits.len(), 10);
    }

    #[test]
    fn test_log_probs_all_equal() {
        let model = UniformDraftModel::new(9);
        let logits = model.log_probs(&[]);
        let first = logits[0];
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                (l - first).abs() < 1e-6,
                "logit[{i}] = {l} differs from logit[0] = {first}"
            );
        }
    }

    #[test]
    fn test_log_probs_ignores_context() {
        let model = UniformDraftModel::new(5);
        let empty = model.log_probs(&[]);
        let with_ctx = model.log_probs(&[1, 2, 3]);

        assert_eq!(empty.len(), with_ctx.len());
        for i in 0..empty.len() {
            assert!((empty[i] - with_ctx[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_log_probs_value_is_zero() {
        // The convention: equal logit of 0.0 (not normalized log-prob).
        // Only relative ranking matters to the decode loop.
        let model = UniformDraftModel::new(4);
        let logits = model.log_probs(&[]);
        assert!(logits.iter().all(|&l| (l - 0.0).abs() < 1e-6));
    }

    #[test]
    fn test_trait_object_dispatch() {
        let model: Box<dyn DraftModel> = Box::new(UniformDraftModel::new(12));
        assert_eq!(model.vocab_size(), 12);

        let logits = model.log_probs(&[5, 6]);
        assert_eq!(logits.len(), 12);
        assert!(logits.iter().all(|&l| (l - 0.0).abs() < 1e-6));
    }

    #[test]
    fn test_clone_preserves_state() {
        let model = UniformDraftModel::new(8);
        let cloned = model.clone();
        assert_eq!(cloned.vocab_size(), model.vocab_size());
        assert_eq!(cloned.log_probs(&[]), model.log_probs(&[]));
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UniformDraftModel>();
    }

    #[test]
    fn test_single_token_vocab() {
        // Degenerate case: vocab of size 1 — only token 0 exists
        let model = UniformDraftModel::new(1);
        let logits = model.log_probs(&[]);
        assert_eq!(logits.len(), 1);
        assert!((logits[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_vocab() {
        let model = UniformDraftModel::new(10_000);
        let logits = model.log_probs(&[]);
        assert_eq!(logits.len(), 10_000);
        // Spot check
        assert!((logits[0] - 0.0).abs() < 1e-6);
        assert!((logits[5_000] - 0.0).abs() < 1e-6);
        assert!((logits[9_999] - 0.0).abs() < 1e-6);
    }
}
