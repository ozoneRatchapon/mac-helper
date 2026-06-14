//! N-gram draft model with Laplace smoothing.
//!
//! A real statistical language model: counts token transitions from a training
//! corpus and converts to log-probabilities. No neural network, no GPU —
//! just counting and normalization.
//!
//! # Order
//!
//! Order N uses the last N-1 tokens as context:
//! - N=1 → unigram (no context, just token frequency)
//! - N=2 → bigram (1 token of context)
//! - N=3 → trigram (2 tokens of context)
//!
//! # Smoothing
//!
//! Laplace (add-one) smoothing prevents zero probabilities for unseen n-grams:
//! ```text
//! P(token | context) = (count(context, token) + 1) / (count(context) + V)
//! ```
//! where V is the vocabulary size.

use crate::traits::DraftModel;
use crate::types::{Logits, TokenId};
use std::collections::HashMap;

/// N-gram statistical draft model.
pub struct NgramDraftModel {
    /// N-gram order (1=unigram, 2=bigram, 3=trigram).
    order: usize,
    /// Vocabulary size.
    vocab_size: usize,
    /// Context → per-token counts. Length = vocab_size per entry.
    counts: HashMap<Vec<TokenId>, Vec<u32>>,
    /// Total count per context (sum of all token counts for that context).
    context_totals: HashMap<Vec<TokenId>, u32>,
}

impl NgramDraftModel {
    /// Create a new n-gram model with the given order and vocabulary size.
    ///
    /// # Panics
    ///
    /// Panics if `order` is 0 or `vocab_size` is 0.
    pub fn new(order: usize, vocab_size: usize) -> Self {
        assert!(order >= 1, "order must be >= 1");
        assert!(vocab_size >= 1, "vocab_size must be >= 1");

        Self {
            order,
            vocab_size,
            counts: HashMap::new(),
            context_totals: HashMap::new(),
        }
    }

    /// Train the model from a token corpus.
    ///
    /// Counts all n-gram transitions. Tokens >= `vocab_size` are skipped
    /// (treated as out-of-vocabulary).
    pub fn train(&mut self, corpus: &[TokenId]) {
        if corpus.len() < self.order {
            return;
        }

        for window in corpus.windows(self.order) {
            let (ctx, next) = window.split_at(self.order - 1);
            let next_tok = next[0];
            let next_idx = next_tok as usize;

            if next_idx >= self.vocab_size {
                continue;
            }

            // Skip if any context token is out of vocabulary
            if ctx.iter().any(|&t| t as usize >= self.vocab_size) {
                continue;
            }

            let counts = self
                .counts
                .entry(ctx.to_vec())
                .or_insert_with(|| vec![0; self.vocab_size]);
            counts[next_idx] += 1;
            *self.context_totals.entry(ctx.to_vec()).or_insert(0) += 1;
        }
    }

    /// Compute log-probability of `token` given `context`, with Laplace smoothing.
    fn log_prob(&self, context: &[TokenId], token: TokenId) -> f32 {
        let total = self.context_totals.get(context).copied().unwrap_or(0);
        let count = self
            .counts
            .get(context)
            .and_then(|c| c.get(token as usize).copied())
            .unwrap_or(0);

        // Laplace: (count + 1) / (total + V)
        let prob = (count as f32 + 1.0) / (total as f32 + self.vocab_size as f32);
        prob.ln()
    }

    /// Number of distinct contexts observed during training.
    pub fn context_count(&self) -> usize {
        self.counts.len()
    }

    /// Total number of n-gram observations recorded.
    pub fn observation_count(&self) -> u32 {
        self.context_totals.values().copied().sum()
    }
}

impl DraftModel for NgramDraftModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn log_probs(&self, context: &[TokenId]) -> Logits {
        // Use last (order-1) tokens as context
        let ctx_len = self.order.saturating_sub(1);
        let start = context.len().saturating_sub(ctx_len);
        let ngram_ctx = &context[start..];

        (0..self.vocab_size as TokenId)
            .map(|tok| self.log_prob(ngram_ctx, tok))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_requires_positive_order() {
        let model = NgramDraftModel::new(2, 10);
        assert_eq!(model.order, 2);
        assert_eq!(model.vocab_size, 10);
    }

    #[test]
    #[should_panic(expected = "order must be >= 1")]
    fn test_new_rejects_zero_order() {
        NgramDraftModel::new(0, 10);
    }

    #[test]
    #[should_panic(expected = "vocab_size must be >= 1")]
    fn test_new_rejects_zero_vocab() {
        NgramDraftModel::new(2, 0);
    }

    #[test]
    fn test_empty_model_returns_uniform_logprobs() {
        let model = NgramDraftModel::new(2, 5);
        let logits = model.log_probs(&[]);

        assert_eq!(logits.len(), 5);
        // All should be equal (Laplace with 0 counts → 1/5 each)
        let expected = (1.0f32 / 5.0f32).ln();
        for &l in &logits {
            assert!((l - expected).abs() < 1e-6, "expected uniform log-prob");
        }
    }

    #[test]
    fn test_train_records_counts() {
        let mut model = NgramDraftModel::new(2, 4);
        // Corpus: 0→1, 1→2, 2→3, 3→0, 0→1 (bigrams)
        let corpus = vec![0u32, 1, 2, 3, 0, 1];
        model.train(&corpus);

        assert_eq!(model.observation_count(), 5);
        assert_eq!(model.context_count(), 4); // contexts: [0], [1], [2], [3]
    }

    #[test]
    fn test_train_skips_out_of_vocab() {
        let mut model = NgramDraftModel::new(2, 3);
        // Token 5 is out of vocab (vocab_size=3), bigrams containing it are skipped
        let corpus = vec![0u32, 1, 5, 2];
        model.train(&corpus);

        // Only bigram (0,1) is fully in-vocab; (1,5) skipped (5 OOV),
        // (5,2) skipped (5 OOV)
        assert_eq!(model.observation_count(), 1);
    }

    #[test]
    fn test_train_short_corpus_no_op() {
        let mut model = NgramDraftModel::new(3, 5);
        model.train(&[0u32, 1]); // length 2 < order 3
        assert_eq!(model.observation_count(), 0);
    }

    #[test]
    fn test_log_probs_reflect_counts() {
        let mut model = NgramDraftModel::new(2, 4);
        // Train: after 0, we see 1 twice and 2 once
        let corpus = vec![0u32, 1, 0, 1, 0, 2];
        model.train(&corpus);

        let logits = model.log_probs(&[0]);

        // P(1|0) = (2+1)/(3+4) = 3/7
        let p1 = (3.0f32 / 7.0).ln();
        // P(2|0) = (1+1)/(3+4) = 2/7
        let p2 = (2.0f32 / 7.0).ln();
        // P(0|0) = (0+1)/(3+4) = 1/7
        let p0 = (1.0f32 / 7.0).ln();

        assert!((logits[1] - p1).abs() < 1e-6, "P(1|0) mismatch");
        assert!((logits[2] - p2).abs() < 1e-6, "P(2|0) mismatch");
        assert!((logits[0] - p0).abs() < 1e-6, "P(0|0) mismatch");
    }

    #[test]
    fn test_log_probs_unseen_context_uniform() {
        let mut model = NgramDraftModel::new(2, 4);
        model.train(&[0u32, 1, 0, 1]);

        // Context [2] was never seen → uniform Laplace: 1/4 each
        let logits = model.log_probs(&[2]);
        let expected = (1.0f32 / 4.0).ln();

        for &l in &logits {
            assert!((l - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_unigram_model() {
        let mut model = NgramDraftModel::new(1, 4);
        // Unigram: just token frequencies
        model.train(&[0u32, 0, 1, 0, 1, 1, 1]);

        let logits = model.log_probs(&[]); // context unused for unigram
        let total = 7.0f32;
        let v = 4.0f32;

        // P(0) = (3+1)/(7+4) = 4/11
        assert!((logits[0] - (4.0 / (total + v)).ln()).abs() < 1e-6);
        // P(1) = (4+1)/(7+4) = 5/11
        assert!((logits[1] - (5.0 / (total + v)).ln()).abs() < 1e-6);
    }

    #[test]
    fn test_trigram_uses_two_token_context() {
        let mut model = NgramDraftModel::new(3, 4);
        // Trigrams: (0,0)→1, (0,1)→2, (1,2)→3
        model.train(&[0u32, 0, 1, 2, 3]);

        // After [0,0], token 1 should have highest prob
        let logits_after_00 = model.log_probs(&[0, 0]);
        let best_after_00 = logits_after_00
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best_after_00, 1);

        // After [0,1], token 2 should have highest prob
        let logits_after_01 = model.log_probs(&[0, 1]);
        let best_after_01 = logits_after_01
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best_after_01, 2);
    }

    #[test]
    fn test_trigram_pads_short_context() {
        let mut model = NgramDraftModel::new(3, 4);
        model.train(&[0u32, 1, 2]);

        // Single-token context [1] → uses [1] as trigram context
        // No trigram (x,1)→y was seen, so uniform Laplace
        let logits = model.log_probs(&[1]);
        let expected = (1.0f32 / 4.0).ln();
        for &l in &logits {
            assert!((l - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_log_probs_len_matches_vocab() {
        let model = NgramDraftModel::new(2, 16);
        let logits = model.log_probs(&[0, 1, 2]);
        assert_eq!(logits.len(), 16);
    }

    #[test]
    fn test_context_count_after_training() {
        let mut model = NgramDraftModel::new(2, 5);
        model.train(&[0u32, 1, 2, 3, 4, 0, 1]);
        // Bigram contexts: [0], [1], [2], [3], [4] → 5 distinct
        assert_eq!(model.context_count(), 5);
    }

    #[test]
    fn test_multiple_train_calls_accumulate() {
        let mut model = NgramDraftModel::new(2, 4);

        model.train(&[0u32, 1]);
        assert_eq!(model.observation_count(), 1);

        model.train(&[0u32, 1]);
        assert_eq!(model.observation_count(), 2);

        // P(1|0) = (2+1)/(2+4) = 3/6 = 0.5
        let logits = model.log_probs(&[0]);
        assert!((logits[1] - 0.5f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_vocab_size_accessor() {
        let model = NgramDraftModel::new(2, 42);
        assert_eq!(model.vocab_size(), 42);
    }
}
