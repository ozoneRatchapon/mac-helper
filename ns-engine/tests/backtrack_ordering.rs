//! Regression test: backtracking search must explore candidates in
//! DESCENDING draft preference, not inverted.
//!
//! Found 2026-08-16 via the ollama-lab bridge experiment: `untried[depth]`
//! was stored best-first while `pop()` consumes from the end, so backtrack
//! mode explored the WORST-ranked candidate first. Every prior test used
//! `UniformDraftModel` (all logits tied → shuffled), which cannot observe
//! ordering. A rigged non-uniform draft makes the ordering visible.

use ns_engine::pruners::NoPruner;
use ns_engine::{speculative_decode, DecodeConfig, DraftModel, Logits, TokenId};

/// Draft model with fixed, distinct logits: token 1 best, then 2, then 0.
struct RiggedDraft;

impl DraftModel for RiggedDraft {
    fn vocab_size(&self) -> usize {
        3
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        vec![0.1, 0.9, 0.5]
    }
}

#[test]
fn backtrack_explores_best_candidate_first() {
    let draft = RiggedDraft;
    let mut pruner = NoPruner::new();
    let config = DecodeConfig {
        max_tokens: 4,
        top_k: 3,
        seed: 42,
        backtrack: true,
        max_attempts: 100,
    };

    let result = speculative_decode(&draft, &mut pruner, &config);

    // With a permissive pruner nothing forces backtracking, so the DFS
    // should commit the highest-logit token (1) at every depth.
    assert_eq!(
        result.tokens,
        vec![1, 1, 1, 1],
        "backtracking search must consume candidates best-first \
         (untried is a stack: best candidate belongs on top)"
    );
}

#[test]
fn greedy_and_backtrack_agree_on_unconstrained_argmax() {
    let draft = RiggedDraft;
    let config_greedy = DecodeConfig {
        max_tokens: 4,
        top_k: 3,
        seed: 42,
        backtrack: false,
        max_attempts: 100,
    };
    let config_backtrack = DecodeConfig {
        backtrack: true,
        ..config_greedy.clone()
    };

    let mut pruner = NoPruner::new();
    let greedy = speculative_decode(&draft, &mut pruner, &config_greedy);
    let mut pruner = NoPruner::new();
    let backtrack = speculative_decode(&draft, &mut pruner, &config_backtrack);

    // Greedy takes `valid.first()` (correct by construction); backtrack must
    // match it when no dead-ends occur.
    assert_eq!(greedy.tokens, backtrack.tokens);
}
