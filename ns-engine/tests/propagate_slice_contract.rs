//! Regression: `propagate` / `on_backtrack` must receive the same slice shape
//! from `speculative_decode` and `speculative_generate`.
//!
//! Found 2026-08-16. `speculative_generate` called `propagate` BEFORE pushing
//! the action (`parent_tokens.len() == depth`), while `speculative_decode`
//! pushed first (`len == depth + 1`). A stateful pruner therefore could not be
//! correct in both loops — the concrete casualty was `BanditPruner`, which
//! hashed the whole argument and so keyed differently from its own read path
//! (see `tests/bandit_arm_attribution.rs`).
//!
//! `ConstraintPruner::propagate` now documents the post-commit contract:
//! `parent_tokens.last() == Some(&token)` and `parent_tokens.len() == depth + 1`.

use ns_engine::pruners::NoPruner;
use ns_engine::traits::{ConstraintPruner, DraftModel};
use ns_engine::{
    speculative_decode, speculative_generate, DecodeConfig, GameState, Logits, TokenId,
};
use std::sync::{Arc, Mutex};

/// Records the (depth, token, slice-length, last-element) seen by each hook.
type Seen = Arc<Mutex<Vec<(&'static str, usize, TokenId, usize, Option<TokenId>)>>>;

struct ContractSpy {
    seen: Seen,
}

impl ConstraintPruner for ContractSpy {
    fn is_valid(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> bool {
        true
    }

    fn batch_is_valid(&self, _d: usize, c: &[TokenId], _p: &[TokenId], r: &mut [bool]) {
        let len = c.len().min(r.len());
        r[..len].fill(true);
    }

    fn manifold_score(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> f32 {
        1.0
    }

    fn propagate(&mut self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) {
        self.seen.lock().unwrap().push((
            "propagate",
            depth,
            token,
            parent_tokens.len(),
            parent_tokens.last().copied(),
        ));
    }

    fn on_backtrack(&mut self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) {
        self.seen.lock().unwrap().push((
            "on_backtrack",
            depth,
            token,
            parent_tokens.len(),
            parent_tokens.last().copied(),
        ));
    }
}

struct Uniform {
    vocab: usize,
}

impl DraftModel for Uniform {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn log_probs(&self, _c: &[TokenId]) -> Logits {
        vec![1.0; self.vocab]
    }
}

/// Linear chain of `len` states; the only legal action is `0`, goal at the end.
#[derive(Clone)]
struct Chain {
    at: usize,
    len: usize,
    last: Option<TokenId>,
}

impl GameState for Chain {
    fn last_action(&self) -> Option<TokenId> {
        self.last
    }
    fn step(&self, action: TokenId) -> Self {
        Self {
            at: self.at + 1,
            len: self.len,
            last: Some(action),
        }
    }
    fn legal_actions(&self) -> Vec<TokenId> {
        if self.is_terminal() {
            Vec::new()
        } else {
            vec![0]
        }
    }
    fn is_terminal(&self) -> bool {
        self.at >= self.len
    }
    fn is_goal(&self) -> bool {
        self.at >= self.len
    }
    fn reward(&self) -> f32 {
        if self.is_goal() {
            1.0
        } else {
            0.0
        }
    }
    fn hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&(self.at as u64).to_le_bytes());
        *h.finalize().as_bytes()
    }
}

fn assert_contract(seen: &Seen, label: &str) {
    let events = seen.lock().unwrap().clone();
    assert!(!events.is_empty(), "{label}: no hook calls recorded");
    for (hook, depth, token, len, last) in events {
        match hook {
            // Post-commit: the sequence includes the token just accepted.
            "propagate" => {
                assert_eq!(
                    len,
                    depth + 1,
                    "{label}: propagate slice must be len == depth + 1"
                );
                assert_eq!(
                    last,
                    Some(token),
                    "{label}: propagate slice must end with the committed token"
                );
            }
            // Post-pop: the token is already gone.
            "on_backtrack" => {
                assert_eq!(
                    len, depth,
                    "{label}: on_backtrack slice must be len == depth"
                );
            }
            other => panic!("unexpected hook {other}"),
        }
    }
}

#[test]
fn decode_passes_post_commit_slice() {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let mut spy = ContractSpy {
        seen: Arc::clone(&seen),
    };
    let config = DecodeConfig {
        max_tokens: 5,
        top_k: 2,
        seed: 42,
        backtrack: true,
        max_attempts: 100,
    };
    let _ = speculative_decode(&Uniform { vocab: 2 }, &mut spy, &config);
    assert_contract(&seen, "speculative_decode");
}

#[test]
fn generate_passes_the_same_slice_shape_as_decode() {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let mut spy = ContractSpy {
        seen: Arc::clone(&seen),
    };
    let initial = Chain {
        at: 0,
        len: 4,
        last: None,
    };
    for backtrack in [false, true] {
        let config = DecodeConfig {
            max_tokens: 8,
            top_k: 1,
            seed: 42,
            backtrack,
            max_attempts: 100,
        };
        let result = speculative_generate(&initial, &Uniform { vocab: 1 }, &mut spy, &config);
        assert!(result.goal, "chain should be solvable (backtrack={backtrack})");
    }
    assert_contract(&seen, "speculative_generate");
}

#[test]
fn no_pruner_still_drives_both_loops() {
    // Sanity: the shared default no-op path is unaffected by the contract.
    let config = DecodeConfig {
        max_tokens: 3,
        top_k: 2,
        seed: 7,
        backtrack: false,
        max_attempts: 50,
    };
    let mut pruner = NoPruner::new();
    let out = speculative_decode(&Uniform { vocab: 2 }, &mut pruner, &config);
    assert_eq!(out.tokens.len(), 3);
}
