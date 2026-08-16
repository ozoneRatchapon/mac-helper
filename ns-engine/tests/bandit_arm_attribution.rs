//! Regression: BanditPruner must attribute rewards to the arm that validated.
//!
//! Found 2026-08-16. `propagate` selected its arm from `pattern_key(depth,
//! parent_tokens)`, but `speculative_decode` passes the POST-push token vec
//! while `batch_is_valid` saw the prefix — different hashes, so the reward,
//! the `committed[]` blame record, and the forwarded `propagate` all went to
//! a different arm than the one that actually filtered the token.

use ns_engine::bandit::{pattern_key, BanditPolicy, BanditPruner};
use ns_engine::traits::{ConstraintPruner, DraftModel, ScreeningPruner};
use ns_engine::{speculative_decode, ArmId, DecodeConfig, Logits, TokenId};
use std::sync::{Arc, Mutex};

type Log = Arc<Mutex<Vec<(&'static str, usize)>>>;

struct Recorder {
    id: usize,
    log: Log,
}

impl ConstraintPruner for Recorder {
    fn is_valid(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> bool {
        true
    }
    fn batch_is_valid(&self, _d: usize, c: &[TokenId], _p: &[TokenId], r: &mut [bool]) {
        self.log.lock().unwrap().push(("validate", self.id));
        let len = c.len().min(r.len());
        r[..len].fill(true);
    }
    fn manifold_score(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> f32 {
        1.0
    }
    fn propagate(&mut self, _d: usize, _t: TokenId, _p: &[TokenId]) {
        self.log.lock().unwrap().push(("propagate", self.id));
    }
}

impl ScreeningPruner for Recorder {
    fn arm_id(&self) -> ArmId {
        self.id as ArmId
    }
    fn arm_label(&self) -> &str {
        "recorder"
    }
    fn screen(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> f32 {
        // Arm 0 reports a poor score so UCB1 learns to prefer arm 1 for
        // patterns it has actually observed.
        if self.id == 0 {
            0.0
        } else {
            1.0
        }
    }
}

struct Uniform;
impl DraftModel for Uniform {
    fn vocab_size(&self) -> usize {
        3
    }
    fn log_probs(&self, _c: &[TokenId]) -> Logits {
        vec![0.5, 0.2, 0.1]
    }
}

#[test]
fn pattern_key_differs_between_prefix_and_post_push_slice() {
    // propagate() selects its arm with pattern_key(depth, parent_tokens) where
    // decode.rs passes the POST-push token vec; batch_is_valid saw the prefix.
    let prefix: Vec<TokenId> = vec![7];
    let post_push: Vec<TokenId> = vec![7, 9];
    assert_ne!(
        pattern_key(1, &prefix),
        pattern_key(1, &post_push),
        "pattern_key hashes the whole slice, so the two call sites key differently"
    );
}

#[test]
fn propagate_forwards_to_the_arm_that_validated() {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let arms: Vec<Box<dyn ScreeningPruner>> = vec![
        Box::new(Recorder { id: 0, log: Arc::clone(&log) }),
        Box::new(Recorder { id: 1, log: Arc::clone(&log) }),
    ];
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let config = DecodeConfig {
        max_tokens: 4,
        top_k: 3,
        seed: 42,
        backtrack: false,
        max_attempts: 100,
    };

    // Two runs: the first seeds the trial log under PREFIX patterns, so the
    // second run's batch_is_valid has stats while propagate's (prefix+token)
    // pattern is still unseen.
    for _ in 0..2 {
        let _ = speculative_decode(&Uniform, &mut bandit, &config);
    }

    let events = log.lock().unwrap().clone();
    let pairs: Vec<_> = events
        .chunks(2)
        .filter(|w| w.len() == 2 && w[0].0 == "validate" && w[1].0 == "propagate")
        .map(|w| (w[0].1, w[1].1))
        .collect();
    let mismatches: Vec<_> = pairs.iter().filter(|(v, p)| v != p).collect();
    println!("validate/propagate arm pairs: {pairs:?}");
    println!("mismatches: {mismatches:?}");
    assert!(
        mismatches.is_empty(),
        "propagate forwarded to a different arm than the one that validated: {mismatches:?}"
    );
}
