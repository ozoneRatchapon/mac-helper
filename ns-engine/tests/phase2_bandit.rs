//! Phase 2 end-to-end proof: multi-armed bandit pruner selection.
//!
//! These tests demonstrate the "speculation on policy" thesis from katopz:
//! instead of always firing the same pruner, the [`BanditPruner`] tracks
//! per-context reward statistics per arm and adapts its selection over time.
//! Intelligence evolves by updating the trial log, not by gradient descent.
//!
//! # Test groups
//!
//! 1. **ScreeningPruner dispatch** — each arm type dispatches correctly
//!    through the trait object.
//! 2. **Arena proof** — bandit total reward > static > random, across three
//!    policies (UCB1, ε-greedy, Thompson).
//! 3. **Per-context learning** — the bandit converges to the optimal arm
//!    for each pattern type.
//! 4. **Decode loop integration** — [`BanditPruner`] wraps a single
//!    [`SudokuPruner`] arm and solves Sudoku via `speculative_decode`.
//! 5. **AbsorbCompress** — stable high-Q patterns are absorbed into hard
//!    rules; unstable patterns are preserved.
//!
//! Run with:
//!   cargo test --test phase2_bandit -- --nocapture

use ns_engine::bandit::{pattern_key, BanditPolicy, BanditPruner, PatternKey};
use ns_engine::draft::{NgramDraftModel, UniformDraftModel};
use ns_engine::pruners::{NgramScreeningPruner, NoPruner, RegexPruner, SudokuPruner};
use ns_engine::{speculative_decode, DecodeConfig, ScreeningPruner, TokenId};

// ── Arena helpers ──────────────────────────────────────────────

/// Build the three Arena arms: Sudoku, N-gram, Regex.
///
/// Each is a real [`ScreeningPruner`] with distinct semantics:
/// - SudokuPruner: hard row/col/box constraints (digits 1-9)
/// - NgramScreeningPruner: graded n-gram probabilities
/// - RegexPruner: regex prefix validity for `[a-z]+`
fn build_arena_arms() -> Vec<Box<dyn ScreeningPruner>> {
    let ngram_model = NgramDraftModel::new(2, 10);
    vec![
        Box::new(SudokuPruner::empty()),
        Box::new(NgramScreeningPruner::new(ngram_model)),
        Box::new(RegexPruner::from_pattern("[a-z]+").expect("valid regex pattern")),
    ]
}

/// Reward table for the Arena: `[pattern_type][arm] = expected_reward`.
///
/// Each pattern type has a DIFFERENT optimal arm:
///
/// ```text
///                  Arm 0 (Sudoku)  Arm 1 (Ngram)  Arm 2 (Regex)
/// Pattern 0          0.90            0.20           0.30
/// Pattern 1          0.10            0.80           0.40
/// Pattern 2          0.10            0.30           0.85
/// ```
///
/// Optimal per-pattern arm: `[0, 1, 2]`.
/// Best static arm (highest cross-pattern average): arm 2 (avg ≈ 0.517).
/// Random baseline average: ≈ 0.439.
const ARENA_REWARD_TABLE: [[f64; 3]; 3] =
    [[0.90, 0.20, 0.30], [0.10, 0.80, 0.40], [0.10, 0.30, 0.85]];

/// Number of arms in the Arena.
const ARENA_ARM_COUNT: usize = 3;

/// Number of pattern types in the Arena.
const ARENA_PATTERN_COUNT: usize = 3;

/// Generate distinct [`PatternKey`]s for each Arena context type.
///
/// Each key is a BLAKE3 hash of a unique `(depth, prefix)` pair, ensuring
/// the bandit tracks them independently.
fn arena_patterns() -> [PatternKey; ARENA_PATTERN_COUNT] {
    [
        pattern_key(0, &[]),
        pattern_key(0, &[42]),
        pattern_key(0, &[99]),
    ]
}

/// Look up the Arena reward for a `(pattern_type, arm)` pair.
fn arena_reward(pattern_type: usize, arm: usize) -> f64 {
    ARENA_REWARD_TABLE[pattern_type][arm]
}

/// Find the arm with the highest average reward across all pattern types.
///
/// This is the optimal **static** (fixed-arm) strategy.
///
/// Clippy's `needless_range_loop` suggests iterating `ARENA_REWARD_TABLE.iter()`,
/// but that iterates over **patterns** (outer dim), not **arms** (inner dim).
/// The loop computes column averages — `arm` indexes the inner array — so the
/// explicit range is semantically correct here.
#[allow(clippy::needless_range_loop)]
fn best_average_arm() -> usize {
    let mut best_arm = 0usize;
    let mut best_avg = f64::NEG_INFINITY;

    for arm in 0..ARENA_ARM_COUNT {
        let avg: f64 = (0..ARENA_PATTERN_COUNT)
            .map(|pat| ARENA_REWARD_TABLE[pat][arm])
            .sum::<f64>()
            / ARENA_PATTERN_COUNT as f64;

        if avg > best_avg {
            best_avg = avg;
            best_arm = arm;
        }
    }

    best_arm
}

/// Run the Arena for `rounds` iterations per pattern type.
///
/// Returns `(bandit_total, static_total, random_total)`.
///
/// - **Bandit**: uses [`BanditPruner`] with the given [`BanditPolicy`].
/// - **Static**: always picks the best-average arm ([`best_average_arm`]).
/// - **Random**: cycles through arms deterministically `(round + pat_type) % arm_count`,
///   giving each arm exactly `rounds / arm_count` trials per pattern.
fn run_arena(policy: BanditPolicy, rounds: usize) -> (f64, f64, f64) {
    let patterns = arena_patterns();
    let arms = build_arena_arms();
    let arm_count = arms.len();

    let mut bandit = BanditPruner::new(arms, policy);

    let mut bandit_total = 0.0_f64;
    let mut static_total = 0.0_f64;
    let mut random_total = 0.0_f64;

    let static_arm = best_average_arm();

    for round in 0..rounds {
        for (pat_type, pattern) in patterns.iter().enumerate() {
            let (_arm, reward) = bandit.trial(pattern, |a| arena_reward(pat_type, a));
            bandit_total += reward;

            static_total += arena_reward(pat_type, static_arm);

            let random_arm = (round + pat_type) % arm_count;
            random_total += arena_reward(pat_type, random_arm);
        }
    }

    (bandit_total, static_total, random_total)
}

// ── Group 1: ScreeningPruner dispatch ──────────────────────────

#[test]
fn sudoku_pruner_dispatches_as_screening_pruner() {
    let pruner: Box<dyn ScreeningPruner> = Box::new(SudokuPruner::empty());
    assert_eq!(pruner.arm_label(), "sudoku");

    assert!(
        pruner.is_valid(0, 5, &[]),
        "digit 5 at depth 0 of empty grid should be valid"
    );
    let score = pruner.screen(0, 5, &[]);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "valid digit should screen 1.0, got {score}"
    );
}

#[test]
fn ngram_pruner_dispatches_as_screening_pruner() {
    let model = NgramDraftModel::new(2, 10);
    let pruner: Box<dyn ScreeningPruner> = Box::new(NgramScreeningPruner::new(model));
    assert_eq!(pruner.arm_label(), "ngram-screen");

    let score = pruner.screen(0, 5, &[]);
    let expected = 1.0 / 10.0;
    assert!(
        (score - expected).abs() < 1e-6,
        "untrained model should give uniform 1/V = {expected}, got {score}"
    );
}

#[test]
fn regex_pruner_dispatches_as_screening_pruner() {
    let pruner: Box<dyn ScreeningPruner> =
        Box::new(RegexPruner::from_pattern("[a-z]+").expect("valid regex"));
    assert_eq!(pruner.arm_label(), "regex");

    let token_a = b'a' as TokenId;
    assert!(
        pruner.is_valid(0, token_a, &[]),
        "token 'a' should be valid for [a-z]+"
    );
    let score = pruner.screen(0, token_a, &[]);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "valid token should screen 1.0, got {score}"
    );
}

#[test]
fn no_pruner_dispatches_as_screening_pruner() {
    let pruner: Box<dyn ScreeningPruner> = Box::new(NoPruner::new());
    assert_eq!(pruner.arm_label(), "no-pruner");

    assert!(pruner.is_valid(0, 999, &[]));
    let score = pruner.screen(42, 777, &[1, 2, 3]);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "NoPruner should always screen 1.0, got {score}"
    );
}

#[test]
fn three_distinct_arms_have_distinct_labels() {
    let arms = build_arena_arms();
    let labels: Vec<&str> = arms.iter().map(|a| a.arm_label()).collect();
    assert_eq!(labels.len(), 3);
    assert!(labels.iter().all(|l| !l.is_empty()));
    assert_eq!(labels[0], "sudoku");
    assert_eq!(labels[1], "ngram-screen");
    assert_eq!(labels[2], "regex");
}

// ── Group 2: Arena proof (bandit > static > random) ────────────

#[test]
fn arena_ucb1_bandit_beats_static_and_random() {
    let rounds = 300usize;
    let (bandit_total, static_total, random_total) = run_arena(BanditPolicy::ucb1(), rounds);

    assert!(
        bandit_total > static_total,
        "UCB1 bandit ({bandit_total:.1}) must beat static ({static_total:.1})"
    );
    assert!(
        static_total > random_total,
        "static ({static_total:.1}) must beat random ({random_total:.1})"
    );
}

#[test]
fn arena_epsilon_greedy_bandit_beats_static_and_random() {
    let rounds = 600usize;
    let (bandit_total, static_total, random_total) =
        run_arena(BanditPolicy::epsilon_greedy(), rounds);

    assert!(
        bandit_total > static_total,
        "ε-greedy bandit ({bandit_total:.1}) must beat static ({static_total:.1})"
    );
    assert!(
        static_total > random_total,
        "static ({static_total:.1}) must beat random ({random_total:.1})"
    );
}

#[test]
fn arena_thompson_bandit_beats_static_and_random() {
    let rounds = 600usize;
    let (bandit_total, static_total, random_total) = run_arena(BanditPolicy::thompson(), rounds);

    assert!(
        bandit_total > static_total,
        "Thompson bandit ({bandit_total:.1}) must beat static ({static_total:.1})"
    );
    assert!(
        static_total > random_total,
        "static ({static_total:.1}) must beat random ({random_total:.1})"
    );
}

#[test]
fn arena_optimal_arm_per_pattern_is_unique() {
    let mut optimal_arms = [0usize; ARENA_PATTERN_COUNT];
    for pat in 0..ARENA_PATTERN_COUNT {
        let mut best_arm = 0usize;
        let mut best_reward = f64::NEG_INFINITY;
        for (arm, &r) in ARENA_REWARD_TABLE[pat].iter().enumerate() {
            if r > best_reward {
                best_reward = r;
                best_arm = arm;
            }
        }
        optimal_arms[pat] = best_arm;
    }

    assert_eq!(
        optimal_arms,
        [0, 1, 2],
        "each pattern type must have a different optimal arm"
    );
}

// ── Group 3: Per-context learning ──────────────────────────────

#[test]
fn bandit_learns_per_context_optimal_arm() {
    let arms = build_arena_arms();
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let patterns = arena_patterns();

    for _ in 0..200 {
        for (pat_type, pattern) in patterns.iter().enumerate() {
            let _ = bandit.trial(pattern, |a| arena_reward(pat_type, a));
        }
    }

    let expected_arms = [0usize, 1, 2];
    for (pat_type, pattern) in patterns.iter().enumerate() {
        let selected = bandit.select_arm(pattern);
        assert_eq!(
            selected, expected_arms[pat_type],
            "pattern type {pat_type}: expected arm {}, got {selected}",
            expected_arms[pat_type]
        );
    }
}

#[test]
fn bandit_log_reflects_learning() {
    let arms = build_arena_arms();
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let patterns = arena_patterns();
    let rounds = 200usize;

    for _ in 0..rounds {
        for (pat_type, pattern) in patterns.iter().enumerate() {
            let _ = bandit.trial(pattern, |a| arena_reward(pat_type, a));
        }
    }

    for (pat_type, pattern) in patterns.iter().enumerate() {
        let log = bandit.log();
        let total_pulls = log.total_pulls_for_pattern(pattern);
        assert!(
            total_pulls >= rounds as u64,
            "pattern {pat_type}: expected >= {rounds} pulls, got {total_pulls}"
        );

        let best_arm = [0usize, 1, 2][pat_type];
        let best_stats = log
            .arm_stats(pattern, best_arm)
            .expect("best arm should have stats after training");
        let other_stats = log
            .arm_stats(pattern, (best_arm + 1) % 3)
            .expect("other arm should have stats");

        assert!(
            best_stats.pulls() > other_stats.pulls(),
            "pattern {pat_type}: best arm {best_arm} (pulls={}) should outweigh other arm (pulls={})",
            best_stats.pulls(),
            other_stats.pulls()
        );
    }
}

// ── Group 4: Decode loop integration ───────────────────────────

#[test]
fn bandit_wrapping_sudoku_pruner_solves_empty_grid() {
    let arms: Vec<Box<dyn ScreeningPruner>> = vec![Box::new(SudokuPruner::empty())];
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());

    let draft = UniformDraftModel::new(10);
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 500_000,
        seed: 42,
    };

    let result = speculative_decode(&draft, &mut bandit, &config);

    assert!(result.verified, "bandit must solve empty Sudoku grid");
    assert_eq!(result.tokens.len(), 81, "must fill all 81 cells");
}

#[test]
fn bandit_wrapping_sudoku_pruner_solves_medium_puzzle() {
    const MEDIUM_PUZZLE: &str = "
        5 3 . . 7 . . . .
        6 . . 1 9 5 . . .
        . 9 8 . . . . 6 .
        8 . . . 6 . . . 3
        4 . . 8 . 3 . . 1
        7 . . . 2 . . . 6
        . 6 . . . . 2 8 .
        . . . 4 1 9 . . 5
        . . . . 8 . . 7 9
    ";

    let pruner = SudokuPruner::from_puzzle(MEDIUM_PUZZLE).expect("valid puzzle");
    let arms: Vec<Box<dyn ScreeningPruner>> = vec![Box::new(pruner)];
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());

    let draft = UniformDraftModel::new(10);
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 500_000,
        seed: 42,
    };

    let result = speculative_decode(&draft, &mut bandit, &config);

    assert!(result.verified, "bandit must solve medium Sudoku puzzle");
    assert_eq!(result.tokens.len(), 81, "must fill all 81 cells");
}

#[test]
fn bandit_decode_result_has_audit_hash() {
    let arms: Vec<Box<dyn ScreeningPruner>> = vec![Box::new(SudokuPruner::empty())];
    let mut bandit_a = BanditPruner::new(arms, BanditPolicy::ucb1());

    let arms2: Vec<Box<dyn ScreeningPruner>> = vec![Box::new(SudokuPruner::empty())];
    let mut bandit_b = BanditPruner::new(arms2, BanditPolicy::ucb1());

    let draft = UniformDraftModel::new(10);
    let config = DecodeConfig {
        max_tokens: 81,
        top_k: 10,
        backtrack: true,
        max_attempts: 500_000,
        seed: 42,
    };

    let result_a = speculative_decode(&draft, &mut bandit_a, &config);
    let result_b = speculative_decode(&draft, &mut bandit_b, &config);

    assert_eq!(
        result_a.hash, result_b.hash,
        "same seed + same bandit config must produce same solution hash"
    );
    assert_ne!(
        result_a.hash, [0u8; 32],
        "hash must not be all zeros for a valid solution"
    );
}

// ── Group 5: AbsorbCompress integration ────────────────────────

#[test]
fn absorb_compress_locks_stable_high_q_arm() {
    let arms = build_arena_arms();
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let pattern = pattern_key(0, &[]);

    for _ in 0..50 {
        let _ = bandit.trial(&pattern, |a| match a {
            0 => 0.95,
            _ => 0.10,
        });
    }

    assert!(
        bandit.log().arms_for(&pattern).is_some(),
        "pattern should be in log before absorb"
    );

    let dropped = bandit.absorb_compress();

    assert_eq!(dropped, 1, "one pattern should be absorbed");
    assert!(
        bandit.log().arms_for(&pattern).is_none(),
        "absorbed pattern should be removed from log"
    );
}

#[test]
fn absorb_compress_preserves_unstable_patterns() {
    let arms = build_arena_arms();
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let pattern = pattern_key(0, &[]);

    for i in 0..20u32 {
        let _ = bandit.trial(&pattern, |a| match (a, i % 2) {
            (0, 0) => 1.0,
            (0, _) => 0.0,
            _ => 0.5,
        });
    }

    let dropped = bandit.absorb_compress();

    assert_eq!(dropped, 0, "high-variance pattern should not be absorbed");
    assert!(
        bandit.log().arms_for(&pattern).is_some(),
        "unstable pattern should remain in log"
    );
}

#[test]
fn absorb_compress_after_arena_run() {
    let arms = build_arena_arms();
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let patterns = arena_patterns();

    for _ in 0..100 {
        for (pat_type, pattern) in patterns.iter().enumerate() {
            let _ = bandit.trial(pattern, |a| arena_reward(pat_type, a));
        }
    }

    let log_before = bandit.log().pattern_count();
    assert_eq!(log_before, 3, "three patterns should be tracked");

    let dropped = bandit.absorb_compress();

    // With 100 rounds × 3 patterns = 300 pulls per pattern, each pattern's
    // optimal arm should be stable high-Q (reward ≥ 0.85, pulls ≥ 10).
    // All three patterns should be absorbed into hard locks.
    assert!(
        dropped >= 2,
        "at least 2 of 3 patterns should be absorbed, got {dropped}"
    );
    assert!(
        dropped <= 3,
        "at most 3 patterns can be absorbed, got {dropped}"
    );
}

// ── Group 6: decode-loop reward attribution ────────────────────
//
// Why this group exists (added 2026-08-19):
//
// Groups 2 and 3 prove the bandit LEARNS — but they drive it through
// `BanditPruner::trial`, a direct API that never enters the decode loop. That
// left the loop's wiring unproven, and it was in fact broken: `propagate`
// re-derived its arm from a pattern hashed over the POST-push token vector
// while `batch_is_valid` had hashed the prefix, so the reward, the
// `committed[]` blame record, and the forwarded `propagate` could all land on
// a different arm than the one that actually filtered the token (fixed in
// 021b13a). `AGENTS.md` §6 names this exact failure — "if selection differs
// between read and write, the bandit rewards an arm that didn't actually fire
// → silent corruption of the trial log" — yet every test in this file passed
// 17/17 both before and after the fix, so the gate certified nothing about it.
//
// These tests close that hole: they run through `speculative_decode` and
// assert the loop's attribution directly. Reintroduce the bug and they fail.

use std::sync::{Arc, Mutex};

/// Records which arm index served each hook, so read/write agreement can be
/// asserted from outside the bandit.
#[derive(Clone)]
struct AttributionSpy {
    index: usize,
    calls: Arc<Mutex<Vec<(&'static str, usize)>>>,
}

impl ns_engine::ConstraintPruner for AttributionSpy {
    fn is_valid(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> bool {
        true
    }

    fn batch_is_valid(&self, _d: usize, c: &[TokenId], _p: &[TokenId], r: &mut [bool]) {
        self.calls.lock().unwrap().push(("validate", self.index));
        let len = c.len().min(r.len());
        r[..len].fill(true);
    }

    fn manifold_score(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> f32 {
        1.0
    }

    fn propagate(&mut self, _d: usize, _t: TokenId, _p: &[TokenId]) {
        self.calls.lock().unwrap().push(("propagate", self.index));
    }
}

impl ScreeningPruner for AttributionSpy {
    fn arm_id(&self) -> ns_engine::ArmId {
        self.index as ns_engine::ArmId
    }

    fn arm_label(&self) -> &str {
        "attribution-spy"
    }

    fn screen(&self, _d: usize, _t: TokenId, _p: &[TokenId]) -> f32 {
        // Distinct screen scores so the two arms are not interchangeable to
        // the policy: arm 0 looks useless, arm 1 looks good.
        match self.index {
            0 => 0.0,
            _ => 1.0,
        }
    }
}

/// Draft with distinct, stable logits — a uniform model would let
/// `shuffle_tied_groups` randomise the order and blur what is being measured.
struct FixedDraft {
    vocab: usize,
}

impl ns_engine::DraftModel for FixedDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn log_probs(&self, _c: &[TokenId]) -> ns_engine::Logits {
        (0..self.vocab).map(|i| 1.0 - i as f32 * 0.1).collect()
    }
}

fn attribution_config(backtrack: bool) -> DecodeConfig {
    DecodeConfig {
        max_tokens: 4,
        top_k: 3,
        seed: 42,
        backtrack,
        max_attempts: 100,
    }
}

/// The arm that validates a token must be the arm that receives its
/// `propagate` — the read/write agreement `AGENTS.md` §6 requires.
#[test]
fn decode_loop_propagates_to_the_arm_that_validated() {
    for backtrack in [false, true] {
        let calls: Arc<Mutex<Vec<(&'static str, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let arms: Vec<Box<dyn ScreeningPruner>> = vec![
            Box::new(AttributionSpy {
                index: 0,
                calls: Arc::clone(&calls),
            }),
            Box::new(AttributionSpy {
                index: 1,
                calls: Arc::clone(&calls),
            }),
        ];
        let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
        let draft = FixedDraft { vocab: 3 };

        // Two runs: the first seeds the trial log, so the second exercises the
        // path where a pattern already has statistics. With the pre-fix code
        // the second run mismatches on every step.
        for _ in 0..2 {
            let _ = speculative_decode(&draft, &mut bandit, &attribution_config(backtrack));
        }

        let events = calls.lock().unwrap().clone();
        let pairs: Vec<(usize, usize)> = events
            .chunks(2)
            .filter(|w| w.len() == 2 && w[0].0 == "validate" && w[1].0 == "propagate")
            .map(|w| (w[0].1, w[1].1))
            .collect();

        assert!(
            !pairs.is_empty(),
            "backtrack={backtrack}: no validate/propagate pairs recorded"
        );
        let mismatched: Vec<_> = pairs.iter().filter(|(v, p)| v != p).collect();
        assert!(
            mismatched.is_empty(),
            "backtrack={backtrack}: propagate went to a different arm than validated: {mismatched:?}"
        );
    }
}

/// Driven through the decode loop, the bandit must actually LEARN — the
/// higher-screening arm should end up with more pulls than the useless one.
///
/// This is the Phase 2 claim ("converges to the optimal arm") asserted through
/// `speculative_decode` rather than through the direct `trial` API that groups
/// 2 and 3 use. It is load-bearing for the attribution bug in a way a
/// reward-value check is not: `propagate` computed its reward from the same
/// arm it had just mis-selected, so reward and arm stayed internally
/// consistent even when both were wrong. What the bug destroys is *learning* —
/// selection keyed on a post-push pattern that is fresh almost every time, so
/// the policy never accumulates evidence and keeps falling back to the first
/// untried arm.
#[test]
fn decode_loop_bandit_converges_to_the_better_arm() {
    let calls: Arc<Mutex<Vec<(&'static str, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let arms: Vec<Box<dyn ScreeningPruner>> = vec![
        Box::new(AttributionSpy {
            index: 0,
            calls: Arc::clone(&calls),
        }),
        Box::new(AttributionSpy {
            index: 1,
            calls: Arc::clone(&calls),
        }),
    ];
    let mut bandit = BanditPruner::new(arms, BanditPolicy::ucb1());
    let draft = FixedDraft { vocab: 3 };

    for _ in 0..40 {
        let _ = speculative_decode(&draft, &mut bandit, &attribution_config(true));
    }

    // Count which arm each depth's committed token actually went through.
    let events = calls.lock().unwrap().clone();
    let mut fired = [0usize; 2];
    for (hook, arm) in &events {
        if *hook == "propagate" {
            fired[*arm] += 1;
        }
    }

    assert!(
        fired[0] + fired[1] > 0,
        "no propagate calls recorded across 40 decode runs"
    );
    assert!(
        fired[1] > fired[0],
        "bandit should favour arm 1 (screen 1.0) over arm 0 (screen 0.0) once it has evidence, \
         but arm 0 fired {} times and arm 1 fired {} times — selection is not learning through \
         the decode loop",
        fired[0],
        fired[1]
    );
}
