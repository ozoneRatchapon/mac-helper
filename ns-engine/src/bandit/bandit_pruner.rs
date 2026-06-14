//! Multi-armed bandit pruner: wraps multiple [`ScreeningPruner`] arms and
//! learns which to fire per decode context.
//!
//! The bandit is the "speculation on policy" layer from katopz/katgpt-rs:
//! instead of always firing the same pruner, the bandit tracks per-context
//! reward statistics per arm and adapts its selection over time. Intelligence
//! evolves by updating the trial log, not by gradient descent.
//!
//! # Architecture
//!
//! ```text
//! (depth, parent_tokens)
//!     │
//!     ▼ BLAKE3 hash
//! PatternKey
//!     │
//!     ▼ policy.select(pattern, log)
//! arm_index ──► arms[arm].batch_is_valid / propagate / on_backtrack
//!     │
//!     ▼ reward observation
//! TrialLog.observe(pattern, arm, reward)
//! ```
//!
//! # Reward signal
//!
//! - `propagate(depth, token, ...)`: positive reward = the chosen arm's
//!   [`screen`](ScreeningPruner::screen) score for the accepted token. A
//!   confident, well-matched arm produces a high reward.
//! - `on_backtrack(depth, token, ...)`: negative reward `-1.0`. The arm
//!   that fired at `depth` chose a token whose subtree dead-ended.
//!
//! Over many decodes, the log accumulates per-pattern arm statistics. Policies
//! (UCB1, ε-greedy, Thompson) read the log to select arms that balance
//! exploitation (high mean) with exploration (under-sampled).
//!
//! # Determinism guarantee
//!
//! `batch_is_valid` takes `&self` and cannot mutate the log. Arm selection is
//! therefore deterministic from `(policy, pattern, log)`. The matching
//! `propagate` call (which takes `&mut self`) re-derives the same arm because
//! the log state is unchanged between `batch_is_valid` and `propagate`. The
//! `committed` Vec remembers the (pattern, arm) pair so `on_backtrack` doesn't
//! need to re-derive.
//!
//! # Direct API (Arena tests)
//!
//! The bandit exposes [`select_arm`](BanditPruner::select_arm),
//! [`observe`](BanditPruner::observe), and [`trial`](BanditPruner::trial) for
//! callers that want to drive arm selection without going through the decode
//! loop. This is how the Phase 2 Arena proves `HL (bandit) > static > random`.

use crate::bandit::policies::BanditPolicy;
use crate::bandit::trial_log::{pattern_key, PatternKey, TrialLog};
use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

/// Multi-armed bandit pruner.
///
/// Wraps a fixed-size vector of [`ScreeningPruner`] arms. Each
/// `(depth, parent_tokens)` context maps to a [`PatternKey`]; the bandit
/// maintains per-pattern, per-arm reward statistics and selects the next
/// arm via its [`BanditPolicy`].
///
/// Implements [`ConstraintPruner`] so it can be dropped into
/// [`speculative_decode`](crate::speculative_decode) in place of any other
/// pruner. Also implements [`ScreeningPruner`] for composability (a bandit
/// can itself be an arm of an outer bandit).
pub struct BanditPruner {
    /// Fixed-size vector of arm pruners. Arm index = position in this vec.
    arms: Vec<Box<dyn ScreeningPruner>>,
    /// Per-pattern, per-arm reward statistics. The bandit's memory.
    log: TrialLog,
    /// Arm-selection strategy.
    policy: BanditPolicy,
    /// `committed[d] = Some((pattern, arm))` if a token is currently placed
    /// at depth `d`. Used by `on_backtrack` to attribute negative reward to
    /// the arm that fired at the popped depth.
    ///
    /// Mutated only in `propagate` and `on_backtrack` (both `&mut self`).
    committed: Vec<Option<(PatternKey, usize)>>,
}

impl BanditPruner {
    /// Build a new bandit with the given arms and policy.
    ///
    /// # Panics
    ///
    /// Panics if `arms` is empty — a bandit with zero arms cannot make any
    /// decisions and would silently reject every candidate.
    pub fn new(arms: Vec<Box<dyn ScreeningPruner>>, policy: BanditPolicy) -> Self {
        assert!(!arms.is_empty(), "BanditPruner requires at least one arm");

        Self {
            arms,
            log: TrialLog::new(),
            policy,
            committed: Vec::new(),
        }
    }

    /// Number of arms in the bandit.
    pub fn arm_count(&self) -> usize {
        self.arms.len()
    }

    /// Read-only access to the trial log.
    pub fn log(&self) -> &TrialLog {
        &self.log
    }

    /// Read-only access to the bandit's policy.
    pub fn policy(&self) -> &BanditPolicy {
        &self.policy
    }

    /// Read-only access to the bandit's arms (for inspection / arena tests).
    pub fn arms(&self) -> &[Box<dyn ScreeningPruner>] {
        &self.arms
    }

    /// Human-readable label for arm at `index`, if it exists.
    pub fn arm_label(&self, index: usize) -> Option<&str> {
        self.arms.get(index).map(|arm| arm.arm_label())
    }

    /// Stable [`ArmId`] for arm at `index`, if it exists.
    pub fn arm_id(&self, index: usize) -> Option<ArmId> {
        self.arms.get(index).map(|arm| arm.arm_id())
    }

    /// Select an arm for `pattern` using the current policy and log state.
    ///
    /// Pure read: does NOT observe a reward. Use [`observe`](Self::observe)
    /// or [`trial`](Self::trial) to record the outcome.
    ///
    /// Returns the selected arm index in `[0, arm_count)`.
    pub fn select_arm(&self, pattern: &PatternKey) -> usize {
        self.policy.select(pattern, &self.log, self.arms.len())
    }

    /// Record a reward observation for `(pattern, arm)`.
    ///
    /// Used by Arena tests to drive the bandit with custom reward functions.
    /// Also called internally by `propagate` and `on_backtrack`.
    pub fn observe(&mut self, pattern: &PatternKey, arm: usize, reward: f64) {
        self.log.observe(pattern, arm, reward);
    }

    /// Convenience for Arena tests: select an arm, compute reward via
    /// `reward_fn(arm)`, observe it, return `(arm, reward)`.
    ///
    /// ```ignore
    /// // 3 arms; arm 2 has the highest payout.
    /// let (arm, reward) = bandit.trial(&pattern, |i| match i {
    ///     0 => 0.2,
    ///     1 => 0.5,
    ///     _ => 0.9,
    /// });
    /// ```
    pub fn trial(
        &mut self,
        pattern: &PatternKey,
        reward_fn: impl Fn(usize) -> f64,
    ) -> (usize, f64) {
        let arm = self.select_arm(pattern);
        let reward = reward_fn(arm);
        self.observe(pattern, arm, reward);
        (arm, reward)
    }

    /// Which arm fired at `depth`, if a token is currently committed there.
    ///
    /// Returns `None` if no token has been committed at `depth`, or if the
    /// depth has been backtracked and cleared.
    pub fn committed_arm_at(&self, depth: usize) -> Option<usize> {
        self.committed
            .get(depth)
            .copied()
            .flatten()
            .map(|(_, arm)| arm)
    }

    /// Reset the bandit to its post-construction state.
    ///
    /// Clears the trial log and committed vector. Arms and policy are kept.
    pub fn reset(&mut self) {
        self.log.clear();
        self.committed.clear();
    }

    /// Total cumulative reward across all patterns and arms.
    ///
    /// Convenience for Arena tests measuring aggregate performance.
    pub fn total_reward(&self) -> f64 {
        self.log.total_reward()
    }

    /// Total pulls across all patterns and arms.
    pub fn total_pulls(&self) -> u64 {
        self.log.total_pulls()
    }

    /// Run the absorb-compress pass: scan the log for stable arms and lock
    /// in their decisions as hard rules. See [`AbsorbCompress`](super::absorb).
    ///
    /// Returns the number of patterns that gained a hard rule.
    pub fn absorb_compress(&mut self) -> usize {
        crate::bandit::absorb::absorb_compress(&mut self.log, &self.arms)
    }

    // ── Internals ───────────────────────────────────────────────

    /// Ensure `committed` has at least `depth + 1` slots, padding with `None`.
    fn ensure_committed_capacity(&mut self, depth: usize) {
        if self.committed.len() <= depth {
            self.committed.resize(depth + 1, None);
        }
    }

    /// Compute the prefix slice for a depth given the post-push token vec.
    ///
    /// `parent_tokens` passed to `propagate` includes the just-pushed token;
    /// the prefix for the pattern hash is `[0..depth)` (everything before).
    fn prefix_at_depth(depth: usize, parent_tokens: &[TokenId]) -> &[TokenId] {
        let end = depth.min(parent_tokens.len());
        &parent_tokens[..end]
    }
}

// ── ConstraintPruner impl ──────────────────────────────────────

impl ConstraintPruner for BanditPruner {
    fn is_valid(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        match self.arms.first() {
            None => false, // defensive: constructor forbids empty arms
            Some(_) => {
                let pat = pattern_key(depth, parent_tokens);
                let arm = self.policy.select(&pat, &self.log, self.arms.len());
                self.arms[arm].is_valid(depth, token, parent_tokens)
            }
        }
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        match self.arms.first() {
            None => {
                // Defensive: reject everything when no arms exist.
                let len = candidates.len().min(results.len());
                results[..len].fill(false);
            }
            Some(_) => {
                let pat = pattern_key(depth, parent_tokens);
                let arm = self.policy.select(&pat, &self.log, self.arms.len());
                self.arms[arm].batch_is_valid(depth, candidates, parent_tokens, results);
            }
        }
    }

    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        match self.arms.first() {
            None => 0.0,
            Some(_) => {
                let pat = pattern_key(depth, parent_tokens);
                let arm = self.policy.select(&pat, &self.log, self.arms.len());
                self.arms[arm].manifold_score(depth, token, parent_tokens)
            }
        }
    }

    fn propagate(&mut self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) {
        let arm = {
            // Re-derive the same arm the (immutable) batch_is_valid picked.
            // Log state is identical between batch_is_valid and propagate.
            let pat = pattern_key(depth, parent_tokens);
            self.policy.select(&pat, &self.log, self.arms.len())
        };

        // Compute positive reward = the chosen arm's screen score for this
        // token. Different arms have different screen functions, so the bandit
        // learns which arm "liked" the accepted token more.
        let prefix = Self::prefix_at_depth(depth, parent_tokens);
        let pat = pattern_key(depth, prefix);
        let reward = self.arms[arm].screen(depth, token, prefix) as f64;

        self.log.observe(&pat, arm, reward);

        // Remember which arm fired here so on_backtrack can attribute blame.
        self.ensure_committed_capacity(depth);
        self.committed[depth] = Some((pat, arm));

        // Forward the propagate to the chosen arm.
        self.arms[arm].propagate(depth, token, parent_tokens);
    }

    fn on_backtrack(&mut self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) {
        // Look up the arm that fired at `depth`. If the depth was never
        // committed (e.g., the bandit pruner was swapped in mid-decode),
        // there is nothing to attribute.
        let entry = match self.committed.get_mut(depth) {
            Some(slot) => slot.take(),
            None => None,
        };

        let arm = match entry {
            Some((pattern, arm)) => {
                // Negative reward: this arm's choice led to a subtree dead-end.
                self.log.observe(&pattern, arm, -1.0);
                arm
            }
            None => {
                // No committed arm — still forward to a default arm for
                // state cleanup. Use arm 0 (always exists by construction).
                0
            }
        };

        // Forward the backtrack to the chosen arm so it can undo state.
        if arm < self.arms.len() {
            self.arms[arm].on_backtrack(depth, token, parent_tokens);
        }
    }
}

// ── ScreeningPruner impl (bandit can be an arm of an outer bandit) ──

impl ScreeningPruner for BanditPruner {
    fn arm_id(&self) -> ArmId {
        // Stable ID for the bandit itself. The outer bandit tracks inner
        // arms by index, so this is mostly for diagnostics.
        ArmId::MAX
    }

    fn arm_label(&self) -> &str {
        "bandit"
    }

    fn screen(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        let pat = pattern_key(depth, parent_tokens);
        let arm = self.policy.select(&pat, &self.log, self.arms.len());
        self.arms[arm].screen(depth, token, parent_tokens)
    }

    fn batch_screen(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [f32],
    ) {
        let pat = pattern_key(depth, parent_tokens);
        let arm = self.policy.select(&pat, &self.log, self.arms.len());
        self.arms[arm].batch_screen(depth, candidates, parent_tokens, results);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandit::trial_log::pattern_key;
    use crate::pruners::NoPruner;
    use crate::types::{DecodeConfig, Logits};

    /// Stub draft model for tests that don't care about logits.
    struct StubDraft {
        vocab: usize,
    }
    impl crate::traits::DraftModel for StubDraft {
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn log_probs(&self, _ctx: &[TokenId]) -> Logits {
            vec![0.0; self.vocab]
        }
    }

    /// Build a bandit with N NoPruner arms — every arm accepts every token,
    /// so we can isolate the bandit logic.
    fn uniform_bandit(n: usize, policy: BanditPolicy) -> BanditPruner {
        let arms: Vec<Box<dyn ScreeningPruner>> = (0..n)
            .map(|i| {
                let boxed: Box<dyn ScreeningPruner> = Box::new(LabeledNoPruner {
                    id: i as ArmId,
                    label: format!("arm-{i}"),
                });
                boxed
            })
            .collect();
        BanditPruner::new(arms, policy)
    }

    /// NoPruner with configurable arm_id and label.
    struct LabeledNoPruner {
        id: ArmId,
        label: String,
    }
    impl ConstraintPruner for LabeledNoPruner {
        fn is_valid(&self, _: usize, _: TokenId, _: &[TokenId]) -> bool {
            true
        }
        fn batch_is_valid(&self, _: usize, c: &[TokenId], _: &[TokenId], r: &mut [bool]) {
            let len = c.len().min(r.len());
            r[..len].fill(true);
        }
        fn manifold_score(&self, _: usize, _: TokenId, _: &[TokenId]) -> f32 {
            1.0
        }
    }
    impl ScreeningPruner for LabeledNoPruner {
        fn arm_id(&self) -> ArmId {
            self.id
        }
        fn arm_label(&self) -> &str {
            &self.label
        }
        fn screen(&self, _: usize, _: TokenId, _: &[TokenId]) -> f32 {
            1.0
        }
    }

    // ── Construction ──

    #[test]
    fn test_new_requires_nonempty_arms() {
        let result = std::panic::catch_unwind(|| {
            BanditPruner::new(Vec::new(), BanditPolicy::ucb1());
        });
        assert!(result.is_err(), "empty arms must panic");
    }

    #[test]
    fn test_arm_count_matches_input() {
        let bandit = uniform_bandit(3, BanditPolicy::ucb1());
        assert_eq!(bandit.arm_count(), 3);
    }

    #[test]
    fn test_arm_label_lookup() {
        let bandit = uniform_bandit(3, BanditPolicy::ucb1());
        assert_eq!(bandit.arm_label(0), Some("arm-0"));
        assert_eq!(bandit.arm_label(2), Some("arm-2"));
        assert_eq!(bandit.arm_label(5), None);
    }

    #[test]
    fn test_arm_id_lookup() {
        let bandit = uniform_bandit(3, BanditPolicy::ucb1());
        assert_eq!(bandit.arm_id(0), Some(0));
        assert_eq!(bandit.arm_id(2), Some(2));
        assert_eq!(bandit.arm_id(5), None);
    }

    // ── Direct API (Arena-style) ──

    #[test]
    fn test_select_arm_returns_untried_first() {
        let bandit = uniform_bandit(3, BanditPolicy::ucb1());
        let pattern = pattern_key(0, &[]);
        assert_eq!(bandit.select_arm(&pattern), 0);
    }

    #[test]
    fn test_observe_updates_log() {
        let mut bandit = uniform_bandit(3, BanditPolicy::ucb1());
        let pattern = pattern_key(0, &[]);

        bandit.observe(&pattern, 0, 0.5);
        assert_eq!(bandit.total_pulls(), 1);
        assert!((bandit.total_reward() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_trial_returns_arm_and_reward() {
        let mut bandit = uniform_bandit(3, BanditPolicy::ucb1());
        let pattern = pattern_key(0, &[]);

        // First trial: arm 0 (untried), reward_fn(0) = 0.7.
        let (arm, reward) = bandit.trial(&pattern, |i| match i {
            0 => 0.7,
            1 => 0.4,
            _ => 0.1,
        });
        assert_eq!(arm, 0);
        assert!((reward - 0.7).abs() < 1e-9);
        assert_eq!(bandit.total_pulls(), 1);
    }

    #[test]
    fn test_trial_many_rounds_learns_best_arm() {
        let mut bandit = uniform_bandit(3, BanditPolicy::ucb1());
        let pattern = pattern_key(0, &[]);

        // Reward function: arm 1 is clearly best.
        // After many trials, UCB1 should converge to arm 1.
        let mut arm_1_count = 0usize;
        for _ in 0..200 {
            let (arm, _) = bandit.trial(&pattern, |i| match i {
                0 => 0.2,
                1 => 0.9,
                _ => 0.1,
            });
            if arm == 1 {
                arm_1_count += 1;
            }
        }
        assert!(
            arm_1_count >= 150,
            "UCB1 should converge to arm 1, got {arm_1_count}/200"
        );
    }

    #[test]
    fn test_reset_clears_log_and_committed() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());
        let pattern = pattern_key(0, &[]);

        bandit.observe(&pattern, 0, 1.0);
        bandit.observe(&pattern, 1, 1.0);
        assert_eq!(bandit.total_pulls(), 2);

        bandit.reset();
        assert_eq!(bandit.total_pulls(), 0);
        assert_eq!(bandit.total_reward(), 0.0);
        assert_eq!(bandit.committed_arm_at(0), None);
    }

    // ── ConstraintPruner delegation ──

    #[test]
    fn test_is_valid_delegates_to_selected_arm() {
        let bandit = uniform_bandit(3, BanditPolicy::ucb1());
        // All arms are LabeledNoPruner → always valid.
        assert!(bandit.is_valid(0, 5, &[]));
        assert!(bandit.is_valid(3, 99, &[1, 2, 3]));
    }

    #[test]
    fn test_batch_is_valid_delegates() {
        let bandit = uniform_bandit(3, BanditPolicy::ucb1());
        let candidates = vec![0, 1, 2, 3];
        let mut results = vec![false; 4];
        bandit.batch_is_valid(0, &candidates, &[], &mut results);
        assert_eq!(results, vec![true, true, true, true]);
    }

    #[test]
    fn test_manifold_score_delegates() {
        let bandit = uniform_bandit(2, BanditPolicy::ucb1());
        let score = bandit.manifold_score(0, 1, &[]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    // ── propagate / on_backtrack reward attribution ──

    #[test]
    fn test_propagate_records_positive_reward() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());

        // Simulate: arm 0 fired, token 5 accepted at depth 0.
        bandit.propagate(0, 5, &[5]);

        // LabeledNoPruner.screen returns 1.0 → reward = 1.0.
        assert_eq!(bandit.total_pulls(), 1);
        assert!((bandit.total_reward() - 1.0).abs() < 1e-9);
        assert_eq!(bandit.committed_arm_at(0), Some(0));
    }

    #[test]
    fn test_on_backtrack_records_negative_reward() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());

        // Propagate (positive), then backtrack (negative).
        bandit.propagate(0, 5, &[5]);
        bandit.on_backtrack(0, 5, &[]);

        // Net reward: 1.0 + (-1.0) = 0.0, pulls = 2.
        assert_eq!(bandit.total_pulls(), 2);
        assert!((bandit.total_reward() - 0.0).abs() < 1e-9);
        assert_eq!(
            bandit.committed_arm_at(0),
            None,
            "backtrack must clear committed"
        );
    }

    #[test]
    fn test_on_backtrack_uncommitted_depth_is_safe() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());

        // Backtrack a depth that was never committed. Should not panic.
        bandit.on_backtrack(99, 7, &[]);
        assert_eq!(bandit.total_pulls(), 0);
    }

    #[test]
    fn test_propagate_then_backtrack_pattern_matches() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());

        // Propagate at depth 1 with parent [3].
        bandit.propagate(1, 7, &[3, 7]);

        // Verify committed records the right depth.
        assert_eq!(bandit.committed_arm_at(1), Some(0));

        // Backtrack — pops token 7 from depth 1, parent_tokens = [3].
        bandit.on_backtrack(1, 7, &[3]);

        // The negative reward is attributed to the same (pattern, arm) pair.
        let pattern = pattern_key(1, &[3]);
        let stats = bandit
            .log()
            .arm_stats(&pattern, 0)
            .expect("arm 0 stats should exist for pattern");
        assert_eq!(stats.pulls(), 2);
        assert!(
            (stats.total_reward() - 0.0).abs() < 1e-9,
            "propagate +1.0, backtrack -1.0 → net 0"
        );
    }

    // ── Multi-depth propagation ──

    #[test]
    fn test_propagate_multiple_depths() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());

        bandit.propagate(0, 1, &[1]);
        bandit.propagate(1, 2, &[1, 2]);
        bandit.propagate(2, 3, &[1, 2, 3]);

        // Each (depth, prefix) pair hashes to a UNIQUE pattern key. UCB1
        // selects the lowest-index untried arm per pattern, so all three
        // depths pick arm 0 (each pattern is new — arm 0 is untried for it).
        assert_eq!(bandit.committed_arm_at(0), Some(0));
        assert_eq!(bandit.committed_arm_at(1), Some(0));
        assert_eq!(bandit.committed_arm_at(2), Some(0));
    }

    #[test]
    fn test_backtrack_clears_correct_depth() {
        let mut bandit = uniform_bandit(2, BanditPolicy::ucb1());

        bandit.propagate(0, 1, &[1]);
        bandit.propagate(1, 2, &[1, 2]);

        // Backtrack depth 1 only.
        bandit.on_backtrack(1, 2, &[1]);

        assert_eq!(bandit.committed_arm_at(0), Some(0), "depth 0 should remain");
        assert_eq!(
            bandit.committed_arm_at(1),
            None,
            "depth 1 should be cleared"
        );
    }

    // ── ScreeningPruner impl (bandit-as-arm) ──

    #[test]
    fn test_bandit_implements_screening_pruner() {
        let bandit = uniform_bandit(2, BanditPolicy::ucb1());
        // Disambiguate from the inherent `arm_id(index)` / `arm_label(index)`
        // lookup methods — call the ScreeningPruner trait methods directly.
        assert_eq!(ScreeningPruner::arm_id(&bandit), ArmId::MAX);
        assert_eq!(ScreeningPruner::arm_label(&bandit), "bandit");
    }

    #[test]
    fn test_bandit_screen_delegates_to_selected_arm() {
        let bandit = uniform_bandit(2, BanditPolicy::ucb1());
        let score = bandit.screen(0, 5, &[]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bandit_batch_screen_delegates() {
        let bandit = uniform_bandit(2, BanditPolicy::ucb1());
        let candidates = vec![0, 1, 2];
        let mut results = vec![0.0; 3];
        bandit.batch_screen(0, &candidates, &[], &mut results);
        for r in &results {
            assert!((r - 1.0).abs() < 1e-6);
        }
    }

    // ── Integration with speculative_decode ──

    #[test]
    fn test_bandit_works_in_decode_loop() {
        use crate::decode::speculative_decode;

        let mut bandit = uniform_bandit(1, BanditPolicy::ucb1());
        let draft = StubDraft { vocab: 5 };
        let config = DecodeConfig {
            max_tokens: 4,
            top_k: 5,
            backtrack: true,
            max_attempts: 100,
            seed: 42,
        };

        let result = speculative_decode(&draft, &mut bandit, &config);

        // All arms are NoPruner → all candidates valid → trivially solves.
        assert!(result.verified);
        assert_eq!(result.tokens.len(), 4);
    }

    // ── Determinism ──

    #[test]
    fn test_bandit_deterministic_given_same_log() {
        // Two bandits with identical log state must select the same arm.
        let bandit_a = uniform_bandit(3, BanditPolicy::ucb1());
        let bandit_b = uniform_bandit(3, BanditPolicy::ucb1());
        let pattern = pattern_key(0, &[]);

        let a = bandit_a.select_arm(&pattern);
        let b = bandit_b.select_arm(&pattern);
        assert_eq!(a, b);
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BanditPruner>();
    }

    #[test]
    fn test_default_no_pruner_works_as_arm() {
        // Verify NoPruner (the existing baseline pruner) can be adapted.
        // We wrap it via LabeledNoPruner above; here we just sanity-check
        // that NoPruner itself has ScreeningPruner implemented (elsewhere).
        let pruner = NoPruner::new();
        // NoPruner implements ScreeningPruner via blanket impl in pruners/mod.
        let _: &dyn ScreeningPruner = &pruner;
    }
}
