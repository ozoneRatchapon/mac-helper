//! Trial log for multi-armed bandit reward tracking.
//!
//! Each (depth, parent_tokens) context is hashed via BLAKE3 into a
//! [`PatternKey`]. The log stores per-arm statistics per pattern, allowing
//! the bandit to learn context-dependent arm selection.
//!
//! The log is the bandit's memory: pulls, rewards, and variance. Policies
//! (UCB1, ε-greedy, Thompson) read the log to select arms; `BanditPruner`
//! writes to the log on `propagate` (positive reward) and `on_backtrack`
//! (negative reward).

use blake3::Hasher;
use std::collections::HashMap;

use crate::types::TokenId;

/// BLAKE3 hash of a decode context (depth + prefix tokens).
///
/// Used as a stable key for per-context arm statistics. Same context → same
/// key → same arm tracking across calls.
pub type PatternKey = [u8; 32];

/// Compute the pattern key for a decode context.
///
/// `depth` and `parent_tokens` together uniquely identify a decision point
/// in the decode tree. The hash is deterministic: same inputs → same key.
///
/// Zero-allocation via streaming BLAKE3 hasher.
pub fn pattern_key(depth: usize, parent_tokens: &[TokenId]) -> PatternKey {
    let mut hasher = Hasher::new();
    hasher.update(&(depth as u64).to_le_bytes());
    for t in parent_tokens {
        hasher.update(&t.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Per-arm statistics for a single pattern.
///
/// Tracks pulls, cumulative reward, and sum of squared rewards. The latter
/// enables O(1) variance computation for Thompson sampling's posterior
/// update.
#[derive(Clone, Debug, Default)]
pub struct ArmStats {
    pulls: u64,
    total_reward: f64,
    sum_squares: f64,
}

impl ArmStats {
    /// Record a reward observation.
    ///
    /// Pulls increment by 1; total_reward and sum_squares accumulate the
    /// reward and its square (for variance computation).
    pub fn observe(&mut self, reward: f64) {
        self.pulls += 1;
        self.total_reward += reward;
        self.sum_squares += reward * reward;
    }

    /// Number of pulls for this arm in this pattern.
    pub fn pulls(&self) -> u64 {
        self.pulls
    }

    /// Cumulative reward across all pulls.
    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }

    /// Sample mean reward. Returns 0.0 for unpulled arms.
    pub fn mean(&self) -> f64 {
        match self.pulls {
            0 => 0.0,
            n => self.total_reward / n as f64,
        }
    }

    /// Population variance of observed rewards.
    ///
    /// Uses σ² = E[X²] - E[X]². Returns 0.0 for < 2 pulls (no information).
    /// Used by Thompson sampling to set posterior uncertainty.
    pub fn variance(&self) -> f64 {
        if self.pulls < 2 {
            return 0.0;
        }
        let n = self.pulls as f64;
        let mean = self.mean();
        let mean_of_squares = self.sum_squares / n;
        let variance = mean_of_squares - mean * mean;
        // Guard against tiny negative variance from floating-point error.
        variance.max(0.0)
    }

    /// Standard deviation of observed rewards. Returns 0.0 for < 2 pulls.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Reset to default (zero pulls, zero reward).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Trial log: per-pattern, per-arm reward statistics.
///
/// The bandit's memory. Each pattern key maps to a vector of arm statistics,
/// indexed by the arm's position in the [`BanditPruner`](super::BanditPruner).
/// Arms are lazily resized as observations arrive.
#[derive(Clone, Debug, Default)]
pub struct TrialLog {
    /// `stats.get(pattern) = vec![ArmStats; arm_count]`.
    stats: HashMap<PatternKey, Vec<ArmStats>>,
}

impl TrialLog {
    /// Create an empty trial log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a reward observation for `(pattern, arm)`.
    ///
    /// Lazily grows the per-pattern arm vector if `arm` exceeds its length.
    pub fn observe(&mut self, pattern: &PatternKey, arm: usize, reward: f64) {
        let arms = self.stats.entry(*pattern).or_default();
        if arm >= arms.len() {
            arms.resize(arm + 1, ArmStats::default());
        }
        arms[arm].observe(reward);
    }

    /// Read per-arm statistics for a pattern. Returns None for unseen patterns.
    pub fn arms_for(&self, pattern: &PatternKey) -> Option<&[ArmStats]> {
        self.stats.get(pattern).map(Vec::as_slice)
    }

    /// Read statistics for a specific `(pattern, arm)` pair.
    pub fn arm_stats(&self, pattern: &PatternKey, arm: usize) -> Option<&ArmStats> {
        self.stats.get(pattern).and_then(|arms| arms.get(arm))
    }

    /// Iterate over all observed pattern keys.
    pub fn patterns(&self) -> impl Iterator<Item = &PatternKey> {
        self.stats.keys()
    }

    /// Total pulls across all arms for a single pattern.
    ///
    /// Used by UCB1 (the `N` in the exploration bonus) and as a seed source
    /// for stochastic policies (ε-greedy, Thompson).
    pub fn total_pulls_for_pattern(&self, pattern: &PatternKey) -> u64 {
        self.stats
            .get(pattern)
            .map(|arms| arms.iter().map(|a| a.pulls()).sum())
            .unwrap_or(0)
    }

    /// Total pulls across all arms across all patterns.
    pub fn total_pulls(&self) -> u64 {
        self.stats
            .values()
            .flat_map(|arms| arms.iter().map(|a| a.pulls()))
            .sum()
    }

    /// Total reward across all arms across all patterns.
    pub fn total_reward(&self) -> f64 {
        self.stats
            .values()
            .flat_map(|arms| arms.iter().map(|a| a.total_reward()))
            .sum()
    }

    /// Number of distinct patterns observed.
    pub fn pattern_count(&self) -> usize {
        self.stats.len()
    }

    /// Maximum arm index seen across all patterns.
    ///
    /// Used by policies to size the candidate arm range when a pattern is
    /// unseen (the bandit's arm count is fixed across patterns).
    pub fn max_arm_index(&self) -> Option<usize> {
        self.stats
            .values()
            .map(|arms| arms.len().saturating_sub(1))
            .max()
    }

    /// Mean reward for `(pattern, arm)`. Returns None if no data.
    pub fn arm_mean(&self, pattern: &PatternKey, arm: usize) -> Option<f64> {
        self.arm_stats(pattern, arm).map(|s| s.mean())
    }

    /// Clear all log entries. The log returns to its initial empty state.
    pub fn clear(&mut self) {
        self.stats.clear();
    }

    /// Remove a pattern entirely. Returns true if the pattern was present.
    pub fn drop_pattern(&mut self, pattern: &PatternKey) -> bool {
        self.stats.remove(pattern).is_some()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_key_deterministic() {
        let depth = 3usize;
        let tokens = vec![1u32, 2, 3];
        let key_a = pattern_key(depth, &tokens);
        let key_b = pattern_key(depth, &tokens);
        assert_eq!(key_a, key_b, "same inputs must produce same key");
    }

    #[test]
    fn test_pattern_key_differs_on_depth() {
        let tokens = vec![1u32, 2, 3];
        let key_a = pattern_key(2, &tokens);
        let key_b = pattern_key(3, &tokens);
        assert_ne!(key_a, key_b, "different depths must produce different keys");
    }

    #[test]
    fn test_pattern_key_differs_on_tokens() {
        let a = pattern_key(3, &[1u32, 2, 3]);
        let b = pattern_key(3, &[1u32, 2, 4]);
        assert_ne!(a, b, "different tokens must produce different keys");
    }

    #[test]
    fn test_pattern_key_differs_on_token_count() {
        let a = pattern_key(3, &[1u32, 2, 3]);
        let b = pattern_key(3, &[1u32, 2, 3, 4]);
        assert_ne!(a, b, "different token counts must produce different keys");
    }

    #[test]
    fn test_pattern_key_empty_prefix() {
        // Depth 0, empty prefix — the root of the decode tree.
        let key = pattern_key(0, &[]);
        assert_ne!(key, [0u8; 32], "hash must not be all zeros");
    }

    #[test]
    fn test_arm_stats_starts_empty() {
        let stats = ArmStats::default();
        assert_eq!(stats.pulls(), 0);
        assert_eq!(stats.total_reward(), 0.0);
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.variance(), 0.0);
    }

    #[test]
    fn test_arm_stats_observe_accumulates() {
        let mut stats = ArmStats::default();
        stats.observe(1.0);
        assert_eq!(stats.pulls(), 1);
        assert!((stats.total_reward() - 1.0).abs() < 1e-9);
        assert!((stats.mean() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_arm_stats_observe_multiple() {
        let mut stats = ArmStats::default();
        stats.observe(0.5);
        stats.observe(1.5);
        assert_eq!(stats.pulls(), 2);
        assert!((stats.total_reward() - 2.0).abs() < 1e-9);
        assert!((stats.mean() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_arm_stats_variance_constant() {
        let mut stats = ArmStats::default();
        for _ in 0..5 {
            stats.observe(1.0);
        }
        assert_eq!(stats.pulls(), 5);
        assert!((stats.mean() - 1.0).abs() < 1e-9);
        assert!(
            stats.variance() < 1e-9,
            "constant observations → zero variance"
        );
    }

    #[test]
    fn test_arm_stats_variance_known() {
        // Observations: 1, 2, 3 → mean = 2, variance = ((-1)^2 + 0^2 + 1^2)/3 = 2/3
        let mut stats = ArmStats::default();
        stats.observe(1.0);
        stats.observe(2.0);
        stats.observe(3.0);
        assert!((stats.mean() - 2.0).abs() < 1e-9);
        let variance = stats.variance();
        let expected = 2.0 / 3.0;
        assert!(
            (variance - expected).abs() < 1e-9,
            "variance {variance} != expected {expected}"
        );
    }

    #[test]
    fn test_arm_stats_std_dev_matches_variance() {
        let mut stats = ArmStats::default();
        stats.observe(0.0);
        stats.observe(2.0);
        let std_dev = stats.std_dev();
        let variance = stats.variance();
        assert!((std_dev * std_dev - variance).abs() < 1e-9);
    }

    #[test]
    fn test_arm_stats_variance_unpulled_zero() {
        let stats = ArmStats::default();
        assert_eq!(stats.variance(), 0.0);
    }

    #[test]
    fn test_arm_stats_variance_single_pull_zero() {
        let mut stats = ArmStats::default();
        stats.observe(5.0);
        assert_eq!(stats.variance(), 0.0);
    }

    #[test]
    fn test_arm_stats_reset() {
        let mut stats = ArmStats::default();
        stats.observe(1.0);
        stats.observe(2.0);
        stats.reset();
        assert_eq!(stats.pulls(), 0);
        assert_eq!(stats.total_reward(), 0.0);
    }

    #[test]
    fn test_arm_stats_negative_reward() {
        let mut stats = ArmStats::default();
        stats.observe(-1.0);
        stats.observe(1.0);
        assert_eq!(stats.pulls(), 2);
        assert!((stats.total_reward() - 0.0).abs() < 1e-9);
        assert!((stats.mean() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_trial_log_starts_empty() {
        let log = TrialLog::new();
        assert_eq!(log.pattern_count(), 0);
        assert_eq!(log.total_pulls(), 0);
        assert_eq!(log.total_reward(), 0.0);
    }

    #[test]
    fn test_trial_log_observe_and_read() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 1.0);
        log.observe(&pattern, 1, 0.5);

        let arm0 = log.arm_stats(&pattern, 0).expect("arm 0 should exist");
        let arm1 = log.arm_stats(&pattern, 1).expect("arm 1 should exist");

        assert_eq!(arm0.pulls(), 1);
        assert!((arm0.mean() - 1.0).abs() < 1e-9);
        assert_eq!(arm1.pulls(), 1);
        assert!((arm1.mean() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_trial_log_observe_grows_arm_vector() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // First observe at index 5 — arm vector must resize.
        log.observe(&pattern, 5, 1.0);

        let arms = log.arms_for(&pattern).expect("pattern should exist");
        assert_eq!(arms.len(), 6);
        assert_eq!(arms[5].pulls(), 1);
        // Arms 0..5 are default (zero pulls).
        for (i, arm) in arms.iter().enumerate().take(5) {
            assert_eq!(arm.pulls(), 0, "arm {i} should be untouched");
        }
    }

    #[test]
    fn test_trial_log_total_pulls_for_pattern() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        log.observe(&pattern_a, 0, 1.0);
        log.observe(&pattern_a, 1, 0.5);
        log.observe(&pattern_b, 0, 1.0);

        assert_eq!(log.total_pulls_for_pattern(&pattern_a), 2);
        assert_eq!(log.total_pulls_for_pattern(&pattern_b), 1);
    }

    #[test]
    fn test_trial_log_total_pulls_for_unseen_pattern() {
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        assert_eq!(log.total_pulls_for_pattern(&pattern), 0);
    }

    #[test]
    fn test_trial_log_total_pulls_across_patterns() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        log.observe(&pattern_a, 0, 1.0);
        log.observe(&pattern_a, 0, 1.0);
        log.observe(&pattern_b, 0, 0.5);

        assert_eq!(log.total_pulls(), 3);
    }

    #[test]
    fn test_trial_log_total_reward_across_patterns() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        log.observe(&pattern_a, 0, 1.0);
        log.observe(&pattern_a, 0, 0.5);
        log.observe(&pattern_b, 0, 2.0);

        assert!((log.total_reward() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_trial_log_arm_mean() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 1.0);
        log.observe(&pattern, 0, 0.5);
        log.observe(&pattern, 0, 0.5);

        let mean = log.arm_mean(&pattern, 0).expect("arm 0 should have data");
        let expected = (1.0 + 0.5 + 0.5) / 3.0;
        assert!((mean - expected).abs() < 1e-9);
    }

    #[test]
    fn test_trial_log_arm_mean_no_data() {
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        assert!(log.arm_mean(&pattern, 0).is_none());
    }

    #[test]
    fn test_trial_log_clear() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        log.observe(&pattern, 0, 1.0);
        assert_eq!(log.pattern_count(), 1);

        log.clear();
        assert_eq!(log.pattern_count(), 0);
        assert_eq!(log.total_pulls(), 0);
    }

    #[test]
    fn test_trial_log_drop_pattern() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        log.observe(&pattern_a, 0, 1.0);
        log.observe(&pattern_b, 0, 1.0);
        assert_eq!(log.pattern_count(), 2);

        assert!(log.drop_pattern(&pattern_a));
        assert_eq!(log.pattern_count(), 1);
        assert!(log.arms_for(&pattern_a).is_none());

        // Dropping the same pattern twice returns false.
        assert!(!log.drop_pattern(&pattern_a));
    }

    #[test]
    fn test_trial_log_max_arm_index() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        log.observe(&pattern_a, 2, 1.0);
        log.observe(&pattern_b, 5, 1.0);

        assert_eq!(log.max_arm_index(), Some(5));
    }

    #[test]
    fn test_trial_log_max_arm_index_empty() {
        let log = TrialLog::new();
        assert_eq!(log.max_arm_index(), None);
    }

    #[test]
    fn test_trial_log_patterns_iterator() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);
        let pattern_c = pattern_key(2, &[1, 2]);

        log.observe(&pattern_a, 0, 1.0);
        log.observe(&pattern_b, 0, 1.0);
        log.observe(&pattern_c, 0, 1.0);

        let mut pattern_vec: Vec<PatternKey> = log.patterns().copied().collect();
        pattern_vec.sort();
        assert_eq!(pattern_vec.len(), 3);
    }

    #[test]
    fn test_trial_log_accumulate_across_calls() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        let arm = 0usize;

        // Simulate 10 trials of reward 1.0
        for _ in 0..10 {
            log.observe(&pattern, arm, 1.0);
        }

        let stats = log.arm_stats(&pattern, arm).expect("arm should exist");
        assert_eq!(stats.pulls(), 10);
        assert!((stats.mean() - 1.0).abs() < 1e-9);
        assert!(stats.variance() < 1e-9);
    }

    #[test]
    fn test_trial_log_mixed_sign_rewards() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Simulate propagate (+1) followed by on_backtrack (-1).
        log.observe(&pattern, 0, 1.0);
        log.observe(&pattern, 0, -1.0);

        let stats = log.arm_stats(&pattern, 0).expect("arm should exist");
        assert_eq!(stats.pulls(), 2);
        assert!((stats.total_reward() - 0.0).abs() < 1e-9);
        assert!((stats.mean() - 0.0).abs() < 1e-9);
    }
}
