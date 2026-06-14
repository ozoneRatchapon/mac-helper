//! AbsorbCompress: promotes stable arms to hard rules, compressing the
//! bandit's working memory.
//!
//! # Concept (from katopz/katgpt-rs)
//!
//! As the bandit accumulates trials, some `(pattern, arm)` pairs converge to
//! stable outcomes:
//!
//! - **Stable high-Q**: an arm consistently delivers high reward for a pattern.
//!   The bandit keeps re-selecting it — the trial log entry is dead weight.
//! - **Stable low-Q**: an arm consistently delivers low reward for a pattern.
//!   The bandit keeps avoiding it – the entry is also dead weight.
//!
//! AbsorbCompress scans the log and converts these stable entries into hard
//! rules:
//!
//! - `Lock(pattern, arm)` — always fire this arm for this pattern.
//! - `Reject(pattern, arm)` — never fire this arm for this pattern.
//!
//! Once absorbed, the corresponding log entries are dropped, freeing memory
//! and short-circuiting future policy lookups.
//!
//! # Heuristic vs exact
//!
//! "Stable" is a heuristic: `pulls >= min_pulls AND variance <= max_variance`.
//! This avoids absorbing arms after only one or two lucky trials. The exact
//! thresholds are configurable via [`AbsorbConfig`].
//!
//! # Integration
//!
//! Today AbsorbCompress runs as an offline pass: build rules from a snapshot
//! of the log, then apply (drain absorbed entries). A future iteration can
//! hold an [`AbsorbCompress`] inside [`BanditPruner`](super::BanditPruner) and
//! consult it before each policy lookup, fully short-circuiting the bandit
//! for absorbed patterns.

use std::collections::{HashMap, HashSet};

use super::trial_log::{PatternKey, TrialLog};

/// Tunable thresholds for when an arm qualifies as "stable".
#[derive(Clone, Debug)]
pub struct AbsorbConfig {
    /// Minimum pulls before an arm is even considered for absorption.
    /// Guards against locking in decisions based on one or two lucky trials.
    pub min_pulls: u64,

    /// Mean reward at or above which an arm is "stable high-Q" → lock.
    pub high_mean_threshold: f64,

    /// Mean reward at or below which an arm is "stable low-Q" → reject.
    pub low_mean_threshold: f64,

    /// Variance at or below which an arm is considered "stable".
    /// Combined with the mean thresholds above.
    pub max_variance: f64,
}

impl Default for AbsorbConfig {
    fn default() -> Self {
        Self {
            // 10 pulls is enough to form a stable mean for most reward
            // distributions encountered in decode loops (Bernoulli-ish).
            min_pulls: 10,
            // Reward of 0.8+ over many pulls with low variance → lock.
            high_mean_threshold: 0.8,
            // Reward of 0.2- or below over many pulls with low variance → reject.
            low_mean_threshold: 0.2,
            // Variance under 0.05 → rewards are tightly clustered around the mean.
            max_variance: 0.05,
        }
    }
}

/// Decision returned by [`AbsorbCompress::lookup`] for a pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbsorbDecision {
    /// No hard rule for this pattern — consult the bandit policy as usual.
    Consult,
    /// Hard-select `arm` for this pattern. Skip the bandit entirely.
    Lock(usize),
    /// Hard-reject the listed arms for this pattern. The bandit may still
    /// pick among the remaining arms.
    Reject(HashSet<usize>),
}

/// A frozen set of hard rules derived from a [`TrialLog`] snapshot.
///
/// Built via [`AbsorbCompress::from_log`]. Once built, it can be queried
/// with [`lookup`](Self::lookup) during arm selection, and applied to the
/// log with [`apply`](Self::apply) to drain absorbed entries.
#[derive(Clone, Debug, Default)]
pub struct AbsorbCompress {
    /// `locks.get(pattern) = Some(arm)` — fire this arm unconditionally.
    locks: HashMap<PatternKey, usize>,
    /// `rejects.get(pattern) = Some(set)` — never fire these arms.
    rejects: HashMap<PatternKey, HashSet<usize>>,
}

impl AbsorbCompress {
    /// Create an empty rule set (no locks, no rejects).
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan `log` and build hard rules per [`config`](AbsorbConfig).
    ///
    /// Does not mutate `log`. Use [`apply`](Self::apply) to drain absorbed
    /// entries afterward.
    pub fn from_log(log: &TrialLog, config: &AbsorbConfig) -> Self {
        let mut out = Self::new();

        for pattern in log.patterns() {
            // Some arms may have stats, others may not. Iterate over the
            // union of indices that have any data for this pattern.
            let arm_indices: Vec<usize> = match log.arms_for(pattern) {
                Some(arms) => (0..arms.len()).collect(),
                None => continue,
            };

            let mut rejects_for_pattern: HashSet<usize> = HashSet::new();
            let mut lock_candidate: Option<usize> = None;

            for arm in arm_indices {
                let Some(stats) = log.arm_stats(pattern, arm) else {
                    continue;
                };

                if stats.pulls() < config.min_pulls {
                    continue;
                }

                let mean = stats.mean();
                let variance = stats.variance();

                if variance > config.max_variance {
                    continue;
                }

                if mean >= config.high_mean_threshold {
                    // Stable high-Q. If we already have a lock candidate,
                    // keep the higher mean (deterministic tie-break by value).
                    match lock_candidate {
                        None => lock_candidate = Some(arm),
                        Some(prev) => {
                            let prev_mean = log
                                .arm_stats(pattern, prev)
                                .map(|s| s.mean())
                                .unwrap_or(f64::NEG_INFINITY);
                            if mean > prev_mean {
                                lock_candidate = Some(arm);
                            }
                        }
                    }
                } else if mean <= config.low_mean_threshold {
                    // Stable low-Q.
                    rejects_for_pattern.insert(arm);
                }
            }

            if let Some(arm) = lock_candidate {
                out.locks.insert(*pattern, arm);
            }
            if !rejects_for_pattern.is_empty() {
                out.rejects.insert(*pattern, rejects_for_pattern);
            }
        }

        out
    }

    /// Look up the hard rule (if any) for `pattern`.
    ///
    /// Returns [`AbsorbDecision::Lock`] if a single arm is locked for this
    /// pattern; [`AbsorbDecision::Reject`] if any arms are rejected (and no
    /// lock exists); [`AbsorbDecision::Consult`] otherwise.
    ///
    /// Lock takes precedence over reject: a locked arm is always fired,
    /// regardless of any stale reject entries.
    pub fn lookup(&self, pattern: &PatternKey) -> AbsorbDecision {
        if let Some(&arm) = self.locks.get(pattern) {
            return AbsorbDecision::Lock(arm);
        }
        match self.rejects.get(pattern) {
            Some(set) if !set.is_empty() => AbsorbDecision::Reject(set.clone()),
            _ => AbsorbDecision::Consult,
        }
    }

    /// Apply the rules to `log`, draining absorbed `(pattern, arm)` entries.
    ///
    /// Returns the number of patterns fully removed from the log (i.e.,
    /// patterns that gained a lock and could be dropped entirely).
    ///
    /// For patterns with only rejects (no lock), the rejected arm entries
    /// are zeroed but the pattern is retained – the bandit still needs to
    /// pick among the surviving arms.
    pub fn apply(&self, log: &mut TrialLog) -> usize {
        let mut dropped_patterns = 0usize;

        // Fully drop patterns that gained a lock – the bandit no longer
        // needs to track stats for them.
        for pattern in self.locks.keys() {
            if log.drop_pattern(pattern) {
                dropped_patterns += 1;
            }
        }

        // For patterns with only rejects, we leave the log entry intact.
        // A future iteration could prune the rejected arm sub-entries, but
        // the per-pattern HashMap value is a Vec indexed by arm, so pruning
        // would require reshuffling indices – not worth the complexity for
        // the marginal memory saving in Phase 2.

        dropped_patterns
    }

    /// Number of patterns with a hard lock rule.
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }

    /// Number of patterns with at least one hard reject rule.
    pub fn reject_count(&self) -> usize {
        self.rejects.len()
    }

    /// Total number of patterns with any hard rule (lock or reject).
    ///
    /// Patterns with both a lock and rejects (unusual, since lock takes
    /// precedence) are counted once.
    pub fn total_rules(&self) -> usize {
        let mut patterns: HashSet<&PatternKey> = self.locks.keys().collect();
        patterns.extend(self.rejects.keys());
        patterns.len()
    }

    /// Read-only access to the lock table (for diagnostics / tests).
    pub fn locks(&self) -> &HashMap<PatternKey, usize> {
        &self.locks
    }

    /// Read-only access to the reject table (for diagnostics / tests).
    pub fn rejects(&self) -> &HashMap<PatternKey, HashSet<usize>> {
        &self.rejects
    }

    /// Merge another rule set into this one.
    ///
    /// Locks from `other` overwrite ours on conflict (last-write-wins).
    /// Rejects are unioned.
    pub fn merge(&mut self, other: AbsorbCompress) {
        for (pattern, arm) in other.locks {
            self.locks.insert(pattern, arm);
        }
        for (pattern, rejected) in other.rejects {
            self.rejects.entry(pattern).or_default().extend(rejected);
        }
    }

    /// Drop all rules. Returns to the empty state.
    pub fn clear(&mut self) {
        self.locks.clear();
        self.rejects.clear();
    }
}

/// One-shot helper: build rules from `log`, apply them, return the count
/// of patterns dropped.
///
/// Convenience wrapper for the common case where you just want to compact
/// the log without inspecting the resulting [`AbsorbCompress`] struct.
///
/// The `arms` slice is currently unused by the absorption logic itself
/// (which is purely log-driven), but is accepted for API symmetry with
/// future arm-aware absorption heuristics and to keep the call site at
/// [`BanditPruner::absorb_compress`](super::BanditPruner::absorb_compress)
/// self-documenting.
pub fn absorb_compress(
    log: &mut TrialLog,
    _arms: &[Box<dyn crate::traits::ScreeningPruner>],
) -> usize {
    let config = AbsorbConfig::default();
    let rules = AbsorbCompress::from_log(log, &config);
    rules.apply(log)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandit::trial_log::pattern_key;

    // ── AbsorbConfig ──

    #[test]
    fn test_absorb_config_default_values() {
        let config = AbsorbConfig::default();
        assert!(config.min_pulls >= 1);
        assert!(config.high_mean_threshold > config.low_mean_threshold);
        assert!(config.max_variance > 0.0);
    }

    // ── AbsorbCompress::from_log ──

    #[test]
    fn test_from_log_empty_log_yields_empty_rules() {
        let log = TrialLog::new();
        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lock_count(), 0);
        assert_eq!(rules.reject_count(), 0);
        assert_eq!(rules.total_rules(), 0);
    }

    #[test]
    fn test_from_log_locks_stable_high_q_arm() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: many pulls at 0.95, low variance → stable high-Q.
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.95);
        }

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lock_count(), 1);
        assert_eq!(rules.lookup(&pattern), AbsorbDecision::Lock(0));
    }

    #[test]
    fn test_from_log_rejects_stable_low_q_arm() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: many pulls at 0.05, low variance → stable low-Q.
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.05);
        }

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.reject_count(), 1);
        match rules.lookup(&pattern) {
            AbsorbDecision::Reject(set) => assert!(set.contains(&0)),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn test_from_log_skips_under_sampled_arms() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: only 2 pulls (below default min_pulls=10) at high reward.
        log.observe(&pattern, 0, 1.0);
        log.observe(&pattern, 0, 1.0);

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lock_count(), 0, "should not lock under-sampled arm");
    }

    #[test]
    fn test_from_log_skips_high_variance_arms() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: 20 pulls, mean ~0.5 but huge variance.
        for _ in 0..10 {
            log.observe(&pattern, 0, 1.0);
            log.observe(&pattern, 0, 0.0);
        }

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lock_count(), 0);
        assert_eq!(rules.reject_count(), 0, "high-variance arm is not stable");
    }

    #[test]
    fn test_from_log_lock_picks_highest_mean_among_candidates() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Two arms both qualify as stable high-Q; lock should pick higher mean.
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.85);
        }
        for _ in 0..20 {
            log.observe(&pattern, 1, 0.95);
        }

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lookup(&pattern), AbsorbDecision::Lock(1));
    }

    #[test]
    fn test_from_log_handles_multiple_patterns_independently() {
        let mut log = TrialLog::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        for _ in 0..20 {
            log.observe(&pattern_a, 0, 0.95); // lock arm 0 for A
        }
        for _ in 0..20 {
            log.observe(&pattern_b, 1, 0.05); // reject arm 1 for B
        }

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lookup(&pattern_a), AbsorbDecision::Lock(0));
        match rules.lookup(&pattern_b) {
            AbsorbDecision::Reject(set) => assert!(set.contains(&1)),
            other => panic!("expected Reject for B, got {other:?}"),
        }
    }

    #[test]
    fn test_from_log_lock_overrides_reject_for_same_pattern() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: stable high-Q.
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.95);
        }
        // Arm 1: stable low-Q.
        for _ in 0..20 {
            log.observe(&pattern, 1, 0.05);
        }

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        // Lock takes precedence in lookup().
        assert_eq!(rules.lookup(&pattern), AbsorbDecision::Lock(0));
    }

    // ── AbsorbCompress::lookup ──

    #[test]
    fn test_lookup_unseen_pattern_consults() {
        let rules = AbsorbCompress::new();
        let pattern = pattern_key(0, &[]);
        assert_eq!(rules.lookup(&pattern), AbsorbDecision::Consult);
    }

    #[test]
    fn test_lookup_after_clear_consults() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.95);
        }

        let mut rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        assert_eq!(rules.lookup(&pattern), AbsorbDecision::Lock(0));

        rules.clear();
        assert_eq!(rules.lookup(&pattern), AbsorbDecision::Consult);
    }

    // ── AbsorbCompress::apply ──

    #[test]
    fn test_apply_drops_locked_patterns() {
        let mut log = TrialLog::new();
        let locked = pattern_key(0, &[]);
        let kept = pattern_key(1, &[1]);

        for _ in 0..20 {
            log.observe(&locked, 0, 0.95);
        }
        log.observe(&kept, 0, 0.5); // not absorbed

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        let dropped = rules.apply(&mut log);

        assert_eq!(dropped, 1);
        assert!(log.arms_for(&locked).is_none());
        assert!(log.arms_for(&kept).is_some());
    }

    #[test]
    fn test_apply_preserves_reject_only_patterns() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        for _ in 0..20 {
            log.observe(&pattern, 0, 0.05); // reject
        }
        log.observe(&pattern, 1, 0.5); // survivor

        let rules = AbsorbCompress::from_log(&log, &AbsorbConfig::default());
        let dropped = rules.apply(&mut log);

        assert_eq!(dropped, 0, "no lock → pattern not dropped");
        assert!(log.arms_for(&pattern).is_some());
    }

    #[test]
    fn test_apply_no_rules_no_change() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        log.observe(&pattern, 0, 0.5);

        let rules = AbsorbCompress::new();
        let dropped = rules.apply(&mut log);

        assert_eq!(dropped, 0);
        assert!(log.arms_for(&pattern).is_some());
    }

    // ── merge ──

    #[test]
    fn test_merge_combines_rules() {
        let mut a = AbsorbCompress::new();
        let mut b = AbsorbCompress::new();
        let pattern_a = pattern_key(0, &[]);
        let pattern_b = pattern_key(1, &[1]);

        a.locks.insert(pattern_a, 0);
        b.locks.insert(pattern_b, 1);

        a.merge(b);

        assert_eq!(a.lock_count(), 2);
        assert_eq!(a.lookup(&pattern_a), AbsorbDecision::Lock(0));
        assert_eq!(a.lookup(&pattern_b), AbsorbDecision::Lock(1));
    }

    #[test]
    fn test_merge_other_overrides_on_conflict() {
        let mut a = AbsorbCompress::new();
        let mut b = AbsorbCompress::new();
        let pattern = pattern_key(0, &[]);

        a.locks.insert(pattern, 0);
        b.locks.insert(pattern, 1); // last-write-wins

        a.merge(b);

        assert_eq!(a.lookup(&pattern), AbsorbDecision::Lock(1));
    }

    #[test]
    fn test_merge_unions_rejects() {
        let mut a = AbsorbCompress::new();
        let mut b = AbsorbCompress::new();
        let pattern = pattern_key(0, &[]);

        a.rejects.insert(pattern, [0, 1].into_iter().collect());
        b.rejects.insert(pattern, [1, 2].into_iter().collect());

        a.merge(b);

        match a.lookup(&pattern) {
            AbsorbDecision::Reject(set) => {
                assert!(set.contains(&0));
                assert!(set.contains(&1));
                assert!(set.contains(&2));
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    // ── absorb_compress free function ──

    #[test]
    fn test_absorb_compress_helper_drops_locked_patterns() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.95);
        }

        let arms: Vec<Box<dyn crate::traits::ScreeningPruner>> = Vec::new();
        // absorb_compress ignores arms; empty slice is fine.
        let dropped = absorb_compress(&mut log, &arms);

        assert_eq!(dropped, 1);
        assert!(log.arms_for(&pattern).is_none());
    }

    #[test]
    fn test_absorb_compress_helper_preserves_unstable() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        log.observe(&pattern, 0, 0.5);

        let arms: Vec<Box<dyn crate::traits::ScreeningPruner>> = Vec::new();
        let dropped = absorb_compress(&mut log, &arms);

        assert_eq!(dropped, 0);
        assert!(log.arms_for(&pattern).is_some());
    }

    #[test]
    fn test_absorb_compress_helper_empty_log() {
        let mut log = TrialLog::new();
        let arms: Vec<Box<dyn crate::traits::ScreeningPruner>> = Vec::new();
        let dropped = absorb_compress(&mut log, &arms);
        assert_eq!(dropped, 0);
    }

    // ── counters ──

    #[test]
    fn test_total_rules_counts_distinct_patterns() {
        let mut rules = AbsorbCompress::new();
        let pattern = pattern_key(0, &[]);

        // Same pattern has both a lock and a reject — should count once.
        rules.locks.insert(pattern, 0);
        rules.rejects.insert(pattern, [1].into_iter().collect());

        assert_eq!(rules.total_rules(), 1);
    }

    #[test]
    fn test_total_rules_distinct_locks_and_rejects() {
        let mut rules = AbsorbCompress::new();
        rules.locks.insert(pattern_key(0, &[]), 0);
        rules.locks.insert(pattern_key(1, &[1]), 1);
        rules
            .rejects
            .insert(pattern_key(2, &[1, 2]), [0].into_iter().collect());

        assert_eq!(rules.total_rules(), 3);
        assert_eq!(rules.lock_count(), 2);
        assert_eq!(rules.reject_count(), 1);
    }

    // ── Custom config ──

    #[test]
    fn test_custom_min_pulls_lower_threshold() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Only 3 pulls — default min_pulls=10 would skip this.
        for _ in 0..3 {
            log.observe(&pattern, 0, 0.95);
        }

        let config = AbsorbConfig {
            min_pulls: 2,
            ..AbsorbConfig::default()
        };
        let rules = AbsorbCompress::from_log(&log, &config);
        assert_eq!(
            rules.lock_count(),
            1,
            "custom min_pulls=2 should allow lock"
        );
    }

    #[test]
    fn test_custom_high_mean_threshold_stricter() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Mean 0.85 — default threshold 0.8 would lock.
        for _ in 0..20 {
            log.observe(&pattern, 0, 0.85);
        }

        let config = AbsorbConfig {
            high_mean_threshold: 0.9,
            ..AbsorbConfig::default()
        };
        let rules = AbsorbCompress::from_log(&log, &config);
        assert_eq!(rules.lock_count(), 0, "stricter threshold should skip");
    }
}
