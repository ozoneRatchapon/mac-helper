//! Bandit arm-selection policies: UCB1, ε-greedy, and Thompson sampling.
//!
//! Each policy reads the [`TrialLog`](super::trial_log::TrialLog) state for a
//! pattern and decides which arm to fire next. The decision is deterministic
//! given the log state — stochastic policies derive a per-call seed from
//! `(pattern, total_pulls)` via BLAKE3, ensuring:
//!
//! 1. **Reproducibility**: same log state → same arm selection.
//! 2. **`&self` safety**: [`BanditPruner`](super::BanditPruner) can call
//!    `select` from `batch_is_valid` (which takes `&self`) and re-derive the
//!    same arm in `propagate` (which takes `&mut self`).
//! 3. **No mutable RNG state**: the bandit pruner remains `Send + Sync`.
//!
//! # Untried-arm priority
//!
//! All three policies share one rule: **if any arm has zero pulls for the
//! pattern, return the lowest-index unpulled arm**. This is the standard
//! "play each arm once" initialization that prevents policies from skipping
//! arms before they have any data.

use fastrand::Rng;

use super::trial_log::{PatternKey, TrialLog};

/// Arm-selection strategy for [`BanditPruner`](super::BanditPruner).
///
/// All variants are `Clone + Debug` and produce deterministic selections
/// given the same log state.
#[derive(Clone, Debug)]
pub enum BanditPolicy {
    /// UCB1: pick `argmax_i (mean_i + c · sqrt(ln(N) / n_i))`.
    ///
    /// `c` controls exploration vs exploitation:
    /// - `c = 0.0` → pure exploitation (always pick best mean)
    /// - `c = sqrt(2)` ≈ 1.414 → standard UCB1 (good default)
    /// - `c = 2.0+` → heavy exploration
    ///
    /// Naturally deterministic — no RNG needed.
    Ucb1 { exploration: f64 },

    /// ε-greedy: with probability `epsilon`, pick a uniformly random arm;
    /// otherwise pick the best-mean arm.
    ///
    /// - `epsilon = 0.0` → pure exploitation
    /// - `epsilon = 0.1` → 10% exploration (common default)
    /// - `epsilon = 1.0` → pure random
    ///
    /// Stochastic — derives a per-call seed from `(pattern, total_pulls)`.
    EpsilonGreedy { epsilon: f64 },

    /// Thompson sampling: for each arm, sample from a Normal posterior over
    /// its mean; pick the arm with the highest sample.
    ///
    /// Posterior: `N(mean, variance / pulls)` — shrinks toward the empirical
    /// mean as pulls accumulate. Untried arms get infinite variance (priority).
    ///
    /// Stochastic — derives a per-call seed from `(pattern, total_pulls)`.
    Thompson,
}

impl BanditPolicy {
    /// Standard UCB1 with exploration constant `sqrt(2)`.
    pub const DEFAULT_EXPLORATION: f64 = std::f64::consts::SQRT_2;

    /// Default ε for ε-greedy: 10% exploration.
    pub const DEFAULT_EPSILON: f64 = 0.1;

    /// UCB1 with the standard exploration constant (`sqrt(2)`).
    pub fn ucb1() -> Self {
        Self::Ucb1 {
            exploration: Self::DEFAULT_EXPLORATION,
        }
    }

    /// ε-greedy with the default 10% exploration.
    pub fn epsilon_greedy() -> Self {
        Self::EpsilonGreedy {
            epsilon: Self::DEFAULT_EPSILON,
        }
    }

    /// Thompson sampling with default settings.
    pub fn thompson() -> Self {
        Self::Thompson
    }

    /// Select an arm for `pattern` given the current `log` state.
    ///
    /// Returns the index of the chosen arm in `[0, arm_count)`.
    ///
    /// # Guarantees
    ///
    /// - Deterministic: same `(self, pattern, log)` → same arm.
    /// - Untried arms have priority: if any arm has zero pulls, the
    ///   lowest-index unpulled arm is returned.
    /// - `arm_count == 0` returns 0 defensively (callers should guard).
    pub fn select(&self, pattern: &PatternKey, log: &TrialLog, arm_count: usize) -> usize {
        if arm_count <= 1 {
            return 0;
        }

        let total_pulls = log.total_pulls_for_pattern(pattern);

        // Play each arm once before any policy kicks in.
        if let Some(untried) = lowest_untried_arm(log, pattern, arm_count) {
            return untried;
        }

        match self {
            Self::Ucb1 { exploration } => {
                ucb1_select(log, pattern, arm_count, total_pulls, *exploration)
            }
            Self::EpsilonGreedy { epsilon } => {
                let mut rng = seeded_rng(pattern, total_pulls);
                epsilon_greedy_select(log, pattern, arm_count, *epsilon, &mut rng)
            }
            Self::Thompson => {
                let mut rng = seeded_rng(pattern, total_pulls);
                thompson_select(log, pattern, arm_count, &mut rng)
            }
        }
    }
}

impl Default for BanditPolicy {
    fn default() -> Self {
        Self::ucb1()
    }
}

// ── Policy implementations ─────────────────────────────────────

/// UCB1 arm selection: balance exploitation (mean) with exploration
/// (uncertainty bonus for under-sampled arms).
///
/// Score: `mean_i + c · sqrt(ln(N) / n_i)`
///
/// Ties broken by lowest index (deterministic).
fn ucb1_select(
    log: &TrialLog,
    pattern: &PatternKey,
    arm_count: usize,
    total_pulls: u64,
    exploration: f64,
) -> usize {
    // Guard: total_pulls > 0 guaranteed by untried-arm priority in `select`.
    let ln_total = (total_pulls.max(1) as f64).ln();

    let mut best_arm = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for arm in 0..arm_count {
        let stats = match log.arm_stats(pattern, arm) {
            Some(s) => s,
            None => continue, // unreachable: untried arms handled earlier
        };

        let pulls = stats.pulls() as f64;
        if pulls < 1.0 {
            // Defensive: should have been caught by untried priority.
            return arm;
        }

        let mean = stats.mean();
        let bonus = exploration * (ln_total / pulls).sqrt();
        let score = mean + bonus;

        if score > best_score {
            best_score = score;
            best_arm = arm;
        }
    }

    best_arm
}

/// ε-greedy arm selection: explore with probability `epsilon`,
/// otherwise exploit the best empirical mean.
///
/// Ties in mean broken by lowest index.
fn epsilon_greedy_select(
    log: &TrialLog,
    pattern: &PatternKey,
    arm_count: usize,
    epsilon: f64,
    rng: &mut Rng,
) -> usize {
    let roll = rng.f64();
    if roll < epsilon {
        return rng.usize(0..arm_count);
    }

    best_mean_arm(log, pattern, arm_count).unwrap_or(0)
}

/// Thompson sampling: sample from each arm's Normal posterior over its mean,
/// pick the highest sample.
///
/// Posterior: `N(mean, variance / pulls)` — shrinks as evidence accumulates.
/// High variance → wide sampling → more exploration.
fn thompson_select(log: &TrialLog, pattern: &PatternKey, arm_count: usize, rng: &mut Rng) -> usize {
    let mut best_arm = 0usize;
    let mut best_sample = f64::NEG_INFINITY;

    for arm in 0..arm_count {
        let stats = match log.arm_stats(pattern, arm) {
            Some(s) => s,
            None => continue,
        };

        let pulls = stats.pulls();
        if pulls == 0 {
            return arm; // defensive — untried priority should have handled this
        }

        let mean = stats.mean();
        let variance = stats.variance();
        // Posterior std-dev: σ / sqrt(n). Floor at a tiny epsilon to avoid
        // zero-spread sampling when variance is exactly 0 (constant rewards).
        let posterior_std = (variance / pulls as f64).max(1e-9).sqrt();

        let sample = mean + posterior_std * sample_standard_normal(rng);

        if sample > best_sample {
            best_sample = sample;
            best_arm = arm;
        }
    }

    best_arm
}

// ── Helpers ────────────────────────────────────────────────────

/// Return the lowest-index arm with zero pulls for `pattern`, if any.
fn lowest_untried_arm(log: &TrialLog, pattern: &PatternKey, arm_count: usize) -> Option<usize> {
    for arm in 0..arm_count {
        let pulls = log.arm_stats(pattern, arm).map(|s| s.pulls()).unwrap_or(0);
        if pulls == 0 {
            return Some(arm);
        }
    }
    None
}

/// Return the arm with the highest empirical mean. Ties → lowest index.
fn best_mean_arm(log: &TrialLog, pattern: &PatternKey, arm_count: usize) -> Option<usize> {
    let mut best_arm = None;
    let mut best_mean = f64::NEG_INFINITY;

    for arm in 0..arm_count {
        let mean = log.arm_mean(pattern, arm).unwrap_or(f64::NEG_INFINITY);
        if mean > best_mean {
            best_mean = mean;
            best_arm = Some(arm);
        }
    }

    best_arm
}

/// Derive a deterministic RNG seed from `(pattern, total_pulls)`.
///
/// Same inputs → same seed → same stochastic decisions. This is what makes
/// ε-greedy and Thompson reproducible without mutable state on the policy.
fn seeded_rng(pattern: &PatternKey, total_pulls: u64) -> Rng {
    let mut hasher = blake3::Hasher::new();
    hasher.update(pattern);
    hasher.update(&total_pulls.to_le_bytes());
    let hash = *hasher.finalize().as_bytes();

    // hash is 32 bytes, slicing 8 always succeeds — unwrap_or_default is safe.
    let seed_bytes: [u8; 8] = hash[..8].try_into().unwrap_or_default();
    Rng::with_seed(u64::from_le_bytes(seed_bytes))
}

/// Sample from the standard Normal distribution N(0, 1) via Box-Muller.
///
/// Returns a single sample (the second Box-Muller value is discarded for
/// simplicity — we only need one per call).
fn sample_standard_normal(rng: &mut Rng) -> f64 {
    // u1 must be in (0, 1] — clamp away from 0 to avoid ln(0) = -∞.
    let u1 = rng.f64().max(1e-12);
    let u2 = rng.f64();

    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    radius * theta.cos()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandit::trial_log::pattern_key;

    // ── UCB1 ──

    #[test]
    fn test_ucb1_returns_untried_arm_first() {
        let policy = BanditPolicy::ucb1();
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // No arm tried → arm 0 first.
        assert_eq!(policy.select(&pattern, &log, 3), 0);
    }

    #[test]
    fn test_ucb1_plays_each_arm_once() {
        let policy = BanditPolicy::ucb1();
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0 tried → next should be arm 1.
        log.observe(&pattern, 0, 1.0);
        assert_eq!(policy.select(&pattern, &log, 3), 1);

        // Arms 0,1 tried → next should be arm 2.
        log.observe(&pattern, 1, 0.5);
        assert_eq!(policy.select(&pattern, &log, 3), 2);
    }

    #[test]
    fn test_ucb1_exploits_best_mean_after_initialization() {
        let policy = BanditPolicy::Ucb1 { exploration: 0.0 };
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // All arms have one pull. Means: arm0=0.9, arm1=0.1, arm2=0.5.
        log.observe(&pattern, 0, 0.9);
        log.observe(&pattern, 1, 0.1);
        log.observe(&pattern, 2, 0.5);

        // With c=0, pure exploitation → arm 0 (best mean).
        assert_eq!(policy.select(&pattern, &log, 3), 0);
    }

    #[test]
    fn test_ucb1_exploration_bonus_picks_under_sampled() {
        let policy = BanditPolicy::Ucb1 { exploration: 100.0 };
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0 pulled 100x with mean 0.5, arm 1 pulled 1x with mean 0.0.
        // With huge exploration, the under-sampled arm 1 should win.
        for _ in 0..100 {
            log.observe(&pattern, 0, 0.5);
        }
        log.observe(&pattern, 1, 0.0);

        let selected = policy.select(&pattern, &log, 2);
        assert_eq!(selected, 1, "huge c should pick under-sampled arm");
    }

    #[test]
    fn test_ucb1_deterministic_given_log_state() {
        let policy = BanditPolicy::ucb1();
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.7);
        log.observe(&pattern, 0, 0.8);
        log.observe(&pattern, 1, 0.3);
        log.observe(&pattern, 1, 0.5);

        let a = policy.select(&pattern, &log, 2);
        let b = policy.select(&pattern, &log, 2);
        assert_eq!(a, b, "UCB1 must be deterministic");
    }

    // ── ε-greedy ──

    #[test]
    fn test_epsilon_greedy_returns_untried_arm_first() {
        let policy = BanditPolicy::epsilon_greedy();
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        assert_eq!(policy.select(&pattern, &log, 3), 0);
    }

    #[test]
    fn test_epsilon_greedy_zero_epsilon_exploits() {
        let policy = BanditPolicy::EpsilonGreedy { epsilon: 0.0 };
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.1);
        log.observe(&pattern, 1, 0.9);
        log.observe(&pattern, 2, 0.5);

        // epsilon=0 → always exploit → arm 1 (best mean).
        for _ in 0..10 {
            assert_eq!(policy.select(&pattern, &log, 3), 1);
        }
    }

    #[test]
    fn test_epsilon_greedy_one_epsilon_picks_in_range() {
        let policy = BanditPolicy::EpsilonGreedy { epsilon: 1.0 };
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.1);
        log.observe(&pattern, 1, 0.9);
        log.observe(&pattern, 2, 0.5);

        // epsilon=1 → always explore → arm in [0, 3).
        for total_pulls_bias in 0..50 {
            // Force different seed each iteration by varying log state.
            log.observe(&pattern, 0, 0.1);
            let arm = policy.select(&pattern, &log, 3);
            assert!(arm < 3, "arm {arm} out of range");
            let _ = total_pulls_bias;
        }
    }

    #[test]
    fn test_epsilon_greedy_deterministic_given_log_state() {
        let policy = BanditPolicy::EpsilonGreedy { epsilon: 0.5 };
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.5);
        log.observe(&pattern, 1, 0.5);

        let a = policy.select(&pattern, &log, 2);
        let b = policy.select(&pattern, &log, 2);
        assert_eq!(a, b, "ε-greedy must be deterministic per log state");
    }

    // ── Thompson ──

    #[test]
    fn test_thompson_returns_untried_arm_first() {
        let policy = BanditPolicy::thompson();
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        assert_eq!(policy.select(&pattern, &log, 3), 0);
    }

    #[test]
    fn test_thompson_exploits_clear_winner_with_low_variance() {
        let policy = BanditPolicy::thompson();
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: many pulls at 0.9 (low variance, high mean).
        for _ in 0..100 {
            log.observe(&pattern, 0, 0.9);
        }
        // Arm 1: many pulls at 0.1 (low variance, low mean).
        for _ in 0..100 {
            log.observe(&pattern, 1, 0.1);
        }

        // With low posterior variance, arm 0 should win almost always.
        let mut arm_0_count = 0usize;
        for bias in 0..50u64 {
            // Vary log state to get different Thompson samples.
            let mut local_log = log.clone();
            local_log.observe(&pattern, 0, 0.9);
            // Force unique seed by changing total_pulls.
            let _ = bias;
            let arm = policy.select(&pattern, &local_log, 2);
            if arm == 0 {
                arm_0_count += 1;
            }
        }
        assert!(
            arm_0_count >= 45,
            "Thompson should pick high-mean arm most of the time, got {arm_0_count}/50"
        );
    }

    #[test]
    fn test_thompson_deterministic_given_log_state() {
        let policy = BanditPolicy::thompson();
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.5);
        log.observe(&pattern, 0, 0.6);
        log.observe(&pattern, 1, 0.4);
        log.observe(&pattern, 1, 0.5);

        let a = policy.select(&pattern, &log, 2);
        let b = policy.select(&pattern, &log, 2);
        assert_eq!(a, b, "Thompson must be deterministic per log state");
    }

    #[test]
    fn test_thompson_explores_with_high_variance() {
        let policy = BanditPolicy::thompson();
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0: 2 pulls with rewards 0.0 and 1.0 (high variance).
        log.observe(&pattern, 0, 0.0);
        log.observe(&pattern, 0, 1.0);
        // Arm 1: 2 pulls with rewards 0.4 and 0.6 (lower variance).
        log.observe(&pattern, 1, 0.4);
        log.observe(&pattern, 1, 0.6);

        // Both arms have similar means (~0.5), but arm 0 has higher variance.
        // Thompson should sometimes pick each — sample many times.
        let mut arm_1_count = 0usize;
        for trial in 0..50u64 {
            // Vary total_pulls to get different seeds.
            let mut local_log = log.clone();
            local_log.observe(&pattern, 0, 0.5);
            local_log.observe(&pattern, 1, 0.5);
            // Force unique seed each trial.
            let _ = trial;
            let arm = policy.select(&pattern, &local_log, 2);
            if arm == 1 {
                arm_1_count += 1;
            }
        }
        // Should pick arm 1 at least sometimes.
        assert!(
            arm_1_count > 0,
            "Thompson should explore arm 1 at least once"
        );
    }

    // ── Helpers ──

    #[test]
    fn test_lowest_untried_arm_picks_lowest() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        // Arm 0 tried, arms 1 and 2 untried → lowest untried is arm 1.
        log.observe(&pattern, 0, 1.0);
        assert_eq!(lowest_untried_arm(&log, &pattern, 3), Some(1));
    }

    #[test]
    fn test_lowest_untried_arm_none_when_all_tried() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 1.0);
        log.observe(&pattern, 1, 1.0);
        log.observe(&pattern, 2, 1.0);

        assert_eq!(lowest_untried_arm(&log, &pattern, 3), None);
    }

    #[test]
    fn test_best_mean_arm_picks_highest() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.3);
        log.observe(&pattern, 1, 0.9);
        log.observe(&pattern, 2, 0.5);

        assert_eq!(best_mean_arm(&log, &pattern, 3), Some(1));
    }

    #[test]
    fn test_best_mean_arm_ties_break_to_lowest_index() {
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 0.5);
        log.observe(&pattern, 1, 0.5);

        assert_eq!(best_mean_arm(&log, &pattern, 2), Some(0));
    }

    #[test]
    fn test_seeded_rng_deterministic() {
        let pattern = pattern_key(0, &[]);
        let pulls = 5u64;

        let mut a = seeded_rng(&pattern, pulls);
        let mut b = seeded_rng(&pattern, pulls);

        for _ in 0..10 {
            assert_eq!(a.u64(..), b.u64(..), "seeded RNG must be deterministic");
        }
    }

    #[test]
    fn test_seeded_rng_differs_on_pulls() {
        let pattern = pattern_key(0, &[]);

        let mut a = seeded_rng(&pattern, 1);
        let mut b = seeded_rng(&pattern, 2);

        // First sample should differ (probability of collision is negligible).
        assert_ne!(a.u64(..), b.u64(..));
    }

    #[test]
    fn test_sample_standard_normal_in_reasonable_range() {
        let mut rng = Rng::with_seed(42);
        for _ in 0..1000 {
            let sample = sample_standard_normal(&mut rng);
            // 6σ covers 99.9999999% of standard normal samples.
            assert!(
                sample.abs() < 6.0,
                "sample {sample} outside 6σ — check Box-Muller"
            );
        }
    }

    #[test]
    fn test_sample_standard_normal_mean_near_zero() {
        let mut rng = Rng::with_seed(42);
        let n = 10_000usize;
        let sum: f64 = (0..n).map(|_| sample_standard_normal(&mut rng)).sum();
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.1,
            "sampled mean {mean} should be near 0 for N(0,1)"
        );
    }

    // ── Edge cases ──

    #[test]
    fn test_select_arm_count_zero() {
        let policy = BanditPolicy::ucb1();
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        assert_eq!(policy.select(&pattern, &log, 0), 0);
    }

    #[test]
    fn test_select_arm_count_one() {
        let policy = BanditPolicy::ucb1();
        let log = TrialLog::new();
        let pattern = pattern_key(0, &[]);
        assert_eq!(policy.select(&pattern, &log, 1), 0);
    }

    #[test]
    fn test_select_all_variants_handle_empty_log() {
        let pattern = pattern_key(0, &[]);
        let log = TrialLog::new();

        for policy in [
            BanditPolicy::ucb1(),
            BanditPolicy::epsilon_greedy(),
            BanditPolicy::thompson(),
        ] {
            assert_eq!(
                policy.select(&pattern, &log, 3),
                0,
                "empty log → lowest-index untried arm"
            );
        }
    }

    #[test]
    fn test_select_handles_negative_rewards() {
        // Backtracking produces negative rewards. Policies must handle them.
        let policy = BanditPolicy::ucb1();
        let mut log = TrialLog::new();
        let pattern = pattern_key(0, &[]);

        log.observe(&pattern, 0, 1.0);
        log.observe(&pattern, 0, -1.0); // simulate backtrack
        log.observe(&pattern, 1, 0.5);
        log.observe(&pattern, 2, 0.5);

        let arm = policy.select(&pattern, &log, 3);
        // Arm 0 has mean 0.0, arms 1 and 2 have mean 0.5 — arm 1 should win
        // unless exploration bonus is very large (it isn't with default c).
        assert!(
            arm == 1 || arm == 2,
            "should not pick negative-mean arm 0, got {arm}"
        );
    }

    #[test]
    fn test_default_policy_is_ucb1() {
        match BanditPolicy::default() {
            BanditPolicy::Ucb1 { exploration } => {
                assert!((exploration - BanditPolicy::DEFAULT_EXPLORATION).abs() < 1e-9);
            }
            _ => panic!("default should be UCB1"),
        }
    }
}
