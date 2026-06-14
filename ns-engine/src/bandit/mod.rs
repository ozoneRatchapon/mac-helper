//! Multi-armed bandit layer: adaptive pruner selection over multiple arms.
//!
//! The bandit is the "speculation on policy" layer from katopz/katgpt-rs:
//! instead of always firing the same [`ConstraintPruner`](crate::ConstraintPruner),
//! the bandit tracks per-context reward statistics per arm and adapts its
//! selection over time. Intelligence evolves by updating the trial log, not
//! by gradient descent.
//!
//! # Submodules
//!
//! - [`trial_log`] — BLAKE3-keyed per-pattern, per-arm reward statistics
//! - [`policies`] — UCB1, ε-greedy, Thompson sampling arm selection
//! - [`bandit_pruner`] — `BanditPruner` implementing `ConstraintPruner`
//! - [`absorb`] — `AbsorbCompress` for promoting stable arms to hard rules

pub mod absorb;
pub mod bandit_pruner;
pub mod policies;
pub mod trial_log;

pub use absorb::{absorb_compress, AbsorbCompress, AbsorbConfig, AbsorbDecision};
pub use bandit_pruner::BanditPruner;
pub use policies::BanditPolicy;
pub use trial_log::{pattern_key, ArmStats, PatternKey, TrialLog};
