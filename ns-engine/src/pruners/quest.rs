//! `QuestActionPruner` — RPG quest-tree domain: forward model + legality pruner.
//!
//! Phase 5 domain-generalization deliverable. The modelless thesis applied
//! to an RPG quest tree: deterministic prerequisite semantics encoded as a
//! [`GameState`] forward model ([`QuestState`]), with a matching
//! [`ConstraintPruner`] ([`QuestActionPruner`]) that mirrors legality by
//! stateless replay. No model weights, no learned policy — just deterministic
//! forward-model replay.
//!
//! # Quest lifecycle
//!
//! ```text
//!   NotStarted ──(ACCEPT, prereqs met)──▶ Active ──┬─(COMPLETE)─▶ Completed
//!                                                  └─(FAIL)────▶ Failed
//! ```
//!
//! - **NotStarted**: quest not yet accepted. Becomes *Available* (derivable)
//!   when ALL its prerequisites are `Completed`.
//! - **Active**: accepted, in progress. Can be `COMPLETE`d or `FAIL`ed.
//! - **Completed**: done successfully. Unlocks dependents.
//! - **Failed**: done, failed. Does NOT unlock dependents (prereqs require
//!   `Completed`). Failing a goal-path quest can make the puzzle unwinnable.
//!
//! # Action space (dynamic vocab)
//!
//! Unlike [`BomberState`](crate::pruners::BomberState) (fixed 6-token vocab),
//! the quest action space scales with `num_quests`: each quest exposes 3
//! action kinds (ACCEPT/COMPLETE/FAIL), so `vocab = 3 × num_quests`. Tokens
//! encode `(quest_index, kind)` via [`encode_quest_action`] /
//! [`decode_quest_action`].
//!
//! # Terminal conditions
//!
//! - **Goal**: every quest in `config.goal_quests` is `Completed`. Implies
//!   terminal; reward `+1.0`.
//! - **Dead-end terminal**: no quest can transition further (no `Available`
//!   quest, no `Active` quest) AND not a goal. Reward `−1.0`.
//!
//! # Stateless replay convention
//!
//! [`QuestActionPruner::is_valid`] holds no mutable state. It replays
//! `parent_tokens` from [`QuestState::initial`] to reconstruct the current
//! state, then delegates to [`QuestState::is_legal`]. By construction,
//! `pruner.is_valid ≡ state.is_legal`.
//!
//! # Token kinds
//!
//! Per-quest action-kind offsets:
//! - 0 = ACCEPT, 1 = COMPLETE, 2 = FAIL
//!
//! `QUEST_ARM_ID = 8` (Sudoku=1, Ngram=2, Regex=3, NoPruner=4, Escrow=5,
//! JsonSchema=6, Bomber=7, Quest=8).

use blake3::Hasher;

use crate::game::GameState;
use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

// ── Action token space ─────────────────────────────────────────

/// Per-quest action kind: accept a quest (NotStarted → Active).
pub const QUEST_ACTION_ACCEPT: TokenId = 0;
/// Per-quest action kind: complete a quest (Active → Completed).
pub const QUEST_ACTION_COMPLETE: TokenId = 1;
/// Per-quest action kind: fail a quest (Active → Failed).
pub const QUEST_ACTION_FAIL: TokenId = 2;

/// Number of action kinds per quest (ACCEPT/COMPLETE/FAIL).
pub const ACTIONS_PER_QUEST: usize = 3;

/// Stable arm identifier for [`QuestActionPruner`] within a
/// [`BanditPruner`](crate::bandit::BanditPruner).
///
/// Sudoku=1, Ngram=2, Regex=3, NoPruner=4, Escrow=5, JsonSchema=6, Bomber=7,
/// Quest=8.
pub const QUEST_ARM_ID: ArmId = 8;

/// Human-readable arm label for diagnostics and logging.
pub const QUEST_LABEL: &str = "quest-action";

/// Encode `(quest_index, kind)` into a flat action token.
///
/// Inverse of [`decode_quest_action`]. The quest index is multiplied by
/// [`ACTIONS_PER_QUEST`] and the kind is added as a low offset.
pub const fn encode_quest_action(quest: usize, kind: TokenId) -> TokenId {
    (quest as TokenId) * (ACTIONS_PER_QUEST as TokenId) + kind
}

/// Decode an action token into `(quest_index, kind)`.
///
/// Inverse of [`encode_quest_action`]. Always succeeds (every token maps to a
/// valid kind, since `token % ACTIONS_PER_QUEST ∈ {0,1,2}`); callers must
/// validate that `quest_index < num_quests` separately.
pub const fn decode_quest_action(token: TokenId) -> (usize, TokenId) {
    let apq = ACTIONS_PER_QUEST as TokenId;
    ((token / apq) as usize, token % apq)
}

// ── QuestStatus ────────────────────────────────────────────────

/// Lifecycle state of a single quest.
///
/// `NotStarted` covers both *locked* (prereqs unmet) and *available* (prereqs
/// met) — the distinction is derived from [`QuestState::is_available`] rather
/// than stored, because it changes as prerequisites complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuestStatus {
    /// Not yet accepted. Becomes *available* (and ACCEPT becomes legal) once
    /// all prerequisites are `Completed`.
    NotStarted,
    /// Accepted and in progress. Can be `COMPLETE`d or `FAIL`ed.
    Active,
    /// Completed successfully. Counts toward unlocking dependents.
    Completed,
    /// Failed. Does NOT unlock dependents (prereqs require `Completed`).
    Failed,
}

// ── QuestConfig ────────────────────────────────────────────────

/// Configuration for an RPG quest tree: prerequisite graph + goal set.
///
/// The config is held by both [`QuestState`] (for the forward model) and
/// [`QuestActionPruner`] (for stateless replay). The pruner's config must
/// match the state's initial config for legality to agree — this is the
/// canonical-origin invariant described in
/// [`speculative_generate`](crate::speculative_generate).
///
/// # Validation
///
/// Use [`validate`](Self::validate) before constructing a state. Invalid
/// configs cause [`QuestState::initial`] to panic (fail-fast on programmer
/// error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuestConfig {
    /// Number of quests in the tree. Must be ≥ 1.
    pub num_quests: usize,
    /// `prerequisites[i]` = quest IDs that must ALL be `Completed` before
    /// quest `i` can be accepted. Length must equal `num_quests`.
    pub prerequisites: Vec<Vec<usize>>,
    /// Quest IDs that must ALL be `Completed` for victory. Each must be a
    /// valid index `< num_quests`. Empty is permitted but makes the puzzle
    /// unwinnable (goal is never reached).
    pub goal_quests: Vec<usize>,
}

/// Validation error for [`QuestConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestConfigError {
    /// `num_quests` is zero.
    ZeroQuests,
    /// `prerequisites.len() != num_quests`.
    PrerequisiteCountMismatch,
    /// A prerequisite index is `>= num_quests`.
    PrerequisiteOutOfBounds,
    /// Quest `i` lists itself as a prerequisite (structural deadlock).
    SelfPrerequisite,
    /// A `goal_quests` entry is `>= num_quests`.
    GoalQuestOutOfBounds,
}

impl std::fmt::Display for QuestConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroQuests => write!(f, "num_quests must be >= 1"),
            Self::PrerequisiteCountMismatch => {
                write!(f, "prerequisites length must equal num_quests")
            }
            Self::PrerequisiteOutOfBounds => {
                write!(f, "prerequisite index must be < num_quests")
            }
            Self::SelfPrerequisite => write!(f, "quest cannot list itself as a prerequisite"),
            Self::GoalQuestOutOfBounds => write!(f, "goal quest index must be < num_quests"),
        }
    }
}

impl std::error::Error for QuestConfigError {}

impl QuestConfig {
    /// Validate the config invariants.
    ///
    /// Returns `Ok(())` if the config is safe to pass to [`QuestState::initial`].
    /// Does NOT detect multi-quest prerequisite cycles — quests in such a
    /// cycle simply stay locked forever (a valid, unwinnable configuration).
    pub fn validate(&self) -> Result<(), QuestConfigError> {
        if self.num_quests == 0 {
            return Err(QuestConfigError::ZeroQuests);
        }
        if self.prerequisites.len() != self.num_quests {
            return Err(QuestConfigError::PrerequisiteCountMismatch);
        }
        for q in 0..self.num_quests {
            for &p in &self.prerequisites[q] {
                if p >= self.num_quests {
                    return Err(QuestConfigError::PrerequisiteOutOfBounds);
                }
                if p == q {
                    return Err(QuestConfigError::SelfPrerequisite);
                }
            }
        }
        for &g in &self.goal_quests {
            if g >= self.num_quests {
                return Err(QuestConfigError::GoalQuestOutOfBounds);
            }
        }
        Ok(())
    }

    /// Total action-token vocabulary size: `ACTIONS_PER_QUEST × num_quests`.
    pub fn vocab(&self) -> usize {
        ACTIONS_PER_QUEST * self.num_quests
    }

    /// Boolean mask: `true` for goal quests and their transitive prerequisites.
    ///
    /// A quest on the goal path MUST be `Completed` for the goal to be
    /// reachable. Used by the screening heuristic to weight goal-path
    /// progress higher than off-path distraction.
    pub fn goal_path_mask(&self) -> Vec<bool> {
        let mut on_path = vec![false; self.num_quests];
        // Reverse reachability from goal quests through prerequisites.
        let mut stack: Vec<usize> = self.goal_quests.clone();
        while let Some(q) = stack.pop() {
            if q >= self.num_quests || on_path[q] {
                continue;
            }
            on_path[q] = true;
            for &p in &self.prerequisites[q] {
                if p < self.num_quests && !on_path[p] {
                    stack.push(p);
                }
            }
        }
        on_path
    }

    /// Is quest `q` a goal quest?
    fn is_goal_quest(&self, q: usize) -> bool {
        self.goal_quests.contains(&q)
    }
}

impl Default for QuestConfig {
    /// A minimal 3-quest linear chain: Q0 → Q1 → Q2, goal = [Q2].
    ///
    /// Solution: accept0, complete0, accept1, complete1, accept2, complete2
    /// (6 actions).
    fn default() -> Self {
        Self {
            num_quests: 3,
            prerequisites: vec![vec![], vec![0], vec![1]],
            goal_quests: vec![2],
        }
    }
}

// ── QuestState ─────────────────────────────────────────────────

/// The dynamic state of an RPG quest tree: per-quest status vector.
///
/// Implements [`GameState`] for use with
/// [`speculative_generate`](crate::speculative_generate). The `step` method
/// is pure (snapshot semantics): it clones `self`, applies the action, and
/// returns the successor — without mutating the receiver.
///
/// # Snapshot semantics
///
/// Because `step` returns a new state, the speculative generate loop
/// snapshots by calling `step` at branch points. No undo / backtrack logic
/// is needed — the old state is simply retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuestState {
    /// Per-quest status, indexed by quest ID. Length equals
    /// `config.num_quests`.
    pub status: Vec<QuestStatus>,
    /// Action token that produced this state, or `None` for the initial state.
    pub last_action: Option<TokenId>,
    /// The quest-tree configuration (immutable graph + goal set).
    pub config: QuestConfig,
}

impl QuestState {
    /// Construct the initial state from a config: every quest `NotStarted`.
    ///
    /// # Panics
    ///
    /// Panics if `config.validate()` returns an error. This is a fail-fast
    /// precondition: invalid configs are programmer errors, not runtime
    /// conditions.
    pub fn initial(config: QuestConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("invalid QuestConfig: {e}");
        }
        Self {
            status: vec![QuestStatus::NotStarted; config.num_quests],
            last_action: None,
            config,
        }
    }

    /// Human-readable one-line description for test diagnostics.
    pub fn describe(&self) -> String {
        let statuses = self
            .status
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{i}:{s:?}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "quests=[{statuses}] last={last}",
            statuses = statuses,
            last = match self.last_action {
                Some(a) => a.to_string(),
                None => "-".to_string(),
            },
        )
    }

    // ── Derived predicates ─────────────────────────────────────

    /// Is quest `q` *available* — `NotStarted` AND all prerequisites are
    /// `Completed`?
    ///
    /// Caller must ensure `q < num_quests`.
    fn is_available(&self, q: usize) -> bool {
        if self.status[q] != QuestStatus::NotStarted {
            return false;
        }
        self.config.prerequisites[q]
            .iter()
            .all(|&p| self.status[p] == QuestStatus::Completed)
    }

    // ── Forward model mechanics ────────────────────────────────

    /// Apply the action, mutating status in place.
    ///
    /// # Panics
    ///
    /// Panics if the action is illegal in this state. Callers must pre-filter
    /// via [`is_legal`](Self::is_legal) or [`try_step`](GameState::try_step).
    fn apply_action(&mut self, action: TokenId) {
        assert!(!self.is_terminal(), "apply_action called on terminal state");
        let (q, kind) = decode_quest_action(action);
        assert!(
            q < self.config.num_quests,
            "quest index {q} out of range",
            q = q,
        );
        match kind {
            QUEST_ACTION_ACCEPT => {
                assert!(
                    self.status[q] == QuestStatus::NotStarted && self.is_available(q),
                    "ACCEPT requires quest {q} Available",
                    q = q,
                );
                self.status[q] = QuestStatus::Active;
            }
            QUEST_ACTION_COMPLETE => {
                assert!(
                    self.status[q] == QuestStatus::Active,
                    "COMPLETE requires quest {q} Active",
                    q = q,
                );
                self.status[q] = QuestStatus::Completed;
            }
            QUEST_ACTION_FAIL => {
                assert!(
                    self.status[q] == QuestStatus::Active,
                    "FAIL requires quest {q} Active",
                    q = q,
                );
                self.status[q] = QuestStatus::Failed;
            }
            _ => panic!("unknown quest action kind: {kind}"),
        }
    }

    // ── Screening heuristic ────────────────────────────────────

    /// Graded screen score ∈ [0.0, 1.0] for a **legal** action in this state.
    ///
    /// Used by [`QuestActionPruner::screen`]. Encodes a lightweight
    /// domain heuristic (no search):
    /// - `COMPLETE` a goal quest → `0.95`
    /// - `COMPLETE` a goal-path prerequisite (non-goal) → `0.85`
    /// - `COMPLETE` an off-path quest → `0.6`
    /// - `ACCEPT` a goal-path quest → `0.75`
    /// - `ACCEPT` an off-path quest → `0.55`
    /// - `FAIL` any quest → `0.2` (failing is almost always counterproductive)
    ///
    /// Caller must ensure the action is legal; this method returns `0.0` for
    /// illegal or unknown actions.
    fn action_screen_score(&self, action: TokenId) -> f32 {
        let (q, kind) = decode_quest_action(action);
        if q >= self.config.num_quests {
            return 0.0;
        }
        let goal_path = self.config.goal_path_mask();
        let is_goal_quest = self.config.is_goal_quest(q);
        match kind {
            QUEST_ACTION_COMPLETE => {
                if is_goal_quest {
                    0.95
                } else if goal_path[q] {
                    0.85
                } else {
                    0.6
                }
            }
            QUEST_ACTION_ACCEPT => {
                if goal_path[q] {
                    0.75
                } else {
                    0.55
                }
            }
            QUEST_ACTION_FAIL => 0.2,
            _ => 0.0,
        }
    }
}

// ── GameState impl ─────────────────────────────────────────────

impl GameState for QuestState {
    fn last_action(&self) -> Option<TokenId> {
        self.last_action
    }

    fn step(&self, action: TokenId) -> Self {
        let mut next = self.clone();
        next.last_action = Some(action);
        next.apply_action(action);
        next
    }

    /// Direct legality predicate — O(prereqs of q) per call. This is the
    /// authoritative legality check that [`QuestActionPruner`] mirrors via
    /// stateless replay.
    fn is_legal(&self, action: TokenId) -> bool {
        if self.is_terminal() {
            return false;
        }
        let (q, kind) = decode_quest_action(action);
        if q >= self.config.num_quests {
            return false;
        }
        match kind {
            QUEST_ACTION_ACCEPT => {
                self.status[q] == QuestStatus::NotStarted && self.is_available(q)
            }
            QUEST_ACTION_COMPLETE => self.status[q] == QuestStatus::Active,
            QUEST_ACTION_FAIL => self.status[q] == QuestStatus::Active,
            _ => false,
        }
    }

    fn legal_actions(&self) -> Vec<TokenId> {
        if self.is_terminal() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for q in 0..self.config.num_quests {
            match self.status[q] {
                QuestStatus::NotStarted if self.is_available(q) => {
                    out.push(encode_quest_action(q, QUEST_ACTION_ACCEPT));
                }
                QuestStatus::Active => {
                    out.push(encode_quest_action(q, QUEST_ACTION_COMPLETE));
                    out.push(encode_quest_action(q, QUEST_ACTION_FAIL));
                }
                QuestStatus::Completed | QuestStatus::Failed | QuestStatus::NotStarted => {}
            }
        }
        out
    }

    /// Terminal iff goal reached OR no quest can transition further.
    ///
    /// Computed directly (does NOT call `legal_actions` / `is_legal`) to
    /// avoid the recursion: `is_legal` calls `is_terminal`, so `is_terminal`
    /// must not call back into them.
    fn is_terminal(&self) -> bool {
        if self.is_goal() {
            return true;
        }
        for q in 0..self.config.num_quests {
            match self.status[q] {
                QuestStatus::Active => return false,
                QuestStatus::NotStarted if self.is_available(q) => return false,
                QuestStatus::Completed | QuestStatus::Failed | QuestStatus::NotStarted => {}
            }
        }
        true
    }

    fn is_goal(&self) -> bool {
        // Empty goal_quests → never a goal (vacuous truth would wrongly
        // mark the initial state as goal; guard explicitly).
        if self.config.goal_quests.is_empty() {
            return false;
        }
        self.config
            .goal_quests
            .iter()
            .all(|&g| self.status.get(g) == Some(&QuestStatus::Completed))
    }

    fn reward(&self) -> f32 {
        if self.is_goal() {
            1.0
        } else if self.is_terminal() {
            -1.0
        } else {
            0.0
        }
    }

    fn hash(&self) -> [u8; 32] {
        let mut h = Hasher::new();
        h.update(&(self.status.len() as u32).to_le_bytes());
        for &s in &self.status {
            h.update(&[s as u8]);
        }
        *h.finalize().as_bytes()
    }
}

// ── QuestActionPruner ──────────────────────────────────────────

/// Stateless-replay legality pruner for the RPG quest-tree domain.
///
/// The modelless thesis in action: the entire quest-tree legality ruleset is
/// encoded as a deterministic forward model ([`QuestState`]), and this pruner
/// mirrors it by replaying `parent_tokens` from a canonical initial state.
/// No model weights, no learned policy — just deterministic replay.
///
/// The pruner holds an immutable [`QuestConfig`] (the canonical origin).
/// [`is_valid`](ConstraintPruner::is_valid) replays `parent_tokens` to
/// reconstruct the current state, then delegates to [`QuestState::is_legal`].
/// By construction, the pruner's verdict is identical to the forward model's
/// own legality check.
///
/// # Example
///
/// ```
/// use ns_engine::pruners::{
///     encode_quest_action, quest::{QUEST_ACTION_ACCEPT, QUEST_ACTION_COMPLETE},
///     QuestActionPruner, QuestConfig, QuestState,
/// };
/// use ns_engine::traits::ConstraintPruner;
///
/// let pruner = QuestActionPruner::new(QuestConfig::default());
///
/// // In the default linear chain (Q0 → Q1 → Q2, goal Q2), only Q0 is
/// // available at the start: ACCEPT(0) is legal, ACCEPT(1) is not.
/// let accept0 = encode_quest_action(0, QUEST_ACTION_ACCEPT);
/// let accept1 = encode_quest_action(1, QUEST_ACTION_ACCEPT);
/// assert!(pruner.is_valid(0, accept0, &[]));
/// assert!(!pruner.is_valid(0, accept1, &[]));
///
/// // After ACCEPT(0), COMPLETE(0) becomes legal (Q0 is Active).
/// let complete0 = encode_quest_action(0, QUEST_ACTION_COMPLETE);
/// assert!(pruner.is_valid(1, complete0, &[accept0]));
/// ```
pub struct QuestActionPruner {
    config: QuestConfig,
}

impl QuestActionPruner {
    /// Create a new pruner with the given canonical config.
    ///
    /// The config must match the [`QuestState`] initial config used by the
    /// generate loop. A mismatch would cause the pruner's legality verdict to
    /// diverge from the real forward model.
    ///
    /// # Panics
    ///
    /// Panics if `config.validate()` returns an error.
    pub fn new(config: QuestConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("invalid QuestConfig: {e}");
        }
        Self { config }
    }

    /// Borrow the canonical config.
    pub fn config(&self) -> &QuestConfig {
        &self.config
    }

    /// Reconstruct the state after replaying `history` from the canonical
    /// initial state. Returns `None` if `history` contains an illegal action
    /// (the trace is unreachable from the canonical origin).
    pub fn state_of(&self, history: &[TokenId]) -> Option<QuestState> {
        self.replay(history)
    }

    /// Stateless replay: reconstruct the current state from `parent_tokens`.
    fn replay(&self, parent_tokens: &[TokenId]) -> Option<QuestState> {
        let mut state = QuestState::initial(self.config.clone());
        for &a in parent_tokens {
            state = state.try_step(a)?;
        }
        Some(state)
    }
}

impl ConstraintPruner for QuestActionPruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        match self.replay(parent_tokens) {
            Some(state) => state.is_legal(token),
            None => false,
        }
    }
}

impl ScreeningPruner for QuestActionPruner {
    fn arm_id(&self) -> ArmId {
        QUEST_ARM_ID
    }

    fn arm_label(&self) -> &str {
        QUEST_LABEL
    }

    fn screen(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        let state = match self.replay(parent_tokens) {
            Some(s) => s,
            None => return 0.0,
        };
        if !state.is_legal(token) {
            return 0.0;
        }
        state.action_screen_score(token)
    }
}

// ── Inline unit tests for private internals ────────────────────
//
// Integration tests covering the full public API live in `tests/quest.rs`.
// These inline tests cover the private mechanics: encode/decode round-trip,
// is_available predicate, goal_path_mask, and action_screen_score categories.

#[cfg(test)]
mod tests {
    use super::*;

    // ── encode/decode round-trip ──

    #[test]
    fn test_encode_decode_round_trip() {
        for quest in 0..10_usize {
            for kind in 0..ACTIONS_PER_QUEST as TokenId {
                let token = encode_quest_action(quest, kind);
                let (q, k) = decode_quest_action(token);
                assert_eq!(q, quest, "quest mismatch for kind {kind}");
                assert_eq!(k, kind, "kind mismatch for quest {quest}");
            }
        }
    }

    #[test]
    fn test_encode_quest_action_layout() {
        // Quest 0 occupies tokens 0,1,2; quest 1 occupies 3,4,5; etc.
        assert_eq!(encode_quest_action(0, QUEST_ACTION_ACCEPT), 0);
        assert_eq!(encode_quest_action(0, QUEST_ACTION_COMPLETE), 1);
        assert_eq!(encode_quest_action(0, QUEST_ACTION_FAIL), 2);
        assert_eq!(encode_quest_action(1, QUEST_ACTION_ACCEPT), 3);
        assert_eq!(encode_quest_action(2, QUEST_ACTION_FAIL), 8);
    }

    // ── QuestConfig::goal_path_mask ──

    #[test]
    fn test_goal_path_mask_linear_chain() {
        // Default: Q0 → Q1 → Q2, goal [Q2]. All three on the goal path.
        let cfg = QuestConfig::default();
        let mask = cfg.goal_path_mask();
        assert_eq!(mask, vec![true, true, true]);
    }

    #[test]
    fn test_goal_path_mask_with_distraction() {
        // Q0 (off-path distractor), Q1 → Q2 (goal path), goal [Q2].
        let cfg = QuestConfig {
            num_quests: 3,
            prerequisites: vec![vec![], vec![], vec![1]],
            goal_quests: vec![2],
        };
        let mask = cfg.goal_path_mask();
        assert_eq!(mask, vec![false, true, true]);
    }

    #[test]
    fn test_goal_path_mask_diamond() {
        // Q0 → Q1, Q0 → Q2, {Q1,Q2} → Q3, goal [Q3]. All on path.
        let cfg = QuestConfig {
            num_quests: 4,
            prerequisites: vec![vec![], vec![0], vec![0], vec![1, 2]],
            goal_quests: vec![3],
        };
        assert_eq!(cfg.goal_path_mask(), vec![true, true, true, true]);
    }

    // ── QuestState::is_available ──

    #[test]
    fn test_is_available_no_prereqs() {
        let state = QuestState::initial(QuestConfig::default());
        // Q0 has no prereqs → available at start.
        assert!(state.is_available(0));
        // Q1 requires Q0 (not yet completed) → not available.
        assert!(!state.is_available(1));
        // Q2 requires Q1 → not available.
        assert!(!state.is_available(2));
    }

    #[test]
    fn test_is_available_after_prereq_completed() {
        let mut state = QuestState::initial(QuestConfig::default());
        state.status[0] = QuestStatus::Completed;
        // Q1 requires Q0 (now Completed) → available.
        assert!(state.is_available(1));
        // Q0 itself is no longer NotStarted → not "available" (already done).
        assert!(!state.is_available(0));
    }

    // ── action_screen_score categories ──

    #[test]
    fn test_screen_score_complete_goal_quest() {
        // Default chain: Q2 is the goal quest. Make Q2 Active, score COMPLETE.
        let mut state = QuestState::initial(QuestConfig::default());
        state.status[2] = QuestStatus::Active;
        let a = encode_quest_action(2, QUEST_ACTION_COMPLETE);
        assert_eq!(state.action_screen_score(a), 0.95);
    }

    #[test]
    fn test_screen_score_complete_goal_path_prereq() {
        // Default chain: Q0 and Q1 are on goal path but not goal quests.
        let mut state = QuestState::initial(QuestConfig::default());
        state.status[0] = QuestStatus::Active;
        state.status[1] = QuestStatus::Active;
        let a0 = encode_quest_action(0, QUEST_ACTION_COMPLETE);
        let a1 = encode_quest_action(1, QUEST_ACTION_COMPLETE);
        assert_eq!(state.action_screen_score(a0), 0.85);
        assert_eq!(state.action_screen_score(a1), 0.85);
    }

    #[test]
    fn test_screen_score_complete_off_path_quest() {
        // Q0 off-path distractor; Q2 goal. COMPLETE Q0 = 0.6.
        let cfg = QuestConfig {
            num_quests: 3,
            prerequisites: vec![vec![], vec![], vec![1]],
            goal_quests: vec![2],
        };
        let mut state = QuestState::initial(cfg);
        state.status[0] = QuestStatus::Active;
        let a = encode_quest_action(0, QUEST_ACTION_COMPLETE);
        assert_eq!(state.action_screen_score(a), 0.6);
    }

    #[test]
    fn test_screen_score_accept_goal_path_vs_off_path() {
        let cfg = QuestConfig {
            num_quests: 3,
            prerequisites: vec![vec![], vec![], vec![1]],
            goal_quests: vec![2],
        };
        let state = QuestState::initial(cfg);
        // Q0 off-path → 0.55; Q1, Q2 on path → 0.75 (when available).
        let a0 = encode_quest_action(0, QUEST_ACTION_ACCEPT);
        let a1 = encode_quest_action(1, QUEST_ACTION_ACCEPT);
        assert_eq!(state.action_screen_score(a0), 0.55);
        assert_eq!(state.action_screen_score(a1), 0.75);
    }

    #[test]
    fn test_screen_score_fail_is_low() {
        let mut state = QuestState::initial(QuestConfig::default());
        state.status[0] = QuestStatus::Active;
        let a = encode_quest_action(0, QUEST_ACTION_FAIL);
        assert_eq!(state.action_screen_score(a), 0.2);
    }

    #[test]
    fn test_screen_score_out_of_range_quest_is_zero() {
        let state = QuestState::initial(QuestConfig::default());
        // num_quests = 3, so quest 5 is out of range.
        let a = encode_quest_action(5, QUEST_ACTION_ACCEPT);
        assert_eq!(state.action_screen_score(a), 0.0);
    }
}
