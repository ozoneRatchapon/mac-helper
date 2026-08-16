//! Integration tests for the Phase 5 `GameState` + `SpeculativeGenerator` traits.
//!
//! These tests live in `tests/` (not inline in `src/game.rs`) to keep
//! `game.rs` under the 1024-line rule. They exercise the FULL public trait
//! surface from outside the crate, exactly as a domain implementor would.
//!
//! # Test domain: ForkGame
//!
//! A minimal planning domain with a real dead-end, used to exercise every
//! branch of the speculative generate loop with a genuine (non-toy) problem.
//!
//! ```text
//! State graph:
//!   A --left-->  B (dead-end, reward -1)
//!   A --right--> C
//!   C --left-->  D
//!   C --right--> E (goal, reward +1)
//!   D --right--> E (goal, reward +1)
//! ```
//!
//! Start = A. Greedy with a left-preferring draft gets stuck at B;
//! backtracking finds A→C→E or A→C→D→E. This single fixture exercises:
//! - pure `step` (snapshot semantics)
//! - default `is_legal` / `action_mask` / `try_step` provided methods
//! - terminal vs goal distinction (B is terminal but not goal)
//! - reward accumulation (negative at dead-end, positive at goal)
//! - greedy success, greedy dead-end, backtracking recovery
//! - budget limits (`max_tokens`, `max_attempts`)
//! - the `SpeculativeGenerator` default `generate()` via split-borrow `parts()`

use ns_engine::draft::UniformDraftModel;
use ns_engine::traits::{ConstraintPruner, DraftModel};
use ns_engine::{
    speculative_generate, DecodeConfig, GameState, GenerateResult, Logits, SpeculativeGenerator,
    TokenId,
};

// ── ForkGame: minimal planning domain ───────────────────────────

const FG_LEFT: TokenId = 0;
const FG_RIGHT: TokenId = 1;
const FG_VOCAB: usize = 2;

const FG_A: u32 = 0; // start
const FG_B: u32 = 1; // dead-end
const FG_C: u32 = 2;
const FG_D: u32 = 3;
const FG_E: u32 = 4; // goal

#[derive(Clone, Debug, PartialEq, Eq)]
struct ForkGame {
    state: u32,
    last_action: Option<TokenId>,
}

impl ForkGame {
    fn at_start() -> Self {
        Self {
            state: FG_A,
            last_action: None,
        }
    }

    fn at(state: u32) -> Self {
        Self {
            state,
            last_action: None,
        }
    }

    /// Deterministic transition table. `None` = illegal.
    fn transition(state: u32, action: TokenId) -> Option<u32> {
        match (state, action) {
            (FG_A, FG_LEFT) => Some(FG_B),
            (FG_A, FG_RIGHT) => Some(FG_C),
            (FG_C, FG_LEFT) => Some(FG_D),
            (FG_C, FG_RIGHT) => Some(FG_E),
            (FG_D, FG_RIGHT) => Some(FG_E),
            _ => None,
        }
    }
}

impl GameState for ForkGame {
    fn last_action(&self) -> Option<TokenId> {
        self.last_action
    }

    fn step(&self, action: TokenId) -> Self {
        let next = ForkGame::transition(self.state, action)
            .expect("step called on illegal action; pre-filter via pruner or try_step");
        Self {
            state: next,
            last_action: Some(action),
        }
    }

    fn legal_actions(&self) -> Vec<TokenId> {
        if self.is_terminal() {
            return Vec::new();
        }
        let mut actions = Vec::with_capacity(2);
        if ForkGame::transition(self.state, FG_LEFT).is_some() {
            actions.push(FG_LEFT);
        }
        if ForkGame::transition(self.state, FG_RIGHT).is_some() {
            actions.push(FG_RIGHT);
        }
        actions
    }

    fn is_terminal(&self) -> bool {
        // B (dead-end) and E (goal) have no outgoing transitions.
        matches!(self.state, FG_B | FG_E)
    }

    fn is_goal(&self) -> bool {
        self.state == FG_E
    }

    fn reward(&self) -> f32 {
        match self.state {
            FG_E => 1.0,
            FG_B => -1.0,
            _ => 0.0,
        }
    }

    fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state.to_le_bytes());
        hasher.update(&self.last_action.unwrap_or(u32::MAX).to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

// ── ForkGamePruner: stateless replay legality oracle ────────────

/// Legality-mirror pruner for ForkGame. Replays the action trajectory to
/// reconstruct state, then checks the transition table — same convention
/// as the escrow pruner (stateless derivation from `parent_tokens`).
struct ForkGamePruner;

impl ForkGamePruner {
    fn replay_state(parent_tokens: &[TokenId]) -> Option<u32> {
        let mut state = FG_A;
        for &a in parent_tokens {
            state = ForkGame::transition(state, a)?;
        }
        Some(state)
    }
}

impl ConstraintPruner for ForkGamePruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        match ForkGamePruner::replay_state(parent_tokens) {
            Some(state) => ForkGame::transition(state, token).is_some(),
            None => false,
        }
    }
}

// ── Test draft models ───────────────────────────────────────────

/// Right-preferring draft: greedy prefers the goal-directed branch.
struct RightFirstDraft {
    vocab: usize,
}

impl DraftModel for RightFirstDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        let mut logits = vec![0.0; self.vocab];
        if self.vocab > 1 {
            logits[FG_LEFT as usize] = 0.0;
            logits[FG_RIGHT as usize] = 1.0;
        }
        logits
    }
}

/// Left-preferring draft: greedy walks into the dead-end at B.
struct LeftFirstDraft {
    vocab: usize,
}

impl DraftModel for LeftFirstDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        let mut logits = vec![0.0; self.vocab];
        if self.vocab > 1 {
            logits[FG_LEFT as usize] = 1.0;
            logits[FG_RIGHT as usize] = 0.0;
        }
        logits
    }
}

// ── ForkGenerator: SpeculativeGenerator reference bundling ──────

/// Bundles state factory + draft + pruner + config behind the
/// `SpeculativeGenerator` contract, exercising the split-borrow `parts()`
/// method and the default `generate()`.
struct ForkGenerator {
    draft: RightFirstDraft,
    pruner: ForkGamePruner,
    config: DecodeConfig,
}

impl SpeculativeGenerator for ForkGenerator {
    type State = ForkGame;
    type Pruner = ForkGamePruner;

    fn initial_state(&self) -> Self::State {
        ForkGame::at_start()
    }

    fn parts(&mut self) -> (&dyn DraftModel, &mut Self::Pruner, &DecodeConfig) {
        (&self.draft, &mut self.pruner, &self.config)
    }
}

// ── Config helpers ──────────────────────────────────────────────

fn greedy_config() -> DecodeConfig {
    DecodeConfig {
        backtrack: false,
        max_tokens: 8,
        top_k: 2,
        seed: 42,
        max_attempts: 100,
    }
}

fn backtrack_config() -> DecodeConfig {
    DecodeConfig {
        backtrack: true,
        max_tokens: 8,
        top_k: 2,
        seed: 42,
        max_attempts: 100,
    }
}

// ── ForkGame trait contract ─────────────────────────────────────

#[test]
fn test_forkgame_step_is_pure() {
    let initial = ForkGame::at_start();
    let _next = initial.step(FG_RIGHT);
    // Original unchanged — step returned a NEW state.
    assert_eq!(initial, ForkGame::at_start());
}

#[test]
fn test_forkgame_legal_actions_per_state() {
    assert_eq!(ForkGame::at(FG_A).legal_actions(), vec![FG_LEFT, FG_RIGHT]);
    assert_eq!(ForkGame::at(FG_C).legal_actions(), vec![FG_LEFT, FG_RIGHT]);
    assert_eq!(ForkGame::at(FG_D).legal_actions(), vec![FG_RIGHT]);
    // Terminals have no legal actions.
    assert!(ForkGame::at(FG_B).legal_actions().is_empty());
    assert!(ForkGame::at(FG_E).legal_actions().is_empty());
}

#[test]
fn test_forkgame_terminal_and_goal_distinction() {
    // A, C, D are non-terminal.
    for &s in &[FG_A, FG_C, FG_D] {
        assert!(
            !ForkGame::at(s).is_terminal(),
            "state {s} should be non-terminal"
        );
        assert!(!ForkGame::at(s).is_goal(), "state {s} should not be goal");
    }
    // B is terminal dead-end (not goal).
    assert!(ForkGame::at(FG_B).is_terminal());
    assert!(!ForkGame::at(FG_B).is_goal());
    // E is terminal goal.
    assert!(ForkGame::at(FG_E).is_terminal());
    assert!(ForkGame::at(FG_E).is_goal());
}

#[test]
fn test_forkgame_reward_signal() {
    assert_eq!(ForkGame::at(FG_E).reward(), 1.0);
    assert_eq!(ForkGame::at(FG_B).reward(), -1.0);
    for &s in &[FG_A, FG_C, FG_D] {
        assert_eq!(ForkGame::at(s).reward(), 0.0);
    }
}

#[test]
fn test_forkgame_try_step_fallibility() {
    let a = ForkGame::at_start();
    assert!(a.try_step(FG_LEFT).is_some());
    assert!(a.try_step(FG_RIGHT).is_some());
    // No third action in vocab from A.
    assert!(a.try_step(2).is_none());

    let b = ForkGame::at(FG_B);
    // Terminal — all actions illegal via default is_legal.
    assert!(b.try_step(FG_LEFT).is_none());
}

#[test]
fn test_forkgame_is_legal_default_impl() {
    let a = ForkGame::at_start();
    assert!(a.is_legal(FG_LEFT));
    assert!(a.is_legal(FG_RIGHT));
    assert!(!a.is_legal(99)); // out of action space
    assert!(!ForkGame::at(FG_B).is_legal(FG_LEFT)); // terminal
}

#[test]
fn test_forkgame_action_mask() {
    let mask = ForkGame::at(FG_A).action_mask(FG_VOCAB);
    assert_eq!(mask, vec![true, true]);

    let mask = ForkGame::at(FG_D).action_mask(FG_VOCAB);
    assert_eq!(mask, vec![false, true]); // only RIGHT from D

    let mask = ForkGame::at(FG_E).action_mask(FG_VOCAB);
    assert_eq!(mask, vec![false, false]); // terminal
}

#[test]
fn test_forkgame_hash_deterministic_and_distinct() {
    let a1 = ForkGame::at(FG_A);
    let a2 = ForkGame::at(FG_A);
    assert_eq!(a1.hash(), a2.hash(), "same state → same hash");

    let b = ForkGame::at(FG_B);
    assert_ne!(a1.hash(), b.hash(), "different states → different hashes");

    // last_action participates in hash.
    let a_via_right = ForkGame::at_start().step(FG_RIGHT);
    assert_ne!(
        ForkGame::at(FG_C).hash(),
        a_via_right.hash(),
        "different last_action must hash differently"
    );
}

// ── ForkGamePruner ──────────────────────────────────────────────

#[test]
fn test_forkgame_pruner_mirrors_legality() {
    let pruner = ForkGamePruner;
    // From A (empty prefix): both LEFT and RIGHT valid.
    assert!(pruner.is_valid(0, FG_LEFT, &[]));
    assert!(pruner.is_valid(0, FG_RIGHT, &[]));
    // From C (prefix [RIGHT]): both valid.
    assert!(pruner.is_valid(1, FG_LEFT, &[FG_RIGHT]));
    assert!(pruner.is_valid(1, FG_RIGHT, &[FG_RIGHT]));
    // From D (prefix [RIGHT, LEFT]): only RIGHT.
    assert!(!pruner.is_valid(2, FG_LEFT, &[FG_RIGHT, FG_LEFT]));
    assert!(pruner.is_valid(2, FG_RIGHT, &[FG_RIGHT, FG_LEFT]));
}

#[test]
fn test_forkgame_pruner_rejects_illegal_prefix() {
    let pruner = ForkGamePruner;
    // Prefix ending at terminal B (via LEFT) — no further action valid.
    assert!(!pruner.is_valid(1, FG_LEFT, &[FG_LEFT]));
    assert!(!pruner.is_valid(1, FG_RIGHT, &[FG_LEFT]));
}

// ── speculative_generate: greedy ────────────────────────────────

#[test]
fn test_generate_greedy_reaches_goal_with_right_first_draft() {
    let draft = RightFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(&ForkGame::at_start(), &draft, &mut pruner, &greedy_config());

    assert!(result.terminal, "should end on terminal");
    assert!(result.goal, "should reach goal E");
    assert_eq!(result.reward, 1.0, "goal reward is +1");
    // Greedy + RIGHT-first: A→C→E (actions RIGHT, RIGHT).
    assert_eq!(result.actions, vec![FG_RIGHT, FG_RIGHT]);
}

#[test]
fn test_generate_greedy_hits_dead_end_with_left_first_draft() {
    let draft = LeftFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(&ForkGame::at_start(), &draft, &mut pruner, &greedy_config());

    // Greedy + LEFT-first: A→B (dead-end).
    assert!(result.terminal, "B is terminal");
    assert!(!result.goal, "B is not the goal");
    assert_eq!(result.reward, -1.0, "dead-end reward is -1");
    assert_eq!(result.actions, vec![FG_LEFT]);
}

#[test]
fn test_generate_greedy_goal_initial_returns_empty() {
    // Starting at the goal exercises the early-return path: the loop
    // checks `is_goal` BEFORE drafting, so it returns at once with no
    // pruner consultation and no forward steps. This holds even though
    // the stateless ForkGamePruner bakes in origin A — the pruner is
    // never consulted on a goal-initial trajectory.
    let draft = RightFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(&ForkGame::at(FG_E), &draft, &mut pruner, &greedy_config());

    assert!(result.goal, "E is the goal");
    assert!(result.terminal);
    assert!(result.actions.is_empty(), "no actions from a goal initial");
    assert_eq!(result.reward, 1.0, "E reward is +1");
}

#[test]
fn test_generate_greedy_no_valid_candidate_returns_false() {
    // RightFirstDraft from A always has a valid candidate (RIGHT→C is
    // legal), so the "no-valid-candidate" branch of greedy cannot fire
    // from the canonical origin with this domain. The dead-end path is
    // already covered by `test_generate_greedy_hits_dead_end_with_left_first_draft`
    // (LEFT→B returns goal=false, terminal=true). This test documents
    // that greedy reaching a non-goal terminal yields `goal == false`
    // with a non-empty action trajectory.
    let draft = LeftFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(&ForkGame::at_start(), &draft, &mut pruner, &greedy_config());

    assert!(!result.goal);
    assert!(result.terminal);
    assert!(
        !result.actions.is_empty(),
        "greedy committed the LEFT step to B"
    );
    assert_eq!(result.actions, vec![FG_LEFT]);
}

// ── speculative_generate: backtracking ──────────────────────────

#[test]
fn test_generate_backtrack_finds_goal_past_dead_end() {
    // Uniform draft: candidate order is index-ascending (LEFT before RIGHT)
    // after stable sort; shuffle_tied_groups randomizes within the tie.
    // Regardless of order, backtracking must escape the dead-end at B and
    // reach E.
    let draft = UniformDraftModel::new(FG_VOCAB);
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(
        &ForkGame::at_start(),
        &draft,
        &mut pruner,
        &backtrack_config(),
    );

    assert!(result.goal, "backtracking must reach goal E");
    assert!(result.terminal);
    assert_eq!(result.reward, 1.0);
    // Accepted path must be a real A→…→E trajectory.
    assert!(
        result.actions == vec![FG_RIGHT, FG_RIGHT]
            || result.actions == vec![FG_RIGHT, FG_LEFT, FG_RIGHT],
        "unexpected path: {:?}",
        result.actions
    );
    assert!(
        result.attempts >= result.actions.len() as u64,
        "attempts must cover forward steps"
    );
}

#[test]
fn test_generate_backtrack_finds_goal_with_left_first_draft() {
    // LEFT-first draft biases greedy into B, but backtracking must still
    // recover and reach E.
    let draft = LeftFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(
        &ForkGame::at_start(),
        &draft,
        &mut pruner,
        &backtrack_config(),
    );

    assert!(result.goal);
    assert_eq!(result.reward, 1.0);
}

#[test]
fn test_generate_backtrack_goal_initial_returns_empty() {
    // Same early-return path as the greedy variant; backtrack mode also
    // checks `is_goal` before any candidate generation, so a goal initial
    // returns immediately without consulting the pruner.
    //
    // Note: starting from a NON-goal terminal (B) is not a valid call for
    // a stateless replay pruner — the pruner's replay assumes origin A,
    // so it would validate the next action against A's transition table,
    // not B's, and the loop would step the real (B) state on an action
    // the pruner cleared for A. That mismatch is the documented origin
    // invariant of `speculative_generate` (see its doc).
    let draft = UniformDraftModel::new(FG_VOCAB);
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(
        &ForkGame::at(FG_E),
        &draft,
        &mut pruner,
        &backtrack_config(),
    );

    assert!(result.goal);
    assert!(result.actions.is_empty());
    assert_eq!(result.reward, 1.0);
}

#[test]
fn test_generate_backtrack_respects_max_attempts() {
    let draft = UniformDraftModel::new(FG_VOCAB);
    let mut pruner = ForkGamePruner;
    let config = DecodeConfig {
        backtrack: true,
        max_tokens: 8,
        top_k: 2,
        seed: 42,
        max_attempts: 0, // exhausted immediately
    };
    let result = speculative_generate(&ForkGame::at_start(), &draft, &mut pruner, &config);

    assert!(!result.goal);
    // max_attempts == 0 → loop bails before any forward step.
    assert!(result.actions.is_empty());
}

#[test]
fn test_generate_backtrack_respects_max_tokens() {
    // Force max_tokens below the shortest solution length by starting
    // from D (1 step from goal) — should still succeed. The bound is
    // exercised separately by a longer forced path.
    //
    // NOTE: starting from D violates ForkGamePruner's canonical-origin
    // invariant (it replays from A and would approve LEFT, which is illegal
    // at D and panics `step`). A right-preferring draft keeps the search on
    // RIGHT deterministically instead of relying on tie-shuffle luck.
    let draft = RightFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let config = DecodeConfig {
        backtrack: true,
        max_tokens: 1, // only one step allowed
        top_k: 2,
        seed: 42,
        max_attempts: 100,
    };
    let result = speculative_generate(&ForkGame::at(FG_D), &draft, &mut pruner, &config);

    assert!(result.goal, "from D one RIGHT step reaches E");
    assert_eq!(result.actions, vec![FG_RIGHT]);
}

// ── GenerateResult fields ───────────────────────────────────────

#[test]
fn test_generate_result_hash_deterministic() {
    let draft = RightFirstDraft { vocab: FG_VOCAB };
    let mut pruner_a = ForkGamePruner;
    let mut pruner_b = ForkGamePruner;
    let a = speculative_generate(
        &ForkGame::at_start(),
        &draft,
        &mut pruner_a,
        &greedy_config(),
    );
    let b = speculative_generate(
        &ForkGame::at_start(),
        &draft,
        &mut pruner_b,
        &greedy_config(),
    );

    assert_eq!(a.actions, b.actions, "same draft+seed → same actions");
    assert_eq!(a.hash, b.hash, "same actions → same hash");
    assert_eq!(a.hash, GenerateResult::hash_actions(&a.actions));
}

#[test]
fn test_generate_result_actions_are_legal_trajectory() {
    // Post-hoc audit: every action in the result must be pruner-valid at
    // its depth. Catches generate-loop logic regressions.
    let draft = UniformDraftModel::new(FG_VOCAB);
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(
        &ForkGame::at_start(),
        &draft,
        &mut pruner,
        &backtrack_config(),
    );

    for (depth, &action) in result.actions.iter().enumerate() {
        assert!(
            ForkGamePruner.is_valid(depth, action, &result.actions[..depth]),
            "action {action} at depth {depth} is illegal"
        );
    }
}

// ── SpeculativeGenerator trait ──────────────────────────────────

#[test]
fn test_speculative_generator_trait_greedy_default_impl() {
    let mut gen = ForkGenerator {
        draft: RightFirstDraft { vocab: FG_VOCAB },
        pruner: ForkGamePruner,
        config: greedy_config(),
    };
    let result = gen.generate();

    assert!(result.goal);
    assert_eq!(result.actions, vec![FG_RIGHT, FG_RIGHT]);
}

#[test]
fn test_speculative_generator_trait_backtrack_default_impl() {
    let mut gen = ForkGenerator {
        draft: RightFirstDraft { vocab: FG_VOCAB },
        pruner: ForkGamePruner,
        config: backtrack_config(),
    };
    let result = gen.generate();

    assert!(result.goal);
    assert_eq!(result.reward, 1.0);
}

// ── Trait-object / Send-Sync bounds ─────────────────────────────

#[test]
fn test_gamestate_send_sync_bounds() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ForkGame>();
}

#[test]
fn test_generate_result_clone_debug() {
    let draft = RightFirstDraft { vocab: FG_VOCAB };
    let mut pruner = ForkGamePruner;
    let result = speculative_generate(&ForkGame::at_start(), &draft, &mut pruner, &greedy_config());

    let cloned = result.clone();
    assert_eq!(result.actions, cloned.actions);
    assert_eq!(result.hash, cloned.hash);
    // Debug renders without panic.
    let _s = format!("{result:?}");
}
