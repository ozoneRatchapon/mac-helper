//! Phase 5 — GameState + SpeculativeGenerator: domain-generalization layer.
//!
//! Phases 1–4 proved the draft → prune → verify loop on a single domain
//! (Sudoku, then KG triples). Phase 5 generalizes the same backbone to any
//! deterministic, fully-observable search domain by introducing two traits:
//!
//! - [`GameState`] — the forward-model abstraction. Owns DYNAMICS: what
//!   happens when an action fires. Pure `step` returns a successor snapshot.
//! - [`SpeculativeGenerator`] — the generate + validate contract. Bundles a
//!   state factory, draft model, pruner, and config into a runnable unit.
//!
//! The shared driver [`speculative_generate`] is the Phase 5 analog of
//! [`speculative_decode`](crate::speculative_decode): where decode builds a
//! token sequence, generate builds an ACTION TRAJECTORY by advancing a
//! forward model. The same draft → prune → commit loop drives both.
//!
//! # Ownership boundary
//!
//! | Layer              | Owns                      | Examples                 |
//! |--------------------|---------------------------|--------------------------|
//! | [`DraftModel`]     | Fluency over actions      | `UniformDraftModel`      |
//! | [`ConstraintPruner`] | Legality (cheap filter) | `SudokuPruner`, escrow   |
//! | [`GameState`]      | Dynamics (forward model)  | `ForkGame`, escrow state |
//! | [`SpeculativeGenerator`] | Wiring              | domain-specific bundles |
//!
//! The pruner filters illegal branches WITHOUT stepping (cheap); the state
//! commits legal ones via `step` (expensive). This is the speculative
//! philosophy: propose many, validate cheaply, commit rarely.

use crate::traits::{ConstraintPruner, DraftModel};
use crate::types::{DecodeConfig, GenerateResult, TokenId};

// ── GameState trait ────────────────────────────────────────────

/// A state in a deterministic, fully-observable search domain.
///
/// The forward-model abstraction for speculative game-tree exploration.
/// Where [`ConstraintPruner`] owns LEGALITY (whether an action may fire),
/// `GameState` owns DYNAMICS (what happens when it does). The two are
/// separable: the pruner filters illegal branches cheaply without advancing
/// state; `step` commits a legal action to produce the successor.
///
/// # Snapshot semantics
///
/// [`step`](Self::step) is pure: it returns a NEW state and leaves `self`
/// unchanged. Callers snapshot by cloning or by calling `step` at branch
/// points, enabling speculative exploration without undo logic. This mirrors
/// the existing pruner convention where stateless pruners derive everything
/// from `parent_tokens` (no `propagate`/`on_backtrack` work).
///
/// # Token vocabulary
///
/// Actions are [`TokenId`], so the same draft model + pruner + generate loop
/// works across Sudoku, escrow state machines, game NPCs, and quest trees.
/// Each domain maps its action space onto a contiguous range of `TokenId`s.
///
/// # Goal vs dead-end terminals
///
/// [`is_terminal`](Self::is_terminal) is true for BOTH goal states (success)
/// and dead-ends (loss / stuck). [`is_goal`](Self::is_goal) distinguishes
/// them: it is `true` only at success states and always implies
/// `is_terminal`. The generate loop backtracks from non-goal terminals and
/// stops at goal terminals.
pub trait GameState: Clone + Send + Sync {
    /// Token ID of the action that produced this state, or `None` for the
    /// initial state. Used for trajectory reconstruction and diagnostics.
    fn last_action(&self) -> Option<TokenId>;

    /// Forward model: return the successor state after applying `action`.
    ///
    /// Pure — does not mutate `self`. Implementations clone then advance.
    ///
    /// # Panics
    ///
    /// Implementations MAY panic on illegal `action`. Callers should
    /// pre-filter with a [`ConstraintPruner`] or use the fallible
    /// [`try_step`](Self::try_step) for a non-panicking variant.
    fn step(&self, action: TokenId) -> Self;

    /// Fallible forward model: `Some(successor)` if `action` is legal,
    /// `None` otherwise.
    ///
    /// Default delegates to [`is_legal`](Self::is_legal). Domains with a
    /// cheap legality predicate should override to avoid the
    /// [`legal_actions`](Self::legal_actions) enumeration.
    fn try_step(&self, action: TokenId) -> Option<Self> {
        if self.is_legal(action) {
            Some(self.step(action))
        } else {
            None
        }
    }

    /// Is `action` legal in this state?
    ///
    /// Returns `false` for terminals and for actions outside
    /// [`legal_actions`](Self::legal_actions). Default implementation
    /// enumerates legal actions and checks membership — O(n) per call.
    /// Override when a direct predicate is cheaper.
    fn is_legal(&self, action: TokenId) -> bool {
        !self.is_terminal() && self.legal_actions().contains(&action)
    }

    /// All legal actions from this state. Empty iff terminal.
    ///
    /// Order is implementation-defined. The generate loop does not depend on
    /// ordering — the draft model's logit ranking and `shuffle_tied_groups`
    /// determine exploration order.
    fn legal_actions(&self) -> Vec<TokenId>;

    /// Boolean mask over `[0..action_space)`: `true` at index `a` iff action
    /// `a` is legal.
    ///
    /// Convenience for logit masking: callers multiply logits by this mask
    /// before top-k extraction to forbid illegal actions at the draft layer.
    /// Out-of-range legal actions (above `action_space`) are dropped
    /// silently.
    fn action_mask(&self, action_space: usize) -> Vec<bool> {
        let mut mask = vec![false; action_space];
        for &a in self.legal_actions().iter() {
            if (a as usize) < action_space {
                mask[a as usize] = true;
            }
        }
        mask
    }

    /// Terminal test: no further gameplay is possible from this state.
    ///
    /// True at BOTH goals and dead-ends. Use [`is_goal`](Self::is_goal) to
    /// distinguish success terminals from loss terminals.
    fn is_terminal(&self) -> bool;

    /// Success terminal: this state satisfies the domain's objective.
    ///
    /// Always implies [`is_terminal`](Self::is_terminal). Domains whose only
    /// terminal is the goal (e.g., a reach-the-target puzzle) return `true`
    /// exactly when `is_terminal` is true. Domains with loss terminals
    /// (traps, stuck states) return `true` only at the goal — the generator
    /// backtracks from non-goal terminals.
    fn is_goal(&self) -> bool;

    /// Reward / utility of this state for the acting agent.
    ///
    /// Convention: terminal win → positive (typically `1.0`), terminal loss
    /// → negative (typically `-1.0`), non-terminal → `0.0` or a dense
    /// shaping signal. [`speculative_generate`] sums reward over every state
    /// on the final accepted path, supporting both terminal-only scoring and
    /// incremental shaping.
    fn reward(&self) -> f32;

    /// BLAKE3 hash of the state for transposition tables and cycle detection.
    ///
    /// Deterministic: same logical state → same hash. Used by future
    /// transposition-table optimizations to detect when two trajectories
    /// reach the same state.
    fn hash(&self) -> [u8; 32];
}

// ── SpeculativeGenerator trait ─────────────────────────────────

/// Generic generate + validate contract for speculative game-tree exploration.
///
/// Bundles a [`GameState`] factory, a [`DraftModel`], a [`ConstraintPruner`],
/// and a [`DecodeConfig`] into a single runnable unit. The default
/// [`generate`](Self::generate) drives the draft → prune → step loop via
/// [`speculative_generate`].
///
/// This is the Phase 5 generalization of
/// [`speculative_decode`](crate::speculative_decode): one loop, many domains.
/// Sudoku, bethere-escrow, BomberAction, QuestAction — each supplies a
/// `GameState` + `ConstraintPruner` pair; the generator drives them all.
///
/// # Why `parts` instead of separate accessors
///
/// The generate loop needs an immutable draft-model borrow, an immutable
/// config borrow, and a MUTABLE pruner borrow — all from the same `self`.
/// Rust cannot prove three separate accessor methods (`&self` × 2 + `&mut
/// self` × 1) borrow disjoint memory, so a default `generate` calling them
/// cannot compile. The [`parts`](Self::parts) method returns direct field
/// references as a tuple; implementors write it with field access (which the
/// borrow checker DOES accept as disjoint), and the default
/// [`generate`](Self::generate) works for free. This is the idiomatic Rust
/// split-borrow pattern.
pub trait SpeculativeGenerator {
    /// The domain's state type.
    type State: GameState;
    /// The pruner mirroring the state's legality rules.
    type Pruner: ConstraintPruner;

    /// Initial state for a fresh generate run.
    fn initial_state(&self) -> Self::State;

    /// Disjoint borrows of the draft model, pruner, and config.
    ///
    /// Returns direct field references so the borrow checker accepts the
    /// simultaneous immutable draft/config borrows and mutable pruner borrow
    /// that the generate loop needs. Implementors write this as
    /// `(&self.draft, &mut self.pruner, &self.config)` against their own
    /// fields — the disjoint field-access proof is what makes the split work.
    fn parts(&mut self) -> (&dyn DraftModel, &mut Self::Pruner, &DecodeConfig);

    /// Run the generate + validate loop.
    ///
    /// Default implementation delegates to [`speculative_generate`] with the
    /// components from [`parts`](Self::parts). Override only for
    /// domain-specific instrumentation (e.g., logging per-step state hashes).
    fn generate(&mut self) -> GenerateResult {
        let state = self.initial_state();
        let (draft, pruner, config) = self.parts();
        speculative_generate(&state, draft, pruner, config)
    }
}

// ── speculative_generate ───────────────────────────────────────

/// Run the speculative generate loop over a [`GameState`]: draft → prune → step.
///
/// The Phase 5 generalization of
/// [`speculative_decode`](crate::speculative_decode). Where decode builds a
/// token sequence under a pruner, generate builds an ACTION TRAJECTORY by
/// advancing a forward model. The pruner filters illegal actions cheaply
/// (without stepping); the state commits legal ones.
///
/// # Modes
///
/// - **Greedy** (`config.backtrack == false`): pick the best valid candidate
///   each step, stop on dead-end. Faithful to LLM speculative decoding.
/// - **Backtracking** (`config.backtrack == true`): DFS over the valid-only
///   subtree. Maintains a state stack for snapshot restoration on backtrack.
///   Solves planning problems with dead-end branches.
///
/// # Termination
///
/// The loop stops when:
/// - The state reaches a goal ([`GameState::is_goal`]) — success.
/// - The state reaches a non-goal terminal (dead-end):
///   - Greedy: stop with failure.
///   - Backtrack: pop the state stack and try the next candidate.
/// - `max_tokens` is reached without a terminal — budget failure.
/// - `max_attempts` is reached (backtrack mode only) — budget failure.
///
/// # Origin invariant (stateless replay pruners)
///
/// A stateless replay pruner (e.g., `ForkGamePruner` here, the escrow pruner
/// in `pruners/escrow`) derives legality by replaying `parent_tokens` from a
/// baked-in canonical origin. For the pruner's verdict to match the real
/// `GameState`, `initial` MUST equal that canonical origin. Calling
/// `speculative_generate` with a non-canonical `initial` (e.g., resuming
/// mid-search from a terminal dead-end) against a stateless replay pruner is
/// a misuse: the pruner would clear the next action against the origin, the
/// loop would step the real state, and the two would disagree.
///
/// To lift this invariant and resume from any state, supply either:
/// - a stateful pruner that tracks the current state explicitly (the bandit
///   layer's reward bookkeeping is a step in this direction), or
/// - a stateless pruner parameterized by the intended origin hash.
///
/// The goal-initial early-return path (`is_goal` checked before any draft /
/// pruner call) is exempt: it never consults the pruner, so it works with
/// any `initial`.
///
/// # Reused decode helpers
///
/// Candidate extraction (`top_k_indices`), validity filtering
/// (`valid_candidates`), tie-shuffling (`shuffle_tied_groups`), and final
/// verification (`verify_sequence`) are shared with `speculative_decode` to
/// keep the two loops behaviorally identical where they overlap.
pub fn speculative_generate<S, P>(
    initial: &S,
    draft: &dyn DraftModel,
    pruner: &mut P,
    config: &DecodeConfig,
) -> GenerateResult
where
    S: GameState,
    P: ConstraintPruner,
{
    let mut actions: Vec<TokenId> = Vec::with_capacity(config.max_tokens);
    let mut attempts = 0u64;

    let (reached_goal, state_stack) = match config.backtrack {
        true => {
            generate_with_backtrack(initial, draft, pruner, config, &mut actions, &mut attempts)
        }
        false => generate_greedy(initial, draft, pruner, config, &mut actions, &mut attempts),
    };

    let terminal = state_stack.last().map(|s| s.is_terminal()).unwrap_or(false);
    let reward: f32 = state_stack.iter().map(|s| s.reward()).sum();
    let hash = GenerateResult::hash_actions(&actions);

    GenerateResult {
        actions,
        terminal,
        goal: reached_goal,
        reward,
        attempts,
        hash,
    }
}

/// Greedy generate: pick the best valid candidate at each step, stop on dead-end.
///
/// Mirrors `decode_greedy`: at each depth, draft logits, take top-k, filter
/// through the pruner, accept the first valid candidate. If no valid
/// candidate exists OR a non-goal terminal is reached, the run stops with
/// `reached_goal == false`.
fn generate_greedy<S, P>(
    initial: &S,
    draft: &dyn DraftModel,
    pruner: &mut P,
    config: &DecodeConfig,
    actions: &mut Vec<TokenId>,
    attempts: &mut u64,
) -> (bool, Vec<S>)
where
    S: GameState,
    P: ConstraintPruner,
{
    let mut state_stack: Vec<S> = vec![initial.clone()];

    loop {
        let depth = actions.len();

        let is_goal = state_stack.last().expect("state stack non-empty").is_goal();
        if is_goal {
            return (true, state_stack);
        }

        if depth >= config.max_tokens {
            return (false, state_stack);
        }

        let logits = draft.log_probs(actions);
        let candidates = crate::decode::top_k_indices(&logits, config.top_k);
        let valid = crate::decode::valid_candidates(pruner, depth, actions, &candidates);

        *attempts += 1;

        match valid.first() {
            Some(&action) => {
                let next = state_stack
                    .last()
                    .expect("state stack non-empty")
                    .step(action);
                actions.push(action);
                state_stack.push(next);
                // Propagate AFTER the push: `ConstraintPruner::propagate`
                // requires the post-commit slice (len == depth + 1), matching
                // `speculative_decode`. Passing the pre-push slice here made
                // stateful pruners key differently in the two loops.
                pruner.propagate(depth, action, actions);
            }
            None => return (false, state_stack),
        }
    }
}

/// Backtracking generate: DFS over the valid-only subtree.
///
/// Mirrors `decode_with_backtrack` with one addition: a state stack
/// (`state_stack`) alongside the action stack, used to restore the
/// [`GameState`] snapshot on backtrack. Because [`GameState::step`] is pure,
/// popping the state stack is the only "undo" needed — no per-domain
/// `on_backtrack` state repair is required on the GameState side.
///
/// # Dead-end handling
///
/// Non-goal terminal states have no legal actions, so the pruner (which
/// mirrors [`GameState::legal_actions`]) rejects every candidate, yielding
/// an empty `untried[depth]`. The next `pop` returns `None`, triggering
/// backtracking. Goal terminals are caught by the early `is_goal` return
/// before candidate generation.
fn generate_with_backtrack<S, P>(
    initial: &S,
    draft: &dyn DraftModel,
    pruner: &mut P,
    config: &DecodeConfig,
    actions: &mut Vec<TokenId>,
    attempts: &mut u64,
) -> (bool, Vec<S>)
where
    S: GameState,
    P: ConstraintPruner,
{
    let mut rng = fastrand::Rng::with_seed(config.seed);
    let mut state_stack: Vec<S> = vec![initial.clone()];
    // untried[d] = remaining valid candidates at depth d, lazily populated.
    let mut untried: Vec<Vec<TokenId>> = Vec::with_capacity(config.max_tokens);

    loop {
        let depth = actions.len();

        let is_goal = state_stack.last().expect("state stack non-empty").is_goal();
        if is_goal {
            return (true, state_stack);
        }

        if depth >= config.max_tokens {
            return (false, state_stack);
        }
        if *attempts >= config.max_attempts {
            return (false, state_stack);
        }

        // First visit to this depth: generate valid candidates.
        if depth >= untried.len() {
            let logits = draft.log_probs(actions);
            let candidates = crate::decode::top_k_indices(&logits, config.top_k);
            let mut valid = crate::decode::valid_candidates(pruner, depth, actions, &candidates);
            // Shuffle within tied-logit groups for randomized DFS.
            // Same seed → same shuffle → same solution path.
            crate::decode::shuffle_tied_groups(&mut valid, &logits, &mut rng);
            // valid is best-first, but pop() consumes from the end — reverse
            // so the best candidate really is "on top" of the stack.
            valid.reverse();
            untried.push(valid);
        }

        *attempts += 1;

        match untried[depth].pop() {
            Some(action) => {
                let next = state_stack
                    .last()
                    .expect("state stack non-empty")
                    .step(action);
                actions.push(action);
                state_stack.push(next);
                // Post-commit slice, matching decode.rs — see the greedy path.
                pruner.propagate(depth, action, actions);
            }
            None => {
                // Dead-end at this depth — backtrack.
                if actions.is_empty() {
                    return (false, state_stack);
                }
                let action = actions.pop().expect("actions non-empty");
                state_stack.pop();
                let popped_depth = actions.len();
                pruner.on_backtrack(popped_depth, action, actions);
                // Clear stale candidates beyond the new depth — they were
                // generated for a different prefix and must be regenerated.
                untried.truncate(depth);
            }
        }
    }
}
