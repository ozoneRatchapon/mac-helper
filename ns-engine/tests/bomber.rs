//! Integration tests for the BomberActionPruner + BomberState domain.
//!
//! Phase 5 domain-generalization deliverable. Exercises the full public API
//! of `BomberState` (forward model) and `BomberActionPruner` (stateless
//! replay legality mirror) from outside the crate. Covers:
//!
//! - config validation (all error variants)
//! - `GameState` trait surface (purity, legality, terminals, rewards, hash)
//! - bomb mechanics (placement, fuse ticking, detonation, retreat survival)
//! - block destruction opens paths; death by own bomb; victory at exit
//! - pruner legality mirror (property test over the reachable state space)
//! - graded `ScreeningPruner` scores (distance-to-exit, bomb utility)
//! - `speculative_generate`: greedy success/dead-end, backtrack recovery,
//!   deterministic-under-seed, `SpeculativeGenerator` bundle

use ns_engine::pruners::{
    Bomb, BomberActionPruner, BomberConfig, BomberConfigError, BomberState, Cell, ACTION_MOVE_E,
    ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_W, ACTION_PLACE_BOMB, ACTION_WAIT, BOMBER_ARM_ID,
    BOMBER_LABEL, BOMBER_VOCAB,
};
use ns_engine::traits::{ConstraintPruner, DraftModel, ScreeningPruner};
use ns_engine::{
    speculative_generate, DecodeConfig, GameState, Logits, SpeculativeGenerator, TokenId,
};

// ── helpers ────────────────────────────────────────────────────

/// Manhattan distance between two grid indices given `width`.
fn manhattan(a: usize, b: usize, width: usize) -> usize {
    let ax = a % width;
    let ay = a / width;
    let bx = b % width;
    let by = b / width;
    (ax as isize - bx as isize).unsigned_abs() + (ay as isize - by as isize).unsigned_abs()
}

/// `(dx, dy)` delta for a move action, or `None` for non-move actions.
fn delta(action: TokenId) -> Option<(isize, isize)> {
    match action {
        ACTION_MOVE_N => Some((0, -1)),
        ACTION_MOVE_S => Some((0, 1)),
        ACTION_MOVE_E => Some((1, 0)),
        ACTION_MOVE_W => Some((-1, 0)),
        _ => None,
    }
}

// ── test fixtures ──────────────────────────────────────────────

/// The corridor puzzle: a forced-bomb scenario.
///
/// Grid layout (3 wide × 4 tall, row-major):
///
/// ```text
/// col:   0   1   2
/// row0:  P   .   .         (0  1  2)   player start; open top
/// row1:  W   .   W         (3  4  5)
/// row2:  W   B   W         (6  7  8)   block — only path to exit
/// row3:  W   E   W         (9 10 11)   exit behind the block
/// ```
///
/// The exit (index 10) is reachable ONLY through the block at index 7:
/// navigate to 4, place a bomb, retreat two cells north to escape the
/// blast, then proceed south through the destroyed block to the exit.
fn corridor_puzzle_config() -> BomberConfig {
    let grid = vec![
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Wall,
        Cell::Floor,
        Cell::Wall,
        Cell::Wall,
        Cell::Block,
        Cell::Wall,
        Cell::Wall,
        Cell::Exit,
        Cell::Wall,
    ];
    BomberConfig {
        width: 3,
        height: 4,
        initial_grid: grid,
        start: 0,
        bomb_fuse: 3,
        blast_radius: 1,
    }
}

/// Config whose start cell IS the exit (already-won at construction).
fn start_on_exit_config() -> BomberConfig {
    let grid = vec![
        Cell::Exit,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
    ];
    BomberConfig {
        width: 3,
        height: 3,
        initial_grid: grid,
        start: 0,
        bomb_fuse: 3,
        blast_radius: 1,
    }
}

// ── Section 1: config validation ───────────────────────────────

#[test]
fn test_config_default_validates() {
    assert!(BomberConfig::default().validate().is_ok());
}

#[test]
fn test_config_corridor_puzzle_validates() {
    assert!(corridor_puzzle_config().validate().is_ok());
}

#[test]
fn test_config_error_zero_dimension() {
    let cfg = BomberConfig {
        width: 0,
        ..BomberConfig::default()
    };
    assert_eq!(cfg.validate(), Err(BomberConfigError::ZeroDimension));
}

#[test]
fn test_config_error_grid_size_mismatch() {
    let cfg = BomberConfig {
        initial_grid: vec![Cell::Floor; 5], // should be 9 for 3×3
        ..BomberConfig::default()
    };
    assert_eq!(cfg.validate(), Err(BomberConfigError::GridSizeMismatch));
}

#[test]
fn test_config_error_start_out_of_bounds() {
    let cfg = BomberConfig {
        start: 99,
        ..BomberConfig::default()
    };
    assert_eq!(cfg.validate(), Err(BomberConfigError::StartOutOfBounds));
}

#[test]
fn test_config_error_start_not_walkable() {
    let grid = vec![
        Cell::Wall,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Floor,
        Cell::Exit,
    ];
    let cfg = BomberConfig {
        width: 3,
        height: 3,
        initial_grid: grid,
        start: 0,
        bomb_fuse: 3,
        blast_radius: 1,
    };
    assert_eq!(cfg.validate(), Err(BomberConfigError::StartNotWalkable));
}

#[test]
fn test_config_error_zero_fuse() {
    let cfg = BomberConfig {
        bomb_fuse: 0,
        ..BomberConfig::default()
    };
    assert_eq!(cfg.validate(), Err(BomberConfigError::ZeroFuse));
}

#[test]
fn test_config_error_zero_blast_radius() {
    let cfg = BomberConfig {
        blast_radius: 0,
        ..BomberConfig::default()
    };
    assert_eq!(cfg.validate(), Err(BomberConfigError::ZeroBlastRadius));
}

#[test]
fn test_config_exit_index() {
    assert_eq!(BomberConfig::default().exit_index(), Some(8));
    assert_eq!(corridor_puzzle_config().exit_index(), Some(10));
    let no_exit = BomberConfig {
        width: 2,
        height: 1,
        initial_grid: vec![Cell::Floor, Cell::Floor],
        ..BomberConfig::default()
    };
    assert_eq!(no_exit.exit_index(), None);
}

#[test]
#[should_panic(expected = "invalid BomberConfig")]
fn test_state_initial_panics_on_invalid_config() {
    let bad = BomberConfig {
        width: 0,
        height: 0,
        initial_grid: vec![],
        ..BomberConfig::default()
    };
    let _ = BomberState::initial(bad);
}

// ── Section 2: initial state ───────────────────────────────────

#[test]
fn test_initial_state_default() {
    let s = BomberState::initial(BomberConfig::default());
    assert_eq!(s.player, 0);
    assert!(s.bombs.is_empty());
    assert!(s.alive);
    assert!(!s.won);
    assert!(!s.is_terminal());
    assert!(!s.is_goal());
    assert_eq!(s.reward(), 0.0);
    assert_eq!(s.last_action(), None);
}

#[test]
fn test_initial_state_on_exit_is_goal() {
    let s = BomberState::initial(start_on_exit_config());
    assert!(s.won);
    assert!(s.is_terminal());
    assert!(s.is_goal());
    assert_eq!(s.reward(), 1.0);
    assert!(s.legal_actions().is_empty());
}

#[test]
fn test_describe_smoke() {
    let s = BomberState::initial(BomberConfig::default());
    let d = s.describe();
    assert!(d.contains("player=0"));
    assert!(d.contains("alive=true"));
    assert!(d.contains("won=false"));
}

// ── Section 3: GameState surface ───────────────────────────────

#[test]
fn test_step_is_pure() {
    let s = BomberState::initial(BomberConfig::default());
    let original = s.clone();
    let _next = s.step(ACTION_MOVE_E);
    assert_eq!(s, original, "step must not mutate the receiver");
}

#[test]
fn test_step_records_last_action() {
    let s = BomberState::initial(BomberConfig::default());
    let next = s.step(ACTION_MOVE_S);
    assert_eq!(next.last_action(), Some(ACTION_MOVE_S));
}

#[test]
fn test_legal_actions_initial_open_grid() {
    let s = BomberState::initial(BomberConfig::default());
    let legal = s.legal_actions();
    // From (0,0): MOVE_E, MOVE_S, PLACE_BOMB, WAIT. (MOVE_N, MOVE_W OOB.)
    assert!(legal.contains(&ACTION_MOVE_E));
    assert!(legal.contains(&ACTION_MOVE_S));
    assert!(legal.contains(&ACTION_PLACE_BOMB));
    assert!(legal.contains(&ACTION_WAIT));
    assert!(!legal.contains(&ACTION_MOVE_N));
    assert!(!legal.contains(&ACTION_MOVE_W));
}

#[test]
fn test_is_legal_overrides_default_matches_legal_actions() {
    let s = BomberState::initial(corridor_puzzle_config());
    let at_4 = s.step(ACTION_MOVE_E).step(ACTION_MOVE_S);
    for a in 0..BOMBER_VOCAB as TokenId {
        let in_list = at_4.legal_actions().contains(&a);
        assert_eq!(
            at_4.is_legal(a),
            in_list,
            "is_legal({a}) must match legal_actions"
        );
    }
}

#[test]
fn test_action_mask_open_grid_start() {
    let s = BomberState::initial(BomberConfig::default());
    let mask = s.action_mask(BOMBER_VOCAB);
    // indices: 0=N,1=S,2=E,3=W,4=BOMB,5=WAIT
    assert!(!mask[0]); // N OOB
    assert!(mask[1]); // S
    assert!(mask[2]); // E
    assert!(!mask[3]); // W OOB
    assert!(mask[4]); // BOMB
    assert!(mask[5]); // WAIT
}

#[test]
fn test_action_mask_truncates_out_of_range() {
    let s = BomberState::initial(BomberConfig::default());
    let mask = s.action_mask(3);
    assert_eq!(mask.len(), 3);
    assert!(!mask[0]); // N
    assert!(mask[1]); // S
    assert!(mask[2]); // E
}

#[test]
fn test_try_step_legal_and_illegal() {
    let s = BomberState::initial(BomberConfig::default());
    assert!(s.try_step(ACTION_MOVE_E).is_some());
    assert!(s.try_step(99).is_none()); // unknown action
}

#[test]
fn test_try_step_none_on_terminal() {
    let s = BomberState::initial(start_on_exit_config());
    assert!(s.try_step(ACTION_WAIT).is_none());
}

#[test]
fn test_hash_deterministic_same_state() {
    let s = BomberState::initial(BomberConfig::default());
    assert_eq!(s.hash(), s.clone().hash());
}

#[test]
fn test_hash_differs_after_step() {
    let s = BomberState::initial(BomberConfig::default());
    let h0 = s.hash();
    let h1 = s.step(ACTION_MOVE_E).hash();
    assert_ne!(h0, h1);
}

#[test]
fn test_hash_differs_after_bomb_placement() {
    let s = BomberState::initial(BomberConfig::default());
    let h0 = s.hash();
    let h1 = s.step(ACTION_PLACE_BOMB).hash();
    assert_ne!(h0, h1);
}

// ── Section 4: move legality ───────────────────────────────────

#[test]
fn test_move_blocked_by_wall() {
    let s = BomberState::initial(corridor_puzzle_config());
    let at_4 = s.step(ACTION_MOVE_E).step(ACTION_MOVE_S); // 0 → 1 → 4
    assert!(!at_4.is_legal(ACTION_MOVE_E), "4 → 5 is a Wall");
    assert!(!at_4.is_legal(ACTION_MOVE_W), "4 → 3 is a Wall");
}

#[test]
fn test_move_blocked_by_block() {
    let s = BomberState::initial(corridor_puzzle_config());
    let at_4 = s.step(ACTION_MOVE_E).step(ACTION_MOVE_S);
    assert!(!at_4.is_legal(ACTION_MOVE_S), "4 → 7 is a Block");
}

#[test]
fn test_move_onto_bomb_illegal() {
    // Place bomb at index 1, step off, then returning to 1 is illegal.
    let s = BomberState::initial(BomberConfig::default());
    let bombed = s.step(ACTION_MOVE_E).step(ACTION_PLACE_BOMB); // 0 → 1, bomb@1
    let at_4 = bombed.step(ACTION_MOVE_S); // 1 → 4
    assert!(
        !at_4.is_legal(ACTION_MOVE_N),
        "cannot move back onto a bomb cell"
    );
}

#[test]
fn test_move_out_of_bounds_illegal() {
    let s = BomberState::initial(BomberConfig::default());
    assert!(!s.is_legal(ACTION_MOVE_N));
    assert!(!s.is_legal(ACTION_MOVE_W));
}

// ── Section 5: bomb mechanics ──────────────────────────────────

#[test]
fn test_place_bomb_creates_bomb_with_configured_fuse() {
    let s = BomberState::initial(BomberConfig::default());
    assert!(s.bombs.is_empty());
    let next = s.step(ACTION_PLACE_BOMB);
    assert_eq!(next.bombs.len(), 1);
    assert_eq!(next.bombs[0].pos, 0);
    // fuse ticks at end of placement turn: 3 → 2.
    assert_eq!(next.bombs[0].fuse, 2);
}

#[test]
fn test_double_place_bomb_same_cell_illegal() {
    let s = BomberState::initial(BomberConfig::default());
    let at_bomb = s.step(ACTION_PLACE_BOMB); // bomb at player cell 0
    assert!(
        !at_bomb.is_legal(ACTION_PLACE_BOMB),
        "second bomb at same cell illegal"
    );
}

#[test]
fn test_bomb_fuse_ticks_each_turn() {
    let s = BomberState::initial(BomberConfig::default());
    let t0 = s.step(ACTION_PLACE_BOMB); // fuse 3 → 2
    assert_eq!(t0.bombs[0].fuse, 2);
    let t1 = t0.step(ACTION_WAIT); // → 1
    assert_eq!(t1.bombs[0].fuse, 1);
}

#[test]
fn test_bomb_detonates_after_fuse_expires() {
    // fuse=3 → detonates on the 3rd step after placement (placement ticks
    // to 2, then two more ticks to 0).
    let s = BomberState::initial(BomberConfig::default());
    let t_place = s.step(ACTION_PLACE_BOMB); // bomb@0, fuse 2
    let t1 = t_place.step(ACTION_MOVE_S); // 0 → 3, fuse 1
    let t2 = t1.step(ACTION_MOVE_S); // 3 → 6, fuse 0 → DETONATE
    assert!(t2.bombs.is_empty(), "bomb must be removed after detonation");
}

#[test]
fn test_retreat_to_escape_blast_survives() {
    let s = BomberState::initial(corridor_puzzle_config());
    let at_4 = s.step(ACTION_MOVE_E).step(ACTION_MOVE_S); // 0 → 1 → 4
    let bombed = at_4.step(ACTION_PLACE_BOMB); // bomb@4, fuse 2
    assert!(bombed.alive);
    // Retreat 4 →(N) 1 →(W) 0; detonation fires at end of the second move.
    let r1 = bombed.step(ACTION_MOVE_N); // 4 → 1, fuse 1
    assert!(r1.alive, "mid-retreat: not yet detonated");
    let r2 = r1.step(ACTION_MOVE_W); // 1 → 0, fuse 0 → detonate {4,1,7}
    assert!(r2.alive, "player at index 0 is outside blast radius");
    assert!(r2.bombs.is_empty(), "bomb removed after detonation");
    assert_eq!(r2.grid[7], Cell::Floor, "block at 7 destroyed");
}

#[test]
fn test_bomb_field_exposes_bomb_struct() {
    // The `Bomb` struct is part of the public API; spot-check construction.
    let b = Bomb { pos: 4, fuse: 2 };
    assert_eq!(b.pos, 4);
    assert_eq!(b.fuse, 2);
}

// ── Section 6: death & victory ─────────────────────────────────

#[test]
fn test_waiting_on_bomb_kills_player() {
    let s = BomberState::initial(corridor_puzzle_config());
    let bombed = s
        .step(ACTION_MOVE_E)
        .step(ACTION_MOVE_S)
        .step(ACTION_PLACE_BOMB);
    let w1 = bombed.step(ACTION_WAIT); // fuse 2 → 1
    assert!(w1.alive);
    let w2 = w1.step(ACTION_WAIT); // fuse 1 → 0 → detonate, player at 4
    assert!(!w2.alive, "player at bomb center dies");
    assert!(w2.is_terminal());
    assert!(!w2.is_goal());
    assert_eq!(w2.reward(), -1.0);
    assert!(
        w2.legal_actions().is_empty(),
        "dead state has no legal actions"
    );
}

#[test]
fn test_corridor_puzzle_full_solution_reaches_exit() {
    let s = BomberState::initial(corridor_puzzle_config());
    // E, S, BOMB, N, W [detonate; block@7 destroyed], E, S, S, S
    let s = s.step(ACTION_MOVE_E); // 0 → 1
    let s = s.step(ACTION_MOVE_S); // 1 → 4
    let s = s.step(ACTION_PLACE_BOMB); // bomb@4, fuse 2
    let s = s.step(ACTION_MOVE_N); // 4 → 1, fuse 1
    let s = s.step(ACTION_MOVE_W); // 1 → 0, fuse 0 → detonate, block@7 destroyed
    assert_eq!(s.grid[7], Cell::Floor);
    assert!(s.alive);
    let s = s.step(ACTION_MOVE_E); // → 1
    let s = s.step(ACTION_MOVE_S); // → 4
    let s = s.step(ACTION_MOVE_S); // → 7 (now Floor)
    assert_eq!(s.player, 7);
    assert!(!s.is_terminal());
    let s = s.step(ACTION_MOVE_S); // → 10 (Exit)
    assert!(s.won);
    assert!(s.is_terminal());
    assert!(s.is_goal());
    assert_eq!(s.reward(), 1.0);
}

// ── Section 7: BomberActionPruner — stateless replay mirror ────

#[test]
fn test_pruner_is_valid_basic() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert!(pruner.is_valid(0, ACTION_MOVE_E, &[]));
    assert!(pruner.is_valid(0, ACTION_PLACE_BOMB, &[]));
    assert!(!pruner.is_valid(0, ACTION_MOVE_N, &[])); // OOB
    assert!(!pruner.is_valid(0, 99, &[])); // unknown
}

#[test]
fn test_pruner_is_valid_mirrors_state_is_legal_over_state_space() {
    // Property: for every reachable (trajectory, state) pair and every
    // action a, pruner.is_valid(traj.len(), a, &traj) == state.is_legal(a).
    let cfg = corridor_puzzle_config();
    let pruner = BomberActionPruner::new(cfg.clone());
    let initial = BomberState::initial(cfg);
    let mut seen: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut frontier: Vec<(Vec<TokenId>, BomberState)> = vec![(vec![], initial)];
    let mut checked = 0usize;

    while let Some((traj, state)) = frontier.pop() {
        if !seen.insert(state.hash()) {
            continue; // already expanded this exact state
        }
        for a in 0..BOMBER_VOCAB as TokenId {
            let p = pruner.is_valid(traj.len(), a, &traj);
            let st = state.is_legal(a);
            assert_eq!(p, st, "mirror mismatch at traj={traj:?} a={a}");
            checked += 1;
        }
        if traj.len() >= 8 || state.is_terminal() {
            continue;
        }
        for a in state.legal_actions() {
            let mut nt = traj.clone();
            nt.push(a);
            frontier.push((nt, state.step(a)));
        }
    }
    assert!(
        checked > 30,
        "must check many (state, action) pairs; got {checked}"
    );
}

#[test]
fn test_pruner_manifold_score_binary() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert_eq!(pruner.manifold_score(0, ACTION_MOVE_E, &[]), 1.0);
    assert_eq!(pruner.manifold_score(0, ACTION_MOVE_N, &[]), 0.0);
}

#[test]
fn test_pruner_screen_illegal_action_is_zero() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert_eq!(pruner.screen(0, ACTION_MOVE_N, &[]), 0.0);
    assert_eq!(pruner.screen(0, 99, &[]), 0.0);
}

#[test]
fn test_pruner_screen_legal_move_toward_exit() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    // From (0,0), exit (2,2): MOVE_E and MOVE_S reduce distance → 0.9.
    assert!((pruner.screen(0, ACTION_MOVE_E, &[]) - 0.9).abs() < 1e-6);
    assert!((pruner.screen(0, ACTION_MOVE_S, &[]) - 0.9).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_wait_is_low() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert!((pruner.screen(0, ACTION_WAIT, &[]) - 0.4).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_place_bomb_no_adjacent_block() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert!((pruner.screen(0, ACTION_PLACE_BOMB, &[]) - 0.5).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_place_bomb_adjacent_to_block() {
    let pruner = BomberActionPruner::new(corridor_puzzle_config());
    // Reach index 4 (E,S); block at 7 is adjacent (South) → 0.85.
    let score = pruner.screen(2, ACTION_PLACE_BOMB, &[ACTION_MOVE_E, ACTION_MOVE_S]);
    assert!((score - 0.85).abs() < 1e-6);
}

#[test]
fn test_pruner_arm_metadata() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert_eq!(pruner.arm_id(), BOMBER_ARM_ID);
    assert_eq!(pruner.arm_label(), BOMBER_LABEL);
    assert_eq!(BOMBER_LABEL, "bomber-action");
}

#[test]
fn test_pruner_batch_is_valid() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    let candidates = [ACTION_MOVE_N, ACTION_MOVE_E, ACTION_MOVE_S, 99];
    let mut results = vec![false; candidates.len()];
    pruner.batch_is_valid(0, &candidates, &[], &mut results);
    assert_eq!(results, vec![false, true, true, false]);
}

#[test]
fn test_pruner_batch_screen() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    let candidates = [ACTION_MOVE_E, ACTION_WAIT, 99];
    let mut results = vec![0.0; candidates.len()];
    pruner.batch_screen(0, &candidates, &[], &mut results);
    assert!((results[0] - 0.9).abs() < 1e-6);
    assert!((results[1] - 0.4).abs() < 1e-6);
    assert!((results[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_pruner_state_of_replays_correctly() {
    let cfg = corridor_puzzle_config();
    let pruner = BomberActionPruner::new(cfg.clone());
    let direct = BomberState::initial(cfg)
        .step(ACTION_MOVE_E)
        .step(ACTION_MOVE_S);
    let replayed = pruner
        .state_of(&[ACTION_MOVE_E, ACTION_MOVE_S])
        .expect("legal trace");
    assert_eq!(replayed.player, direct.player);
    assert_eq!(replayed.player, 4);
}

#[test]
fn test_pruner_state_of_illegal_trace_is_none() {
    let pruner = BomberActionPruner::new(BomberConfig::default());
    assert!(pruner.state_of(&[ACTION_MOVE_N]).is_none());
}

#[test]
#[should_panic(expected = "invalid BomberConfig")]
fn test_pruner_new_panics_on_invalid_config() {
    let bad = BomberConfig {
        width: 0,
        height: 0,
        initial_grid: vec![],
        ..BomberConfig::default()
    };
    let _ = BomberActionPruner::new(bad);
}

// ── Section 8: domain-specific draft model ─────────────────────

/// Draft model ranking actions by resulting Manhattan distance to the exit
/// (closer = higher logit). Toward-exit moves rank highest; PLACE_BOMB carries
/// a small bonus over WAIT so "stuck" states detonate into a terminal rather
/// than idle to `max_tokens`. Distinct logits (action-index ε) → deterministic
/// DFS order under a seed.
///
/// Greedy behavior at the bomb puzzle's retreat junction: the only toward-exit
/// move is blocked (the block being bombed), so greedy picks WAIT (current
/// distance beats any retreat's distance) and dies in the blast — a clean
/// dead-end.
struct TowardExitDraft {
    config: BomberConfig,
    vocab: usize,
}

impl TowardExitDraft {
    fn new(config: BomberConfig) -> Self {
        Self {
            config,
            vocab: BOMBER_VOCAB,
        }
    }

    /// Reconstruct the current state by replaying `context`. Stops at the
    /// first illegal action (defensive — valid trajectories never hit this).
    fn replay(&self, context: &[TokenId]) -> BomberState {
        let mut state = BomberState::initial(self.config.clone());
        for &a in context {
            match state.try_step(a) {
                Some(s) => state = s,
                None => break,
            }
        }
        state
    }
}

impl DraftModel for TowardExitDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, context: &[TokenId]) -> Logits {
        let state = self.replay(context);
        let w = self.config.width;
        let h = self.config.height;
        let exit = self.config.exit_index();
        let mut logits = vec![0.0; self.vocab];
        for a in 0..self.vocab as TokenId {
            let (dist, bomb_bonus) = match exit {
                None => (0, 0.0), // no exit: no positional signal
                Some(e) => match delta(a) {
                    Some((dx, dy)) => {
                        let px = (state.player % w) as isize;
                        let py = (state.player / w) as isize;
                        let nx = px + dx;
                        let ny = py + dy;
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                            (1000, 0.0) // OOB → ranked very low
                        } else {
                            let ni = ny as usize * w + nx as usize;
                            (manhattan(ni, e, w), 0.0)
                        }
                    }
                    None => {
                        // WAIT and PLACE_BOMB keep the player at the current
                        // cell, so both use the current distance. PLACE_BOMB
                        // gets a bonus so "stuck" states detonate into a
                        // death terminal instead of idling forever — idling
                        // branches reach max_tokens and abort the whole
                        // backtracking search (engine semantics).
                        let d = manhattan(state.player, e, w);
                        let bonus = match a == ACTION_PLACE_BOMB {
                            true => 0.5,
                            false => 0.0,
                        };
                        (d, bonus)
                    }
                },
            };
            // Deterministic tie-break (a × ε): logit-distinct candidates are
            // never shuffled by `shuffle_tied_groups`, yielding a stable DFS
            // order (reproducible under a fixed seed) and no wasted wandering
            // from randomized ties.
            logits[a as usize] = -(dist as f32) + bomb_bonus + (a as f32) * 1e-4;
        }
        logits
    }
}

/// Draft model tuned for the engine's **best-first** backtracking search.
///
/// `generate_with_backtrack` explores the HIGHEST-logit candidate first
/// (same preference order as greedy's `valid.first()`). Priorities:
/// - **Moves** get `logit = 1000 - distance` → nearest-to-exit is highest →
///   explored FIRST (productive exploration).
/// - **PLACE_BOMB** ranks next → tried before reversing when stuck.
/// - **Reverse of last move** ranks below PLACE_BOMB → deferred, breaking the
///   A→B→A 2-cycle the symmetric-movement grid would otherwise oscillate
///   forever (the engine has no cycle detection).
/// - **WAIT** ranks lowest of the legal actions → explored LAST (no
///   WAIT-wandering to `max_tokens`).
/// - **OOB** moves rank lowest overall (filtered as invalid anyway).
///
/// Distinct logits (action-index ε) → deterministic DFS order under a seed.
struct BacktrackDraft {
    config: BomberConfig,
    vocab: usize,
}

impl BacktrackDraft {
    fn new(config: BomberConfig) -> Self {
        Self {
            config,
            vocab: BOMBER_VOCAB,
        }
    }

    /// Reconstruct the current state by replaying `context`.
    fn replay(&self, context: &[TokenId]) -> BomberState {
        let mut state = BomberState::initial(self.config.clone());
        for &a in context {
            match state.try_step(a) {
                Some(s) => state = s,
                None => break,
            }
        }
        state
    }
}

impl DraftModel for BacktrackDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, context: &[TokenId]) -> Logits {
        let state = self.replay(context);
        let w = self.config.width;
        let h = self.config.height;
        let exit = self.config.exit_index();

        // Anti-oscillation: the DFS has no cycle detection, so on a
        // symmetric-movement grid it bounces A→B→A until `max_tokens`
        // aborts the whole run. Deferring the reverse of the last move
        // (ranked below PLACE_BOMB) breaks the 2-cycle: when no forward
        // move is available, PLACE_BOMB is tried before reversing.
        let reverse_of_last = match context.last().copied() {
            Some(ACTION_MOVE_N) => Some(ACTION_MOVE_S),
            Some(ACTION_MOVE_S) => Some(ACTION_MOVE_N),
            Some(ACTION_MOVE_E) => Some(ACTION_MOVE_W),
            Some(ACTION_MOVE_W) => Some(ACTION_MOVE_E),
            _ => None,
        };

        let mut logits = vec![0.0; self.vocab];
        for a in 0..self.vocab as TokenId {
            let logit = match a {
                ACTION_WAIT => 0.0,
                ACTION_PLACE_BOMB => 500.0,
                _ if Some(a) == reverse_of_last => 100.0,
                _ => match exit {
                    None => 0.0,
                    Some(e) => match delta(a) {
                        Some((dx, dy)) => {
                            let px = (state.player % w) as isize;
                            let py = (state.player / w) as isize;
                            let nx = px + dx;
                            let ny = py + dy;
                            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                                -1_000.0 // OOB → lowest (invalid anyway)
                            } else {
                                let ni = ny as usize * w + nx as usize;
                                // nearer = higher → nearest explored first
                                1_000.0 - manhattan(ni, e, w) as f32
                            }
                        }
                        None => 0.0,
                    },
                },
            };
            // Deterministic tie-break: distinct logits → no shuffle, stable
            // DFS order under a fixed seed.
            logits[a as usize] = logit + (a as f32) * 1e-4;
        }
        logits
    }
}

// ── Section 9: speculative_generate ────────────────────────────

fn greedy_decode(max_tokens: usize) -> DecodeConfig {
    DecodeConfig {
        backtrack: false,
        max_tokens,
        top_k: BOMBER_VOCAB,
        seed: 42,
        max_attempts: 100_000,
    }
}

fn backtrack_decode(max_tokens: usize, max_attempts: u64) -> DecodeConfig {
    DecodeConfig {
        backtrack: true,
        max_tokens,
        top_k: BOMBER_VOCAB,
        seed: 42,
        max_attempts,
    }
}

#[test]
fn test_generate_greedy_solves_open_grid() {
    let cfg = BomberConfig::default();
    let initial = BomberState::initial(cfg.clone());
    let draft = TowardExitDraft::new(cfg.clone());
    let mut pruner = BomberActionPruner::new(cfg);
    let result = speculative_generate(&initial, &draft, &mut pruner, &greedy_decode(20));
    assert!(result.terminal, "open grid must reach a terminal");
    assert!(result.goal, "greedy must solve the open grid");
    assert!(
        (result.reward - 1.0).abs() < 1e-6,
        "reward must be +1.0 at exit"
    );
    assert!(!result.actions.is_empty());
}

#[test]
fn test_generate_greedy_fails_bomb_puzzle() {
    // Reaching the exit requires placing a bomb then retreating outside the
    // blast (temporarily increasing exit distance). A "toward exit" greedy
    // can't retreat: it freezes on the bomb (dies) or loops (budgets out).
    let cfg = corridor_puzzle_config();
    let initial = BomberState::initial(cfg.clone());
    let draft = TowardExitDraft::new(cfg.clone());
    let mut pruner = BomberActionPruner::new(cfg);
    let result = speculative_generate(&initial, &draft, &mut pruner, &greedy_decode(20));
    assert!(!result.goal, "greedy must fail the bomb puzzle");
}

#[test]
fn test_generate_backtrack_solves_bomb_puzzle() {
    let cfg = corridor_puzzle_config();
    let initial = BomberState::initial(cfg.clone());
    let draft = BacktrackDraft::new(cfg.clone());
    let mut pruner = BomberActionPruner::new(cfg);
    let result = speculative_generate(
        &initial,
        &draft,
        &mut pruner,
        &backtrack_decode(15, 100_000),
    );
    assert!(result.terminal, "backtrack must reach a terminal");
    assert!(result.goal, "backtrack must solve the bomb puzzle");
    assert!(
        (result.reward - 1.0).abs() < 1e-6,
        "reward must be +1.0 at exit"
    );

    // Re-verify the trajectory end-to-end: every action legal, reaches exit.
    let mut verify = BomberState::initial(corridor_puzzle_config());
    for &a in &result.actions {
        assert!(verify.is_legal(a), "trajectory action {a} must be legal");
        verify = verify.step(a);
    }
    assert!(verify.is_goal(), "replayed trajectory must reach the exit");
    // The solution must include a bomb placement (the block was destroyed).
    assert!(
        result.actions.contains(&ACTION_PLACE_BOMB),
        "winning trajectory must place a bomb to clear the block"
    );
}

#[test]
fn test_generate_backtrack_deterministic_under_seed() {
    // Same seed → same trajectory hash (the shuffle is seeded).
    let cfg = corridor_puzzle_config();
    let mk = || {
        let initial = BomberState::initial(cfg.clone());
        let draft = BacktrackDraft::new(cfg.clone());
        let mut pruner = BomberActionPruner::new(cfg.clone());
        speculative_generate(
            &initial,
            &draft,
            &mut pruner,
            &backtrack_decode(15, 100_000),
        )
    };
    let r1 = mk();
    let r2 = mk();
    assert!(r1.goal && r2.goal);
    assert_eq!(r1.actions, r2.actions, "same seed → same action trajectory");
    assert_eq!(r1.hash, r2.hash);
}

// ── Section 10: SpeculativeGenerator bundle ────────────────────

/// Bundles state factory + draft + pruner + decode config behind the
/// `SpeculativeGenerator` contract, exercising the split-borrow `parts()`
/// method and the default `generate()`.
///
/// Generic over the draft model so the same wiring serves both greedy
/// (`TowardExitDraft`) and backtracking (`BacktrackDraft`) runs — the two
/// modes explore candidates in opposite logit order (best-first vs
/// worst-first), so each needs its own draft.
struct BomberGenerator<D: DraftModel> {
    config: BomberConfig,
    draft: D,
    pruner: BomberActionPruner,
    decode: DecodeConfig,
}

impl<D: DraftModel> BomberGenerator<D> {
    fn new(config: BomberConfig, draft: D, decode: DecodeConfig) -> Self {
        let pruner = BomberActionPruner::new(config.clone());
        Self {
            config,
            draft,
            pruner,
            decode,
        }
    }
}

impl<D: DraftModel> SpeculativeGenerator for BomberGenerator<D> {
    type State = BomberState;
    type Pruner = BomberActionPruner;

    fn initial_state(&self) -> Self::State {
        BomberState::initial(self.config.clone())
    }

    fn parts(&mut self) -> (&dyn DraftModel, &mut Self::Pruner, &DecodeConfig) {
        (&self.draft, &mut self.pruner, &self.decode)
    }
}

#[test]
fn test_generator_default_generate_open_grid() {
    let cfg = BomberConfig::default();
    let draft = TowardExitDraft::new(cfg.clone());
    let mut gen = BomberGenerator::new(cfg, draft, greedy_decode(20));
    let result = gen.generate();
    assert!(result.goal);
    assert!(result.terminal);
}

#[test]
fn test_generator_backtrack_solves_bomb_puzzle() {
    let cfg = corridor_puzzle_config();
    let draft = BacktrackDraft::new(cfg.clone());
    let mut gen = BomberGenerator::new(cfg, draft, backtrack_decode(15, 100_000));
    let result = gen.generate();
    assert!(result.goal);
    assert!(result.terminal);
    assert!((result.reward - 1.0).abs() < 1e-6);
}

#[test]
fn test_generator_initial_state_matches_canonical_origin() {
    // The pruner's canonical origin must equal the generator's initial
    // state — the stateless-replay invariant.
    let cfg = corridor_puzzle_config();
    let draft = TowardExitDraft::new(cfg.clone());
    let gen = BomberGenerator::new(cfg, draft, greedy_decode(5));
    let initial = gen.initial_state();
    let replayed = gen
        .pruner
        .state_of(&[])
        .expect("empty trace → initial state");
    assert_eq!(initial, replayed);
}
