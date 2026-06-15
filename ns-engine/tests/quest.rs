//! Integration tests for the QuestActionPruner + QuestState domain.
//!
//! Phase 5 domain-generalization deliverable. Exercises the full public API
//! of `QuestState` (forward model) and `QuestActionPruner` (stateless replay
//! legality mirror) from outside the crate. Covers:
//!
//! - config validation (all error variants + `goal_path_mask` + `vocab`)
//! - `GameState` trait surface (purity, legality, terminals, rewards, hash)
//! - quest mechanics (accept/complete/fail, prerequisite gating, fail blocks)
//! - terminal conditions (goal victory, dead-end loss, deadlock)
//! - pruner legality mirror (property test over the reachable state space)
//! - graded `ScreeningPruner` scores (goal-path vs off-path, fail low)
//! - `speculative_generate`: greedy success, fail-first dead-end, backtrack
//!   recovery, deterministic-under-seed, `SpeculativeGenerator` bundle

use ns_engine::pruners::{
    decode_quest_action, encode_quest_action, QuestActionPruner, QuestConfig, QuestConfigError,
    QuestState, QuestStatus, ACTIONS_PER_QUEST, QUEST_ACTION_ACCEPT, QUEST_ACTION_COMPLETE,
    QUEST_ACTION_FAIL, QUEST_ARM_ID, QUEST_LABEL,
};
use ns_engine::traits::{ConstraintPruner, DraftModel, ScreeningPruner};
use ns_engine::{
    speculative_generate, DecodeConfig, GameState, Logits, SpeculativeGenerator, TokenId,
};

// ── test fixtures ──────────────────────────────────────────────

/// The default 3-quest linear chain: Q0 → Q1 → Q2, goal = [Q2].
///
/// Solution: accept0, complete0, accept1, complete1, accept2, complete2
/// (6 actions, reaches goal).
fn default_config() -> QuestConfig {
    QuestConfig::default()
}

/// The diamond puzzle: Q0 → {Q1, Q2} → Q3, goal = [Q3].
///
/// Q3 requires BOTH Q1 and Q2 completed; both require Q0. Branching
/// prerequisite DAG. Solution: accept0, complete0, accept1, complete1,
/// accept2, complete2, accept3, complete3 (8 actions).
fn diamond_config() -> QuestConfig {
    QuestConfig {
        num_quests: 4,
        prerequisites: vec![vec![], vec![0], vec![0], vec![1, 2]],
        goal_quests: vec![3],
    }
}

/// Distraction config: Q0 off-path distractor; Q1 → Q2 goal path, goal = [Q2].
///
/// Q0 has no prereqs and no dependents — completing it is pure distraction.
/// Used to exercise graded screen scoring (off-path quest = 0.6 vs goal-path
/// prereq = 0.85).
fn distraction_config() -> QuestConfig {
    QuestConfig {
        num_quests: 3,
        prerequisites: vec![vec![], vec![], vec![1]],
        goal_quests: vec![2],
    }
}

/// Convenience: ACCEPT(q) token.
fn accept(q: usize) -> TokenId {
    encode_quest_action(q, QUEST_ACTION_ACCEPT)
}

/// Convenience: COMPLETE(q) token.
fn complete(q: usize) -> TokenId {
    encode_quest_action(q, QUEST_ACTION_COMPLETE)
}

/// Convenience: FAIL(q) token.
fn fail(q: usize) -> TokenId {
    encode_quest_action(q, QUEST_ACTION_FAIL)
}

// ── Section 1: config validation ───────────────────────────────

#[test]
fn test_config_default_validates() {
    assert!(QuestConfig::default().validate().is_ok());
}

#[test]
fn test_config_diamond_validates() {
    assert!(diamond_config().validate().is_ok());
}

#[test]
fn test_config_distraction_validates() {
    assert!(distraction_config().validate().is_ok());
}

#[test]
fn test_config_error_zero_quests() {
    let cfg = QuestConfig {
        num_quests: 0,
        prerequisites: vec![],
        goal_quests: vec![],
    };
    assert_eq!(cfg.validate(), Err(QuestConfigError::ZeroQuests));
}

#[test]
fn test_config_error_prerequisite_count_mismatch() {
    let cfg = QuestConfig {
        num_quests: 3,
        prerequisites: vec![vec![], vec![0]], // should be 3 entries
        goal_quests: vec![2],
    };
    assert_eq!(
        cfg.validate(),
        Err(QuestConfigError::PrerequisiteCountMismatch)
    );
}

#[test]
fn test_config_error_prerequisite_out_of_bounds() {
    let cfg = QuestConfig {
        num_quests: 2,
        prerequisites: vec![vec![], vec![5]], // prereq 5 >= num_quests
        goal_quests: vec![1],
    };
    assert_eq!(
        cfg.validate(),
        Err(QuestConfigError::PrerequisiteOutOfBounds)
    );
}

#[test]
fn test_config_error_self_prerequisite() {
    let cfg = QuestConfig {
        num_quests: 2,
        prerequisites: vec![vec![], vec![1]], // Q1 requires itself
        goal_quests: vec![1],
    };
    assert_eq!(cfg.validate(), Err(QuestConfigError::SelfPrerequisite));
}

#[test]
fn test_config_error_goal_quest_out_of_bounds() {
    let cfg = QuestConfig {
        num_quests: 2,
        prerequisites: vec![vec![], vec![0]],
        goal_quests: vec![9], // goal 9 >= num_quests
    };
    assert_eq!(cfg.validate(), Err(QuestConfigError::GoalQuestOutOfBounds));
}

#[test]
fn test_config_allows_empty_goal_quests() {
    // Empty goal_quests is structurally valid (puzzle just never wins).
    let cfg = QuestConfig {
        num_quests: 1,
        prerequisites: vec![vec![]],
        goal_quests: vec![],
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_vocab_scales_with_num_quests() {
    assert_eq!(default_config().vocab(), 9); // 3 × 3
    assert_eq!(diamond_config().vocab(), 12); // 3 × 4
    assert_eq!(distraction_config().vocab(), 9);
}

#[test]
fn test_config_goal_path_mask_linear() {
    assert_eq!(default_config().goal_path_mask(), vec![true, true, true]);
}

#[test]
fn test_config_goal_path_mask_diamond() {
    assert_eq!(
        diamond_config().goal_path_mask(),
        vec![true, true, true, true]
    );
}

#[test]
fn test_config_goal_path_mask_excludes_distraction() {
    assert_eq!(
        distraction_config().goal_path_mask(),
        vec![false, true, true]
    );
}

// ── Section 2: GameState trait surface ─────────────────────────

#[test]
fn test_initial_state_all_not_started() {
    let s = QuestState::initial(diamond_config());
    assert_eq!(s.status, vec![QuestStatus::NotStarted; 4]);
    assert!(s.last_action.is_none());
}

#[test]
fn test_initial_is_non_terminal_non_goal() {
    let s = QuestState::initial(default_config());
    assert!(!s.is_terminal(), "Q0 is Available at start → not terminal");
    assert!(!s.is_goal());
    assert!((s.reward() - 0.0).abs() < 1e-6);
}

#[test]
fn test_step_is_pure_snapshot() {
    let s = QuestState::initial(default_config());
    let _next = s.step(accept(0));
    // Receiver unchanged: Q0 still NotStarted.
    assert_eq!(s.status[0], QuestStatus::NotStarted);
    assert!(s.last_action.is_none());
}

#[test]
fn test_try_step_illegal_returns_none() {
    let s = QuestState::initial(default_config());
    // ACCEPT(1) illegal: Q1 locked (Q0 not completed).
    assert!(s.try_step(accept(1)).is_none());
    // COMPLETE(0) illegal: Q0 NotStarted, not Active.
    assert!(s.try_step(complete(0)).is_none());
    // Out-of-range quest.
    assert!(s.try_step(accept(99)).is_none());
}

#[test]
fn test_try_step_legal_returns_successor() {
    let s = QuestState::initial(default_config());
    let next = s.try_step(accept(0)).expect("ACCEPT(0) legal");
    assert_eq!(next.status[0], QuestStatus::Active);
    assert_eq!(next.last_action, Some(accept(0)));
}

#[test]
fn test_is_legal_false_on_terminal() {
    // Reach goal → terminal → every action illegal.
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0))
        .step(accept(1))
        .step(complete(1))
        .step(accept(2))
        .step(complete(2));
    assert!(s.is_goal());
    assert!(!s.is_legal(accept(0)));
    assert!(s.legal_actions().is_empty());
}

#[test]
fn test_legal_actions_initial_state() {
    let s = QuestState::initial(diamond_config());
    // Only Q0 available at start → ACCEPT(0) is the sole legal action.
    assert_eq!(s.legal_actions(), vec![accept(0)]);
}

#[test]
fn test_legal_actions_active_quest() {
    let s = QuestState::initial(default_config()).step(accept(0));
    // Q0 Active → COMPLETE(0) and FAIL(0) legal (Q1 still locked).
    let mut legal = s.legal_actions();
    legal.sort();
    assert_eq!(legal, vec![complete(0), fail(0)]);
}

#[test]
fn test_action_mask() {
    let s = QuestState::initial(default_config());
    let mask = s.action_mask(default_config().vocab());
    // Only ACCEPT(0) = token 0 is legal.
    assert!(mask[accept(0) as usize]);
    assert_eq!(mask.iter().filter(|&&m| m).count(), 1);
}

#[test]
fn test_hash_deterministic_same_status() {
    let s1 = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0));
    let s2 = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0));
    assert_eq!(s1.hash(), s2.hash());
}

#[test]
fn test_hash_differs_for_different_status() {
    let s1 = QuestState::initial(default_config());
    let s2 = QuestState::initial(default_config()).step(accept(0));
    assert_ne!(s1.hash(), s2.hash());
}

// ── Section 3: quest mechanics ─────────────────────────────────

#[test]
fn test_accept_requires_available() {
    let s = QuestState::initial(default_config());
    assert!(s.is_legal(accept(0))); // Q0 available (no prereqs)
    assert!(!s.is_legal(accept(1))); // Q1 locked (needs Q0)
    assert!(!s.is_legal(accept(2))); // Q2 locked (needs Q1)
}

#[test]
fn test_accept_advances_to_active() {
    let s = QuestState::initial(default_config()).step(accept(0));
    assert_eq!(s.status[0], QuestStatus::Active);
    assert_eq!(s.last_action, Some(accept(0)));
}

#[test]
fn test_complete_requires_active() {
    let s = QuestState::initial(default_config());
    // Q0 NotStarted → COMPLETE illegal.
    assert!(!s.is_legal(complete(0)));
    let s = s.step(accept(0));
    // Now Q0 Active → COMPLETE legal.
    assert!(s.is_legal(complete(0)));
}

#[test]
fn test_complete_advances_to_completed() {
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0));
    assert_eq!(s.status[0], QuestStatus::Completed);
}

#[test]
fn test_fail_requires_active() {
    let s = QuestState::initial(default_config()).step(accept(0));
    assert!(s.is_legal(fail(0)));
    let s = s.step(fail(0));
    assert_eq!(s.status[0], QuestStatus::Failed);
}

#[test]
fn test_prerequisite_unlocks_after_completion() {
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0));
    // Q0 completed → Q1 now available.
    assert!(s.is_legal(accept(1)));
}

#[test]
fn test_fail_does_not_unlock_dependents() {
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(fail(0));
    // Q0 Failed → Q1 still locked (prereqs require Completed).
    assert!(!s.is_legal(accept(1)));
}

#[test]
fn test_double_accept_illegal() {
    let s = QuestState::initial(default_config()).step(accept(0));
    // Q0 already Active → ACCEPT(0) illegal.
    assert!(!s.is_legal(accept(0)));
}

#[test]
fn test_complete_already_completed_illegal() {
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0));
    assert!(!s.is_legal(complete(0)));
}

#[test]
fn test_diamond_both_prereqs_required() {
    // Q3 requires Q1 AND Q2. Complete only Q1 → Q3 still locked.
    let s = QuestState::initial(diamond_config())
        .step(accept(0))
        .step(complete(0))
        .step(accept(1))
        .step(complete(1));
    assert!(!s.is_legal(accept(3)), "Q3 locked with only Q1 done");
    // Now complete Q2 → Q3 unlocks.
    let s = s.step(accept(2)).step(complete(2));
    assert!(s.is_legal(accept(3)));
}

// ── Section 4: terminal conditions ─────────────────────────────

#[test]
fn test_goal_reached_is_terminal_with_positive_reward() {
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(complete(0))
        .step(accept(1))
        .step(complete(1))
        .step(accept(2))
        .step(complete(2));
    assert!(s.is_goal());
    assert!(s.is_terminal());
    assert!((s.reward() - 1.0).abs() < 1e-6);
    assert!(s.legal_actions().is_empty());
}

#[test]
fn test_fail_goal_path_quest_is_loss_terminal() {
    // Failing Q0 in the default chain makes Q2 unreachable → dead-end.
    let s = QuestState::initial(default_config())
        .step(accept(0))
        .step(fail(0));
    assert!(!s.is_goal());
    assert!(s.is_terminal(), "no quest can progress after Q0 failed");
    assert!((s.reward() - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_deadlock_when_all_active_quests_failed() {
    // Diamond: accept+fail Q1, accept+fail Q2 → Q3 can never be unlocked.
    let s = QuestState::initial(diamond_config())
        .step(accept(0))
        .step(complete(0))
        .step(accept(1))
        .step(fail(1))
        .step(accept(2))
        .step(fail(2));
    assert!(!s.is_goal());
    assert!(s.is_terminal());
    assert!((s.reward() - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_empty_goal_quests_never_goal_but_can_be_terminal() {
    let cfg = QuestConfig {
        num_quests: 1,
        prerequisites: vec![vec![]],
        goal_quests: vec![],
    };
    let s = QuestState::initial(cfg);
    assert!(!s.is_goal(), "empty goal_quests → never goal");
    assert!(!s.is_terminal(), "Q0 Available → not terminal yet");
    let s = s.step(accept(0)).step(complete(0));
    assert!(!s.is_goal());
    assert!(s.is_terminal(), "all quests done, no progress possible");
    // Non-goal terminal → negative reward.
    assert!((s.reward() - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_goal_quest_completion_freezes_state() {
    // is_goal is reached the instant all goal quests complete, even if other
    // (non-goal) quests are still NotStarted. The state is terminal (frozen):
    // further actions are rejected by is_legal / try_step without panicking.
    let cfg = QuestConfig {
        num_quests: 2,
        prerequisites: vec![vec![], vec![]],
        goal_quests: vec![0],
    };
    let s = QuestState::initial(cfg).step(accept(0)).step(complete(0)); // goal reached — state frozen here
    assert!(s.is_goal());
    assert!(s.is_terminal());
    // Q1 is still NotStarted (and Available — no prereqs), but the frozen
    // goal-terminal state rejects all further actions.
    assert_eq!(s.status[1], QuestStatus::NotStarted);
    assert!(
        !s.is_legal(accept(1)),
        "frozen goal state rejects new actions"
    );
    assert!(s.try_step(accept(1)).is_none(), "try_step None on terminal");
    assert!(s.legal_actions().is_empty());
}

// ── Section 5: QuestActionPruner — stateless replay mirror ─────

#[test]
fn test_pruner_is_valid_basic() {
    let pruner = QuestActionPruner::new(default_config());
    assert!(pruner.is_valid(0, accept(0), &[]));
    assert!(!pruner.is_valid(0, accept(1), &[])); // Q1 locked
    assert!(!pruner.is_valid(0, complete(0), &[])); // Q0 not Active
    assert!(!pruner.is_valid(0, 999, &[])); // out of range
}

#[test]
fn test_pruner_is_valid_mirrors_state_is_legal_over_state_space() {
    // Property: for every reachable (trajectory, state) pair and every action
    // a in [0, vocab), pruner.is_valid(traj.len(), a, &traj) == state.is_legal(a).
    let cfg = diamond_config();
    let pruner = QuestActionPruner::new(cfg.clone());
    let initial = QuestState::initial(cfg.clone());
    let vocab = cfg.vocab();
    let mut seen: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut frontier: Vec<(Vec<TokenId>, QuestState)> = vec![(vec![], initial)];
    let mut checked = 0usize;

    while let Some((traj, state)) = frontier.pop() {
        if !seen.insert(state.hash()) {
            continue; // transposition: same status vector already expanded
        }
        for a in 0..vocab as TokenId {
            let p = pruner.is_valid(traj.len(), a, &traj);
            let st = state.is_legal(a);
            assert_eq!(p, st, "mirror mismatch at traj={traj:?} a={a}");
            checked += 1;
        }
        if traj.len() >= 10 || state.is_terminal() {
            continue;
        }
        for a in state.legal_actions() {
            let mut nt = traj.clone();
            nt.push(a);
            frontier.push((nt, state.step(a)));
        }
    }
    assert!(
        checked > 40,
        "must check many (state, action) pairs; got {checked}"
    );
}

#[test]
fn test_pruner_manifold_score_binary() {
    let pruner = QuestActionPruner::new(default_config());
    assert_eq!(pruner.manifold_score(0, accept(0), &[]), 1.0);
    assert_eq!(pruner.manifold_score(0, accept(1), &[]), 0.0);
}

#[test]
fn test_pruner_batch_is_valid() {
    let pruner = QuestActionPruner::new(default_config());
    let candidates = [accept(0), accept(1), complete(0), 999];
    let mut results = vec![false; candidates.len()];
    pruner.batch_is_valid(0, &candidates, &[], &mut results);
    assert_eq!(results, vec![true, false, false, false]);
}

#[test]
fn test_pruner_state_of_replays_correctly() {
    let cfg = diamond_config();
    let pruner = QuestActionPruner::new(cfg.clone());
    let direct = QuestState::initial(cfg)
        .step(accept(0))
        .step(complete(0))
        .step(accept(1));
    let replayed = pruner
        .state_of(&[accept(0), complete(0), accept(1)])
        .expect("legal trace");
    assert_eq!(replayed.status, direct.status);
    assert_eq!(replayed.status[1], QuestStatus::Active);
}

#[test]
fn test_pruner_state_of_illegal_trace_is_none() {
    let pruner = QuestActionPruner::new(default_config());
    // ACCEPT(1) illegal at start (Q1 locked).
    assert!(pruner.state_of(&[accept(1)]).is_none());
}

#[test]
fn test_pruner_state_of_empty_is_initial() {
    let pruner = QuestActionPruner::new(default_config());
    let s = pruner.state_of(&[]).expect("empty trace → initial");
    assert_eq!(s.status, vec![QuestStatus::NotStarted; 3]);
}

#[test]
#[should_panic(expected = "invalid QuestConfig")]
fn test_pruner_new_panics_on_invalid_config() {
    let bad = QuestConfig {
        num_quests: 0,
        prerequisites: vec![],
        goal_quests: vec![],
    };
    let _ = QuestActionPruner::new(bad);
}

#[test]
fn test_decode_quest_action_round_trip() {
    for q in 0..5_usize {
        for kind in 0..ACTIONS_PER_QUEST as TokenId {
            let token = encode_quest_action(q, kind);
            let (dq, dk) = decode_quest_action(token);
            assert_eq!(dq, q);
            assert_eq!(dk, kind);
        }
    }
}

// ── Section 6: ScreeningPruner graded scores ───────────────────

#[test]
fn test_pruner_screen_illegal_action_is_zero() {
    let pruner = QuestActionPruner::new(default_config());
    assert_eq!(pruner.screen(0, accept(1), &[]), 0.0); // locked
    assert_eq!(pruner.screen(0, 999, &[]), 0.0); // out of range
}

#[test]
fn test_pruner_screen_complete_goal_quest() {
    // Default chain: make Q2 (goal quest) Active, score COMPLETE.
    let pruner = QuestActionPruner::new(default_config());
    let score = pruner.screen(
        1,
        complete(2),
        &[accept(0), complete(0), accept(1), complete(1), accept(2)],
    );
    assert!((score - 0.95).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_complete_goal_path_prereq() {
    // Q0 on goal path but not a goal quest → COMPLETE = 0.85.
    let pruner = QuestActionPruner::new(default_config());
    let score = pruner.screen(1, complete(0), &[accept(0)]);
    assert!((score - 0.85).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_complete_off_path_quest() {
    let pruner = QuestActionPruner::new(distraction_config());
    // Q0 off-path → COMPLETE = 0.6.
    let score = pruner.screen(1, complete(0), &[accept(0)]);
    assert!((score - 0.6).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_accept_goal_path() {
    let pruner = QuestActionPruner::new(default_config());
    // Q0 on goal path → ACCEPT = 0.75.
    assert!((pruner.screen(0, accept(0), &[]) - 0.75).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_accept_off_path() {
    let pruner = QuestActionPruner::new(distraction_config());
    // Q0 off-path → ACCEPT = 0.55.
    assert!((pruner.screen(0, accept(0), &[]) - 0.55).abs() < 1e-6);
}

#[test]
fn test_pruner_screen_fail_is_low() {
    let pruner = QuestActionPruner::new(default_config());
    let score = pruner.screen(1, fail(0), &[accept(0)]);
    assert!((score - 0.2).abs() < 1e-6);
}

#[test]
fn test_pruner_batch_screen() {
    let pruner = QuestActionPruner::new(default_config());
    let candidates = [accept(0), fail(0), accept(1)];
    let mut results = vec![0.0; candidates.len()];
    pruner.batch_screen(0, &candidates, &[], &mut results);
    // accept(0) legal (0.75); fail(0) illegal at NotStarted (0.0); accept(1) illegal (0.0).
    assert!((results[0] - 0.75).abs() < 1e-6);
    assert!((results[1] - 0.0).abs() < 1e-6);
    assert!((results[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_pruner_arm_metadata() {
    let pruner = QuestActionPruner::new(default_config());
    assert_eq!(pruner.arm_id(), QUEST_ARM_ID);
    assert_eq!(pruner.arm_label(), QUEST_LABEL);
    assert_eq!(QUEST_LABEL, "quest-action");
}

#[test]
fn test_pruner_config_accessor() {
    let pruner = QuestActionPruner::new(default_config());
    assert_eq!(pruner.config().num_quests, 3);
    assert_eq!(pruner.config().goal_quests, vec![2]);
}

// ── Section 7: domain-specific draft models ────────────────────

/// Draft model ranking goal-path completion highest (best-first greedy).
/// COMPLETE goal quest > COMPLETE goal-path prereq > ACCEPT goal-path >
/// ACCEPT off-path > COMPLETE off-path > FAIL (lowest). Distinct logits
/// (action-index ε) → deterministic DFS order under a seed.
struct TowardGoalDraft {
    config: QuestConfig,
    vocab: usize,
}

impl TowardGoalDraft {
    fn new(config: QuestConfig) -> Self {
        let vocab = config.vocab();
        Self { config, vocab }
    }
}

impl DraftModel for TowardGoalDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        let goal_path = self.config.goal_path_mask();
        let mut logits = vec![0.0; self.vocab];
        for (q, &gp) in goal_path.iter().enumerate() {
            for kind in 0..ACTIONS_PER_QUEST as TokenId {
                let a = encode_quest_action(q, kind);
                let is_goal_q = self.config.goal_quests.contains(&q);
                let logit = match kind {
                    QUEST_ACTION_COMPLETE => {
                        if is_goal_q {
                            10.0
                        } else if gp {
                            7.0
                        } else {
                            3.0
                        }
                    }
                    QUEST_ACTION_ACCEPT => {
                        if gp {
                            5.0
                        } else {
                            1.0
                        }
                    }
                    QUEST_ACTION_FAIL => -10.0,
                    _ => 0.0,
                };
                logits[a as usize] = logit + (a as f32) * 1e-4;
            }
        }
        logits
    }
}

/// Draft model tuned for the engine's **worst-first** backtracking search.
/// `generate_with_backtrack` pops the LOWEST-logit candidate first. Goal-path
/// actions get the lowest logits (popped first → productive exploration);
/// FAIL gets the highest (popped last → deferred, immediately backtracked
/// from its loss terminal). Distinct logits (action-index ε) → deterministic
/// DFS order under a seed.
struct BacktrackDraft {
    config: QuestConfig,
    vocab: usize,
}

impl BacktrackDraft {
    fn new(config: QuestConfig) -> Self {
        let vocab = config.vocab();
        Self { config, vocab }
    }
}

impl DraftModel for BacktrackDraft {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn log_probs(&self, _context: &[TokenId]) -> Logits {
        let goal_path = self.config.goal_path_mask();
        let mut logits = vec![0.0; self.vocab];
        for (q, &gp) in goal_path.iter().enumerate() {
            for kind in 0..ACTIONS_PER_QUEST as TokenId {
                let a = encode_quest_action(q, kind);
                let is_goal_q = self.config.goal_quests.contains(&q);
                let logit = match kind {
                    QUEST_ACTION_COMPLETE => {
                        if is_goal_q {
                            -10.0
                        } else if gp {
                            -7.0
                        } else {
                            3.0
                        }
                    }
                    QUEST_ACTION_ACCEPT => {
                        if gp {
                            -5.0
                        } else {
                            1.0
                        }
                    }
                    QUEST_ACTION_FAIL => 10.0,
                    _ => 0.0,
                };
                logits[a as usize] = logit + (a as f32) * 1e-4;
            }
        }
        logits
    }
}

// ── Section 8: speculative_generate ────────────────────────────

fn greedy_decode(max_tokens: usize) -> DecodeConfig {
    DecodeConfig {
        backtrack: false,
        max_tokens,
        top_k: 64,
        seed: 42,
        max_attempts: 100_000,
    }
}

fn backtrack_decode(max_tokens: usize, max_attempts: u64) -> DecodeConfig {
    DecodeConfig {
        backtrack: true,
        max_tokens,
        top_k: 64,
        seed: 42,
        max_attempts,
    }
}

#[test]
fn test_generate_greedy_solves_default_chain() {
    let cfg = default_config();
    let initial = QuestState::initial(cfg.clone());
    let draft = TowardGoalDraft::new(cfg.clone());
    let mut pruner = QuestActionPruner::new(cfg);
    let result = speculative_generate(&initial, &draft, &mut pruner, &greedy_decode(20));
    assert!(result.terminal, "must reach a terminal");
    assert!(result.goal, "greedy must solve the linear chain");
    assert!(
        (result.reward - 1.0).abs() < 1e-6,
        "reward must be +1.0 at goal"
    );
    assert!(!result.actions.is_empty());
}

#[test]
fn test_generate_greedy_solves_diamond() {
    let cfg = diamond_config();
    let initial = QuestState::initial(cfg.clone());
    let draft = TowardGoalDraft::new(cfg.clone());
    let mut pruner = QuestActionPruner::new(cfg);
    let result = speculative_generate(&initial, &draft, &mut pruner, &greedy_decode(20));
    assert!(result.terminal);
    assert!(result.goal, "greedy must solve the diamond");
    assert!((result.reward - 1.0).abs() < 1e-6);
}

#[test]
fn test_generate_greedy_fails_with_fail_first_draft() {
    // BacktrackDraft ranks FAIL highest. Greedy (best-first) picks FAIL first
    // → fails the goal-path quest → dead-end loss terminal.
    let cfg = default_config();
    let initial = QuestState::initial(cfg.clone());
    let draft = BacktrackDraft::new(cfg.clone());
    let mut pruner = QuestActionPruner::new(cfg);
    let result = speculative_generate(&initial, &draft, &mut pruner, &greedy_decode(20));
    assert!(
        !result.goal,
        "greedy + fail-first draft must not reach goal"
    );
    assert!(result.terminal, "must still reach a terminal (loss)");
}

#[test]
fn test_generate_backtrack_solves_diamond() {
    let cfg = diamond_config();
    let initial = QuestState::initial(cfg.clone());
    let draft = BacktrackDraft::new(cfg.clone());
    let mut pruner = QuestActionPruner::new(cfg);
    let result = speculative_generate(
        &initial,
        &draft,
        &mut pruner,
        &backtrack_decode(20, 100_000),
    );
    assert!(result.terminal, "backtrack must reach a terminal");
    assert!(result.goal, "backtrack must solve the diamond");
    assert!((result.reward - 1.0).abs() < 1e-6);

    // Re-verify the trajectory end-to-end: every action legal, reaches goal.
    let mut verify = QuestState::initial(diamond_config());
    for &a in &result.actions {
        assert!(verify.is_legal(a), "trajectory action {a} must be legal");
        verify = verify.step(a);
    }
    assert!(verify.is_goal(), "replayed trajectory must reach the goal");
}

#[test]
fn test_generate_backtrack_deterministic_under_seed() {
    let cfg = diamond_config();
    let mk = || {
        let initial = QuestState::initial(cfg.clone());
        let draft = BacktrackDraft::new(cfg.clone());
        let mut pruner = QuestActionPruner::new(cfg.clone());
        speculative_generate(
            &initial,
            &draft,
            &mut pruner,
            &backtrack_decode(20, 100_000),
        )
    };
    let r1 = mk();
    let r2 = mk();
    assert!(r1.goal && r2.goal);
    assert_eq!(r1.actions, r2.actions, "same seed → same action trajectory");
    assert_eq!(r1.hash, r2.hash);
}

#[test]
fn test_generate_reaches_loss_terminal_when_goal_unreachable() {
    // Construct a forced loss: use a draft that always fails the goal-path
    // quest first (BacktrackDraft under greedy). The terminal reached is a
    // non-goal dead-end with reward -1.0.
    let cfg = default_config();
    let initial = QuestState::initial(cfg.clone());
    let draft = BacktrackDraft::new(cfg.clone());
    let mut pruner = QuestActionPruner::new(cfg);
    let result = speculative_generate(&initial, &draft, &mut pruner, &greedy_decode(20));
    assert!(result.terminal);
    assert!(!result.goal);
    assert!(
        (result.reward - (-1.0)).abs() < 1e-6,
        "non-goal terminal must have reward -1.0"
    );
}

// ── Section 9: SpeculativeGenerator bundle ─────────────────────

/// Bundles state factory + draft + pruner + decode config behind the
/// `SpeculativeGenerator` contract, exercising the split-borrow `parts()`
/// method and the default `generate()`. Generic over the draft model so the
/// same wiring serves both greedy and backtracking runs.
struct QuestGenerator<D: DraftModel> {
    config: QuestConfig,
    draft: D,
    pruner: QuestActionPruner,
    decode: DecodeConfig,
}

impl<D: DraftModel> QuestGenerator<D> {
    fn new(config: QuestConfig, draft: D, decode: DecodeConfig) -> Self {
        let pruner = QuestActionPruner::new(config.clone());
        Self {
            config,
            draft,
            pruner,
            decode,
        }
    }
}

impl<D: DraftModel> SpeculativeGenerator for QuestGenerator<D> {
    type State = QuestState;
    type Pruner = QuestActionPruner;

    fn initial_state(&self) -> Self::State {
        QuestState::initial(self.config.clone())
    }

    fn parts(&mut self) -> (&dyn DraftModel, &mut Self::Pruner, &DecodeConfig) {
        (&self.draft, &mut self.pruner, &self.decode)
    }
}

#[test]
fn test_generator_default_generate_solves_chain() {
    let cfg = default_config();
    let draft = TowardGoalDraft::new(cfg.clone());
    let mut gen = QuestGenerator::new(cfg, draft, greedy_decode(20));
    let result = gen.generate();
    assert!(result.goal);
    assert!(result.terminal);
}

#[test]
fn test_generator_backtrack_solves_diamond() {
    let cfg = diamond_config();
    let draft = BacktrackDraft::new(cfg.clone());
    let mut gen = QuestGenerator::new(cfg, draft, backtrack_decode(20, 100_000));
    let result = gen.generate();
    assert!(result.goal);
    assert!(result.terminal);
    assert!((result.reward - 1.0).abs() < 1e-6);
}

#[test]
fn test_generator_initial_state_matches_canonical_origin() {
    // The pruner's canonical origin must equal the generator's initial state
    // — the stateless-replay invariant.
    let cfg = diamond_config();
    let draft = TowardGoalDraft::new(cfg.clone());
    let gen = QuestGenerator::new(cfg, draft, greedy_decode(5));
    let initial = gen.initial_state();
    let replayed = gen
        .pruner
        .state_of(&[])
        .expect("empty trace → initial state");
    assert_eq!(initial.status, replayed.status);
}
