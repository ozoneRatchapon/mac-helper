//! Integration tests for [`BethereEscrowPruner`] — the `ConstraintPruner`
//! test oracle for the `bethere-escrow` Solana program state machine.
//!
//! These tests exercise the **public API** of the pruner exactly as an
//! external test oracle would: feed instruction histories, assert transition
//! legality, and verify the derived state. They mirror the on-chain guard
//! conditions documented in
//! `bethere-escrow/src/instructions/{create_event,deposit,...}.rs`.
//!
//! Moved out of `src/pruners/escrow.rs` to keep the implementation file
//! under the 1024-line limit and to honor the "tests to tests folder" rule.

use ns_engine::pruners::{
    BethereEscrowPruner, ClockPhase, EscrowConfig, EscrowState, ACTION_CLAIM_FORFEITED,
    ACTION_CLOSE_DEPOSIT, ACTION_CLOSE_EVENT, ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT,
    ACTION_DEPOSIT, ACTION_INTROSPECTION, ACTION_MARK_CHECKED_IN, ACTION_REFUND,
    ACTION_ROLLOVER_DEPOSIT, ESCROW_ARM_ID, ESCROW_LABEL,
};
use ns_engine::traits::{ConstraintPruner, ScreeningPruner};
use ns_engine::types::TokenId;

// ── ClockPhase ──

#[test]
fn test_clock_phase_next_sequence() {
    assert_eq!(ClockPhase::Active.next(), ClockPhase::RefundWindow);
    assert_eq!(ClockPhase::RefundWindow.next(), ClockPhase::PostDeadline);
    assert_eq!(ClockPhase::PostDeadline.next(), ClockPhase::PostDeadline);
}

#[test]
fn test_advance_clock_three_steps() {
    let mut p = BethereEscrowPruner::new();
    assert_eq!(p.clock(), ClockPhase::Active);

    p.advance_clock();
    assert_eq!(p.clock(), ClockPhase::RefundWindow);

    p.advance_clock();
    assert_eq!(p.clock(), ClockPhase::PostDeadline);

    // Terminal — stays.
    p.advance_clock();
    assert_eq!(p.clock(), ClockPhase::PostDeadline);
}

#[test]
fn test_set_clock_explicit() {
    let mut p = BethereEscrowPruner::new();
    p.set_clock(ClockPhase::PostDeadline);
    assert_eq!(p.clock(), ClockPhase::PostDeadline);
}

// ── EscrowState::derive ──

#[test]
fn test_derive_empty_history() {
    let s = EscrowState::derive(&[]);
    assert_eq!(s, EscrowState::default());
}

#[test]
fn test_derive_create_only() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT]);
    assert!(s.event_created);
    assert!(s.is_active);
    assert!(!s.deposit_exists);
}

#[test]
fn test_derive_create_then_deposit() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT]);
    assert!(s.event_created);
    assert!(s.is_active);
    assert!(s.deposit_exists);
    assert!(!s.checked_in);
    assert!(!s.settled);
}

#[test]
fn test_derive_full_checkin_lifecycle() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN]);
    assert!(s.checked_in);
    assert!(!s.settled);
}

#[test]
fn test_derive_refund_settles() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_REFUND]);
    assert!(s.settled);
    assert!(s.deposit_exists);
}

#[test]
fn test_derive_claim_settles() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_CLAIM_FORFEITED]);
    assert!(s.settled);
}

#[test]
fn test_derive_rollover_settles() {
    let s = EscrowState::derive(&[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_MARK_CHECKED_IN,
        ACTION_ROLLOVER_DEPOSIT,
    ]);
    assert!(s.settled);
    assert!(s.checked_in);
}

#[test]
fn test_derive_deactivate_sets_inactive() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT]);
    assert!(s.event_created);
    assert!(!s.is_active);
}

#[test]
fn test_derive_close_event_after_settle() {
    let s = EscrowState::derive(&[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_REFUND,
        ACTION_DEACTIVATE_EVENT,
        ACTION_CLOSE_EVENT,
    ]);
    assert!(s.event_closed);
}

#[test]
fn test_derive_close_deposit_after_settle() {
    let s = EscrowState::derive(&[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_REFUND,
        ACTION_CLOSE_DEPOSIT,
    ]);
    assert!(s.deposit_closed);
}

#[test]
fn test_derive_close_event_without_deposit() {
    // No deposit made → vault trivially empty → close allowed.
    let s = EscrowState::derive(&[
        ACTION_CREATE_EVENT,
        ACTION_DEACTIVATE_EVENT,
        ACTION_CLOSE_EVENT,
    ]);
    assert!(s.event_closed);
}

#[test]
fn test_derive_unknown_token_noop() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, 999]);
    assert!(s.event_created);
    // Unknown token did not corrupt state.
    assert_eq!(s, EscrowState::derive(&[ACTION_CREATE_EVENT]));
}

#[test]
fn test_derive_idempotent_invalid_repeat() {
    // create_event twice — second is no-op.
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_CREATE_EVENT]);
    assert_eq!(s, EscrowState::derive(&[ACTION_CREATE_EVENT]));
}

#[test]
fn test_derive_introspection_noop() {
    let a = EscrowState::derive(&[ACTION_CREATE_EVENT]);
    let b = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_INTROSPECTION]);
    assert_eq!(a, b);
}

// ── EscrowState::hash ──

#[test]
fn test_state_hash_deterministic() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT]);
    assert_eq!(s.hash(), s.hash());
}

#[test]
fn test_state_hash_differs_on_state_change() {
    let a = EscrowState::derive(&[ACTION_CREATE_EVENT]);
    let b = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT]);
    assert_ne!(a.hash(), b.hash());
}

// ── EscrowState::describe ──

#[test]
fn test_state_describe_contains_fields() {
    let s = EscrowState::derive(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT]);
    let d = s.describe();
    assert!(d.contains("created=true"));
    assert!(d.contains("deposit=true"));
    assert!(d.contains("settled=false"));
}

// ── ConstraintPruner: create_event ──

#[test]
fn test_create_event_valid_empty_history() {
    let p = BethereEscrowPruner::new();
    assert!(p.is_valid(0, ACTION_CREATE_EVENT, &[]));
}

#[test]
fn test_create_event_invalid_if_already_created() {
    let p = BethereEscrowPruner::new();
    assert!(!p.is_valid(1, ACTION_CREATE_EVENT, &[ACTION_CREATE_EVENT]));
}

#[test]
fn test_create_event_invalid_after_event_closed() {
    let p = BethereEscrowPruner::new();
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEACTIVATE_EVENT,
        ACTION_CLOSE_EVENT,
    ];
    assert!(!p.is_valid(hist.len(), ACTION_CREATE_EVENT, hist));
}

#[test]
fn test_create_event_invalid_outside_active_phase() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    assert!(!p.is_valid(0, ACTION_CREATE_EVENT, &[]));
}

// ── ConstraintPruner: deposit ──

#[test]
fn test_deposit_valid_after_create() {
    let p = BethereEscrowPruner::new();
    assert!(p.is_valid(1, ACTION_DEPOSIT, &[ACTION_CREATE_EVENT]));
}

#[test]
fn test_deposit_invalid_before_create() {
    let p = BethereEscrowPruner::new();
    assert!(!p.is_valid(0, ACTION_DEPOSIT, &[]));
}

#[test]
fn test_deposit_invalid_if_inactive() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT];
    assert!(!p.is_valid(2, ACTION_DEPOSIT, hist));
}

#[test]
fn test_deposit_invalid_if_already_exists() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(2, ACTION_DEPOSIT, hist));
}

#[test]
fn test_deposit_valid_in_refund_window_if_active() {
    // Program allows deposit whenever is_active, regardless of clock.
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    assert!(p.is_valid(1, ACTION_DEPOSIT, &[ACTION_CREATE_EVENT]));
}

// ── ConstraintPruner: mark_checked_in ──

#[test]
fn test_checkin_valid_active_phase() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(p.is_valid(2, ACTION_MARK_CHECKED_IN, hist));
}

#[test]
fn test_checkin_invalid_without_deposit() {
    let p = BethereEscrowPruner::new();
    assert!(!p.is_valid(1, ACTION_MARK_CHECKED_IN, &[ACTION_CREATE_EVENT]));
}

#[test]
fn test_checkin_invalid_if_already_checked_in() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(!p.is_valid(3, ACTION_MARK_CHECKED_IN, hist));
}

#[test]
fn test_checkin_invalid_after_event_end() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(2, ACTION_MARK_CHECKED_IN, hist));
}

#[test]
fn test_checkin_invalid_after_settle() {
    let p = BethereEscrowPruner::new();
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_MARK_CHECKED_IN,
        ACTION_ROLLOVER_DEPOSIT,
    ];
    assert!(!p.is_valid(4, ACTION_MARK_CHECKED_IN, hist));
}

// ── ConstraintPruner: refund ──

#[test]
fn test_refund_no_show_valid_in_window() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(p.is_valid(2, ACTION_REFUND, hist));
}

#[test]
fn test_refund_no_show_invalid_in_active_phase() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(2, ACTION_REFUND, hist));
}

#[test]
fn test_refund_no_show_invalid_after_deadline() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(2, ACTION_REFUND, hist));
}

#[test]
fn test_refund_checked_in_valid_in_window() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(p.is_valid(3, ACTION_REFUND, hist));
}

#[test]
fn test_refund_checked_in_valid_after_deadline() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(p.is_valid(3, ACTION_REFUND, hist));
}

#[test]
fn test_refund_invalid_if_already_settled() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_REFUND];
    assert!(!p.is_valid(3, ACTION_REFUND, hist));
}

#[test]
fn test_refund_invalid_without_deposit() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    assert!(!p.is_valid(1, ACTION_REFUND, &[ACTION_CREATE_EVENT]));
}

// ── ConstraintPruner: claim_forfeited ──

#[test]
fn test_claim_valid_no_show_after_deadline() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(p.is_valid(2, ACTION_CLAIM_FORFEITED, hist));
}

#[test]
fn test_claim_invalid_checked_in_attendee() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(!p.is_valid(3, ACTION_CLAIM_FORFEITED, hist));
}

#[test]
fn test_claim_invalid_before_deadline() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(2, ACTION_CLAIM_FORFEITED, hist));
}

#[test]
fn test_claim_invalid_if_already_settled() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_CLAIM_FORFEITED];
    assert!(!p.is_valid(3, ACTION_CLAIM_FORFEITED, hist));
}

// ── ConstraintPruner: deactivate_event ──

#[test]
fn test_deactivate_valid_when_active() {
    let p = BethereEscrowPruner::new();
    assert!(p.is_valid(1, ACTION_DEACTIVATE_EVENT, &[ACTION_CREATE_EVENT]));
}

#[test]
fn test_deactivate_invalid_when_already_inactive() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT];
    assert!(!p.is_valid(2, ACTION_DEACTIVATE_EVENT, hist));
}

#[test]
fn test_deactivate_invalid_before_create() {
    let p = BethereEscrowPruner::new();
    assert!(!p.is_valid(0, ACTION_DEACTIVATE_EVENT, &[]));
}

// ── ConstraintPruner: close_event ──

#[test]
fn test_close_event_valid_after_deactivate_and_settle() {
    let p = BethereEscrowPruner::new();
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_REFUND,
        ACTION_DEACTIVATE_EVENT,
    ];
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_EVENT, hist));
}

#[test]
fn test_close_event_invalid_while_active() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT];
    assert!(!p.is_valid(hist.len(), ACTION_CLOSE_EVENT, hist));
}

#[test]
fn test_close_event_invalid_with_unsettled_deposit() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_DEACTIVATE_EVENT];
    assert!(!p.is_valid(hist.len(), ACTION_CLOSE_EVENT, hist));
}

#[test]
fn test_close_event_valid_without_any_deposit() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT];
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_EVENT, hist));
}

#[test]
fn test_close_event_invalid_if_already_closed() {
    let p = BethereEscrowPruner::new();
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_REFUND,
        ACTION_DEACTIVATE_EVENT,
        ACTION_CLOSE_EVENT,
    ];
    assert!(!p.is_valid(hist.len(), ACTION_CLOSE_EVENT, hist));
}

// ── ConstraintPruner: close_deposit ──

#[test]
fn test_close_deposit_valid_after_settle() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_REFUND];
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_DEPOSIT, hist));
}

#[test]
fn test_close_deposit_valid_via_gc_path() {
    // Event closed → anyone can close the deposit (GC path).
    let p = BethereEscrowPruner::new();
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_REFUND,
        ACTION_DEACTIVATE_EVENT,
        ACTION_CLOSE_EVENT,
    ];
    // Note: deposit was settled by refund, so this passes via settled too,
    // but the GC path (event_closed) also independently satisfies.
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_DEPOSIT, hist));
}

#[test]
fn test_close_deposit_invalid_before_settle() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(hist.len(), ACTION_CLOSE_DEPOSIT, hist));
}

#[test]
fn test_close_deposit_invalid_if_already_closed() {
    let p = BethereEscrowPruner::new();
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_REFUND,
        ACTION_CLOSE_DEPOSIT,
    ];
    assert!(!p.is_valid(hist.len(), ACTION_CLOSE_DEPOSIT, hist));
}

// ── ConstraintPruner: rollover_deposit ──

#[test]
fn test_rollover_valid_checked_in_with_target() {
    let p = BethereEscrowPruner::with_config(EscrowConfig::with_rollover_target());
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(p.is_valid(hist.len(), ACTION_ROLLOVER_DEPOSIT, hist));
}

#[test]
fn test_rollover_invalid_without_target_event() {
    let p = BethereEscrowPruner::new(); // default config: no target
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(!p.is_valid(hist.len(), ACTION_ROLLOVER_DEPOSIT, hist));
}

#[test]
fn test_rollover_invalid_if_not_checked_in() {
    let p = BethereEscrowPruner::with_config(EscrowConfig::with_rollover_target());
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(hist.len(), ACTION_ROLLOVER_DEPOSIT, hist));
}

#[test]
fn test_rollover_invalid_if_already_settled() {
    let p = BethereEscrowPruner::with_config(EscrowConfig::with_rollover_target());
    let hist = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_MARK_CHECKED_IN,
        ACTION_ROLLOVER_DEPOSIT,
    ];
    assert!(!p.is_valid(hist.len(), ACTION_ROLLOVER_DEPOSIT, hist));
}

// ── ConstraintPruner: introspection ──

#[test]
fn test_introspection_always_valid() {
    let p = BethereEscrowPruner::new();
    assert!(p.is_valid(0, ACTION_INTROSPECTION, &[]));
    assert!(p.is_valid(5, ACTION_INTROSPECTION, &[ACTION_CREATE_EVENT]));
}

// ── ConstraintPruner: unknown tokens ──

#[test]
fn test_unknown_token_invalid() {
    let p = BethereEscrowPruner::new();
    assert!(!p.is_valid(0, 999, &[]));
    assert!(!p.is_valid(0, 0, &[]));
    assert!(!p.is_valid(0, TokenId::MAX, &[]));
}

// ── End-to-end lifecycle scenarios ──

#[test]
fn test_full_refund_lifecycle_legal() {
    let mut p = BethereEscrowPruner::new();
    let mut hist: Vec<TokenId> = Vec::new();

    // create
    assert!(p.is_valid(hist.len(), ACTION_CREATE_EVENT, &hist));
    hist.push(ACTION_CREATE_EVENT);

    // deposit
    assert!(p.is_valid(hist.len(), ACTION_DEPOSIT, &hist));
    hist.push(ACTION_DEPOSIT);

    // checkin
    assert!(p.is_valid(hist.len(), ACTION_MARK_CHECKED_IN, &hist));
    hist.push(ACTION_MARK_CHECKED_IN);

    // advance clock to refund window
    p.advance_clock();

    // refund (checked-in, in window)
    assert!(p.is_valid(hist.len(), ACTION_REFUND, &hist));
    hist.push(ACTION_REFUND);

    // close_deposit
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_DEPOSIT, &hist));
    hist.push(ACTION_CLOSE_DEPOSIT);

    // deactivate
    assert!(p.is_valid(hist.len(), ACTION_DEACTIVATE_EVENT, &hist));
    hist.push(ACTION_DEACTIVATE_EVENT);

    // close_event
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_EVENT, &hist));
}

#[test]
fn test_full_claim_lifecycle_legal() {
    let mut p = BethereEscrowPruner::new();
    let mut hist: Vec<TokenId> = Vec::new();

    hist.push(ACTION_CREATE_EVENT);
    hist.push(ACTION_DEPOSIT);
    // No checkin — no-show.

    // Advance to post-deadline.
    p.advance_clock();
    p.advance_clock();

    assert!(p.is_valid(hist.len(), ACTION_CLAIM_FORFEITED, &hist));
    hist.push(ACTION_CLAIM_FORFEITED);
    assert!(p.state_of(&hist).settled);

    hist.push(ACTION_CLOSE_DEPOSIT);
    hist.push(ACTION_DEACTIVATE_EVENT);
    assert!(p.is_valid(hist.len(), ACTION_CLOSE_EVENT, &hist));
}

#[test]
fn test_full_rollover_lifecycle_legal() {
    let p = BethereEscrowPruner::with_config(EscrowConfig::with_rollover_target());
    let mut hist: Vec<TokenId> = Vec::new();

    hist.push(ACTION_CREATE_EVENT);
    hist.push(ACTION_DEPOSIT);
    hist.push(ACTION_MARK_CHECKED_IN);

    assert!(p.is_valid(hist.len(), ACTION_ROLLOVER_DEPOSIT, &hist));
    hist.push(ACTION_ROLLOVER_DEPOSIT);
    assert!(p.state_of(&hist).settled);
}

#[test]
fn test_no_show_cannot_refund_after_deadline() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!(!p.is_valid(hist.len(), ACTION_REFUND, hist));
}

#[test]
fn test_checked_in_cannot_be_claimed() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];
    assert!(!p.is_valid(hist.len(), ACTION_CLAIM_FORFEITED, hist));
}

#[test]
fn test_double_refund_blocked() {
    let mut p = BethereEscrowPruner::new();
    p.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_REFUND];
    assert!(!p.is_valid(hist.len(), ACTION_REFUND, hist));
}

// ── manifold_score (binary via ConstraintPruner) ──

#[test]
fn test_manifold_score_binary() {
    let p = BethereEscrowPruner::new();
    assert!((p.manifold_score(0, ACTION_CREATE_EVENT, &[]) - 1.0).abs() < 1e-6);
    assert!((p.manifold_score(0, ACTION_DEPOSIT, &[]) - 0.0).abs() < 1e-6);
}

// ── ScreeningPruner ──

#[test]
fn test_arm_id_matches_constant() {
    let p = BethereEscrowPruner::new();
    assert_eq!(p.arm_id(), ESCROW_ARM_ID);
}

#[test]
fn test_arm_label_matches_constant() {
    let p = BethereEscrowPruner::new();
    assert_eq!(p.arm_label(), ESCROW_LABEL);
}

#[test]
fn test_screen_invalid_is_zero() {
    let p = BethereEscrowPruner::new();
    assert!((p.screen(0, ACTION_DEPOSIT, &[]) - 0.0).abs() < 1e-6);
}

#[test]
fn test_screen_settling_action_weighted_one() {
    // refund invalid in Active → 0.0
    let p_active = BethereEscrowPruner::new();
    assert!((p_active.screen(0, ACTION_REFUND, &[ACTION_CREATE_EVENT]) - 0.0).abs() < 1e-6);

    // Advance to window → refund valid → settling weight 1.0
    let mut p_window = BethereEscrowPruner::new();
    p_window.advance_clock();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];
    assert!((p_window.screen(hist.len(), ACTION_REFUND, hist) - 1.0).abs() < 1e-6);
}

#[test]
fn test_screen_progressive_action_weighted_eight() {
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT];
    assert!((p.screen(hist.len(), ACTION_DEPOSIT, hist) - 0.8).abs() < 1e-6);
}

#[test]
fn test_screen_introspection_weighted_half() {
    let p = BethereEscrowPruner::new();
    assert!((p.screen(0, ACTION_INTROSPECTION, &[]) - 0.5).abs() < 1e-6);
}

#[test]
fn test_screening_pruner_trait_object() {
    let p: Box<dyn ScreeningPruner> = Box::new(BethereEscrowPruner::new());
    assert_eq!(p.arm_id(), ESCROW_ARM_ID);
    assert_eq!(p.arm_label(), ESCROW_LABEL);
    assert!(p.is_valid(0, ACTION_CREATE_EVENT, &[]));
}

// ── state_of accessor ──

#[test]
fn test_state_of_accessor() {
    let p = BethereEscrowPruner::new();
    let s = p.state_of(&[ACTION_CREATE_EVENT, ACTION_DEPOSIT]);
    assert!(s.event_created);
    assert!(s.deposit_exists);
}

// ── Send + Sync ──

#[test]
fn test_send_sync_bounds() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BethereEscrowPruner>();
    assert_send_sync::<EscrowState>();
    assert_send_sync::<EscrowConfig>();
    assert_send_sync::<ClockPhase>();
}

#[test]
fn test_default_equals_new() {
    let a = BethereEscrowPruner::new();
    let b = BethereEscrowPruner::default();
    assert_eq!(a.clock(), b.clock());
    assert_eq!(a.config(), b.config());
}

// ── Test-oracle usage: derived state agrees with step-by-step validity ──

#[test]
fn test_oracle_derived_state_matches_progressive_validity() {
    // A hand-written legal refund trace must produce a state where each
    // subsequent guard sees the expected flags. This is the core oracle
    // contract: the pruner and the state machine must agree.
    let mut p = BethereEscrowPruner::new();
    let trace: &[TokenId] = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT, ACTION_MARK_CHECKED_IN];

    // After the trace, every prefix should be replayable to the same state
    // the pruner would compute via parent_tokens.
    for split in 0..=trace.len() {
        let prefix = &trace[..split];
        let direct = EscrowState::derive(prefix);
        let via_pruner = p.state_of(prefix);
        assert_eq!(direct, via_pruner, "state mismatch at split={split}");
    }

    // Advance to refund window — refund must now be legal against the trace.
    p.advance_clock();
    assert!(p.is_valid(trace.len(), ACTION_REFUND, trace));
}

#[test]
fn test_oracle_rejects_every_invalid_terminal_in_active_phase() {
    // In the Active phase, none of the settling/terminal actions should be
    // legal directly after a deposit. This is the oracle's negative contract.
    let p = BethereEscrowPruner::new();
    let hist = &[ACTION_CREATE_EVENT, ACTION_DEPOSIT];

    for &terminal in &[
        ACTION_REFUND,
        ACTION_CLAIM_FORFEITED,
        ACTION_CLOSE_EVENT,
        ACTION_ROLLOVER_DEPOSIT,
    ] {
        assert!(
            !p.is_valid(hist.len(), terminal, hist),
            "terminal {terminal} must be invalid in Active phase after raw deposit"
        );
    }
}

#[test]
fn test_oracle_state_hash_stable_across_replays() {
    // Deterministic BLAKE3 state fingerprint — same trace → same hash, no
    // matter how many times derived. Enables diffing oracle state against
    // on-chain-derived state in CI.
    let trace: &[TokenId] = &[
        ACTION_CREATE_EVENT,
        ACTION_DEPOSIT,
        ACTION_MARK_CHECKED_IN,
        ACTION_ROLLOVER_DEPOSIT,
    ];

    let h1 = EscrowState::derive(trace).hash();
    let h2 = EscrowState::derive(trace).hash();
    let h3 = BethereEscrowPruner::new().state_of(trace).hash();

    assert_eq!(h1, h2, "re-derivation must be stable");
    assert_eq!(h1, h3, "pruner accessor must agree with direct derive");
}
