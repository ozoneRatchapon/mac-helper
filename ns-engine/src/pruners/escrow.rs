//! `BethereEscrowPruner` — `ConstraintPruner` test oracle for the
//! `bethere-escrow` Solana program state machine.
//!
//! This is the **modelless thesis applied to a real production domain**:
//! the entire escrow lifecycle is encoded as discrete, checkable transition
//! rules — no model weights, no learned policy, no probabilistic inference.
//! The pruner answers one question: *is instruction `token` legal given the
//! instruction history `parent_tokens`?* That answer is binary, deterministic,
//! and faithful to the on-chain guards.
//!
//! # State machine
//!
//! The oracle models a single (event, attendee) pair through the full
//! deposit lifecycle:
//!
//! ```text
//! create_event → deposit → mark_checked_in → (refund | claim_forfeited | rollover_deposit)
//!                → deactivate_event → close_event → close_deposit
//! ```
//!
//! Faithful guard conditions are mirrored from the production source:
//! - `create_event.rs` — `event_end > now`, `refund_deadline > event_end`, `deposit_amount > 0`
//! - `deposit.rs` — `is_active`, deposit account `init` (not pre-existing)
//! - `mark_checked_in.rs` — `!checked_in`, `now <= event_end`
//! - `refund.rs` — `!refunded`, `now >= event_end`; no-show needs `now < refund_deadline`
//! - `claim_forfeited.rs` — `!checked_in`, `!refunded`, `now >= refund_deadline`
//! - `deactivate_event.rs` — `is_active`, organizer-signed
//! - `close_event.rs` — `!is_active`, vault empty (`total_deposited == total_refunded + total_forfeited`)
//! - `close_deposit.rs` — `(refunded && signer == attendee) || event_closed`
//! - `rollover_deposit.rs` — `checked_in`, `!refunded`, target event active
//! - `introspection.rs` — read-only, always valid
//!
//! The program's `refunded` flag is a unified "settled" bit — set by
//! `refund`, `claim_forfeited`, and `rollover_deposit` (source). This oracle
//! mirrors that with a single [`EscrowState::settled`] field.
//!
//! # Time modeling
//!
//! Wall-clock time gates several transitions, but [`ConstraintPruner::is_valid`]
//! has no time parameter. The oracle holds a [`ClockPhase`] field (advanced
//! by the test via [`BethereEscrowPruner::advance_clock`]). Phases:
//! - [`ClockPhase::Active`] — `now <= event_end` (create, deposit, checkin allowed)
//! - [`ClockPhase::RefundWindow`] — `event_end < now < refund_deadline` (refund allowed)
//! - [`ClockPhase::PostDeadline`] — `now >= refund_deadline` (claim_forfeited allowed)
//!
//! Boundary values (`now == event_end`, `now == refund_deadline`) are not
//! distinguished from the nearest phase; tests should pick phases that
//! satisfy the strict inequalities of the transition under test.
//!
//! # Multi-event scope
//!
//! `rollover_deposit` spans two events. The oracle models the **source**
//! deposit lifecycle; the target event is abstracted behind
//! [`EscrowConfig::target_event_active`]. After rollover, the source deposit
//! is `settled`; the target lifecycle is out of scope (new event, new oracle
//! instance).
//!
//! # Stateless derivation
//!
//! All structural state (created, active, deposited, checked_in, settled,
//! closed) is **derived from `parent_tokens`** by replaying the instruction
//! history — no mutable state in the pruner for structural facts. Only the
//! clock phase and config live in `&self`. This matches the ns-engine
//! "stateless derivation" convention: `propagate` and `on_backtrack` are
//! no-ops; `is_valid` recomputes state from scratch each call.
//!
//! # Token space
//!
//! The oracle uses its own 1-indexed token space. These are **NOT** the
//! on-chain instruction discriminators (which are 0=create_event, 1=deposit,
//! ...; see `bethere-escrow/src/lib.rs`). The mapping is documented at
//! each [`ACTION_*`] constant. Tests reference tokens by name, never by raw
//! discriminator, so the two spaces never collide.

use blake3::Hasher;

use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

// ── Oracle token space ─────────────────────────────────────────
//
// NOTE: These are oracle-local token IDs, distinct from the on-chain
// instruction discriminators. See module docs.

/// Instruction token: `create_event` — initialize EventEscrow + vault.
///
/// On-chain source: `instructions/create_event.rs`.
/// Guard: event not yet created, `now < event_end`, `refund_deadline > event_end`.
pub const ACTION_CREATE_EVENT: TokenId = 1;

/// Instruction token: `deposit` — attendee deposits into vault.
///
/// On-chain source: `instructions/deposit.rs`.
/// Guard: `is_active`, deposit account not pre-existing.
pub const ACTION_DEPOSIT: TokenId = 2;

/// Instruction token: `mark_checked_in` — organizer marks attendee checked in.
///
/// On-chain source: `instructions/mark_checked_in.rs`.
/// Guard: deposit exists, `!checked_in`, `now <= event_end`.
pub const ACTION_MARK_CHECKED_IN: TokenId = 3;

/// Instruction token: `refund` — refund deposit to attendee.
///
/// On-chain source: `instructions/refund.rs`.
/// Guard: `!settled`, `now >= event_end`; no-show needs `now < refund_deadline`.
pub const ACTION_REFUND: TokenId = 4;

/// Instruction token: `claim_forfeited` — organizer claims no-show deposit.
///
/// On-chain source: `instructions/claim_forfeited.rs`.
/// Guard: `!checked_in`, `!settled`, `now >= refund_deadline`.
pub const ACTION_CLAIM_FORFEITED: TokenId = 5;

/// Instruction token: `deactivate_event` — set `is_active = false`.
///
/// On-chain source: `instructions/deactivate_event.rs`.
/// Guard: `is_active`.
pub const ACTION_DEACTIVATE_EVENT: TokenId = 6;

/// Instruction token: `close_event` — close EventEscrow, reclaim rent.
///
/// On-chain source: `instructions/close_event.rs`.
/// Guard: `!is_active`, vault empty.
pub const ACTION_CLOSE_EVENT: TokenId = 7;

/// Instruction token: `close_deposit` — close AttendeeDeposit, reclaim rent.
///
/// On-chain source: `instructions/close_deposit.rs`.
/// Guard: `settled` (attendee path) OR `event_closed` (GC path).
pub const ACTION_CLOSE_DEPOSIT: TokenId = 8;

/// Instruction token: `rollover_deposit` — move checked-in deposit to new event.
///
/// On-chain source: `instructions/rollover_deposit.rs`.
/// Guard: `checked_in`, `!settled`, target event active.
pub const ACTION_ROLLOVER_DEPOSIT: TokenId = 9;

/// Instruction token: `introspection` — read-only account inspection.
///
/// On-chain source: `instructions/introspection.rs`.
/// Guard: none (always valid, read-only).
pub const ACTION_INTROSPECTION: TokenId = 10;

/// Stable arm identifier for [`BethereEscrowPruner`] within a
/// [`BanditPruner`](crate::bandit::BanditPruner).
///
/// Unique within ns-engine's built-in pruners (NoPruner=4, Regex=3, Ngram=2,
/// Sudoku=1). Must not collide when composed in a bandit.
pub const ESCROW_ARM_ID: ArmId = 5;

/// Default arm label reported by [`BethereEscrowPruner::arm_label`].
pub const ESCROW_LABEL: &str = "bethere-escrow";

// ── Clock phase ────────────────────────────────────────────────

/// Coarse wall-clock phase relative to event time boundaries.
///
/// See module docs for the boundary abstraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClockPhase {
    /// `now <= event_end` — event ongoing; create, deposit, checkin allowed.
    Active,
    /// `event_end < now < refund_deadline` — refunds open; checkin blocked.
    RefundWindow,
    /// `now >= refund_deadline` — no-show refunds blocked; claims allowed.
    PostDeadline,
}

impl ClockPhase {
    /// Advance to the next phase. [`ClockPhase::PostDeadline`] is terminal.
    pub fn next(self) -> Self {
        match self {
            Self::Active => Self::RefundWindow,
            Self::RefundWindow => Self::PostDeadline,
            Self::PostDeadline => Self::PostDeadline,
        }
    }
}

// ── Derived state ──────────────────────────────────────────────

/// Structural state derived by replaying an instruction history.
///
/// Every field is a pure function of `parent_tokens` (stateless derivation).
/// The clock phase is held separately in [`BethereEscrowPruner`] and gates
/// only the candidate transition, not the replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EscrowState {
    /// `EventEscrow` account initialized.
    pub event_created: bool,
    /// `EventEscrow.is_active` — true at create, false after deactivate.
    pub is_active: bool,
    /// `AttendeeDeposit` account initialized for this (event, attendee).
    pub deposit_exists: bool,
    /// `AttendeeDeposit.checked_in` — set by `mark_checked_in`.
    pub checked_in: bool,
    /// Unified "settled" bit — set by `refund`, `claim_forfeited`, `rollover`.
    ///
    /// Mirrors the program's `refunded` field, which is set true by all
    /// three settling instructions to prevent double-spend.
    pub settled: bool,
    /// `EventEscrow` closed (data zeroed) via `close_event`.
    pub event_closed: bool,
    /// `AttendeeDeposit` closed via `close_deposit`.
    pub deposit_closed: bool,
}

impl EscrowState {
    /// Reconstruct state by replaying `history` structurally.
    ///
    /// Each token's effect is applied only if its **structural** precondition
    /// holds at that point in the replay (clock gates are intentionally
    /// ignored during replay — the history is assumed to be a valid trace).
    /// Invalid or unknown tokens are no-ops.
    pub fn derive(history: &[TokenId]) -> Self {
        let mut state = Self::default();
        for &tok in history {
            state.apply(tok);
        }
        state
    }

    /// Apply a single token's structural effect, guarded by preconditions.
    ///
    /// Idempotent against invalid tokens: if the precondition fails, the
    /// state is unchanged.
    fn apply(&mut self, tok: TokenId) {
        match tok {
            ACTION_CREATE_EVENT => {
                if !self.event_created {
                    self.event_created = true;
                    self.is_active = true;
                }
            }
            ACTION_DEPOSIT => {
                if self.event_created && self.is_active && !self.deposit_exists {
                    self.deposit_exists = true;
                }
            }
            ACTION_MARK_CHECKED_IN => {
                if self.deposit_exists && !self.checked_in && !self.settled {
                    self.checked_in = true;
                }
            }
            ACTION_REFUND | ACTION_CLAIM_FORFEITED | ACTION_ROLLOVER_DEPOSIT => {
                if self.deposit_exists && !self.settled {
                    self.settled = true;
                }
            }
            ACTION_DEACTIVATE_EVENT => {
                if self.event_created && self.is_active {
                    self.is_active = false;
                }
            }
            ACTION_CLOSE_EVENT => {
                if self.event_created
                    && !self.is_active
                    && !self.event_closed
                    && (!self.deposit_exists || self.settled)
                {
                    self.event_closed = true;
                }
            }
            ACTION_CLOSE_DEPOSIT => {
                if self.deposit_exists
                    && !self.deposit_closed
                    && (self.settled || self.event_closed)
                {
                    self.deposit_closed = true;
                }
            }
            ACTION_INTROSPECTION => {
                // Read-only — no structural effect.
            }
            _ => {
                // Unknown token — no-op (defensive).
            }
        }
    }

    /// BLAKE3 hash of the state fields, for audit and reproducibility.
    ///
    /// Deterministic: identical states produce identical hashes. Useful for
    /// comparing oracle state against on-chain-derived state in tests.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&(self.event_created as u8).to_le_bytes());
        hasher.update(&(self.is_active as u8).to_le_bytes());
        hasher.update(&(self.deposit_exists as u8).to_le_bytes());
        hasher.update(&(self.checked_in as u8).to_le_bytes());
        hasher.update(&(self.settled as u8).to_le_bytes());
        hasher.update(&(self.event_closed as u8).to_le_bytes());
        hasher.update(&(self.deposit_closed as u8).to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Human-readable one-line description for test diagnostics.
    pub fn describe(&self) -> String {
        format!(
            "created={created} active={active} deposit={deposit} checked_in={checked_in} \
             settled={settled} event_closed={event_closed} deposit_closed={deposit_closed}",
            created = self.event_created,
            active = self.is_active,
            deposit = self.deposit_exists,
            checked_in = self.checked_in,
            settled = self.settled,
            event_closed = self.event_closed,
            deposit_closed = self.deposit_closed,
        )
    }
}

// ── Config ─────────────────────────────────────────────────────

/// Non-derivable configuration for the escrow oracle.
///
/// Captures invariants that cannot be reconstructed from instruction history
/// alone (amounts, signer roles, target-event existence).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EscrowConfig {
    /// Whether a valid target event exists for `rollover_deposit`.
    ///
    /// The real guard requires the target event to be active, same organizer,
    /// same mint, same deposit amount. The oracle abstracts these to a single
    /// boolean the test sets to model the target event's availability.
    ///
    /// `Default` is `target_event_active = false` — rollover is unavailable
    /// unless the test opts in via
    /// [`with_rollover_target`](Self::with_rollover_target).
    pub target_event_active: bool,
}

impl EscrowConfig {
    /// Create config with `target_event_active = true` (rollover available).
    pub fn with_rollover_target() -> Self {
        Self {
            target_event_active: true,
        }
    }
}

// ── Pruner ─────────────────────────────────────────────────────

/// `ConstraintPruner` test oracle for the bethere-escrow state machine.
///
/// # Example
///
/// ```
/// use ns_engine::pruners::{BethereEscrowPruner, ACTION_CREATE_EVENT, ACTION_DEPOSIT};
/// use ns_engine::traits::ConstraintPruner;
///
/// let pruner = BethereEscrowPruner::new();
///
/// // create_event with no history — legal.
/// assert!(pruner.is_valid(0, ACTION_CREATE_EVENT, &[]));
///
/// // deposit before create — illegal.
/// assert!(!pruner.is_valid(0, ACTION_DEPOSIT, &[]));
///
/// // deposit after create — legal.
/// assert!(pruner.is_valid(1, ACTION_DEPOSIT, &[ACTION_CREATE_EVENT]));
/// ```
pub struct BethereEscrowPruner {
    clock: ClockPhase,
    config: EscrowConfig,
}

impl BethereEscrowPruner {
    /// Create a new oracle at [`ClockPhase::Active`] with default config.
    pub fn new() -> Self {
        Self {
            clock: ClockPhase::Active,
            config: EscrowConfig::default(),
        }
    }

    /// Create a new oracle with explicit config.
    pub fn with_config(config: EscrowConfig) -> Self {
        Self {
            clock: ClockPhase::Active,
            config,
        }
    }

    /// Current clock phase.
    pub fn clock(&self) -> ClockPhase {
        self.clock
    }

    /// Set the clock phase explicitly.
    pub fn set_clock(&mut self, phase: ClockPhase) {
        self.clock = phase;
    }

    /// Advance the clock to the next phase (Active → RefundWindow → PostDeadline).
    ///
    /// [`ClockPhase::PostDeadline`] is terminal; further advances are no-ops.
    pub fn advance_clock(&mut self) {
        self.clock = self.clock.next();
    }

    /// Borrow the config.
    pub fn config(&self) -> &EscrowConfig {
        &self.config
    }

    /// Derive the structural state from a history (delegates to
    /// [`EscrowState::derive`]).
    pub fn state_of(&self, history: &[TokenId]) -> EscrowState {
        EscrowState::derive(history)
    }
}

impl Default for BethereEscrowPruner {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintPruner for BethereEscrowPruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        let s = EscrowState::derive(parent_tokens);
        self.check_transition(token, &s)
    }

    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        match self.is_valid(depth, token, parent_tokens) {
            true => 1.0,
            false => 0.0,
        }
    }
}

impl ScreeningPruner for BethereEscrowPruner {
    fn arm_id(&self) -> ArmId {
        ESCROW_ARM_ID
    }

    fn arm_label(&self) -> &str {
        ESCROW_LABEL
    }

    fn screen(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        let s = EscrowState::derive(parent_tokens);
        match self.check_transition(token, &s) {
            false => 0.0,
            true => self.transition_weight(token),
        }
    }
}

impl BethereEscrowPruner {
    /// Core transition legality check, factored for reuse by `is_valid` and
    /// `screen`.
    ///
    /// Pure function of `(token, derived_state, self.clock, self.config)`.
    fn check_transition(&self, token: TokenId, s: &EscrowState) -> bool {
        match token {
            ACTION_CREATE_EVENT => self.valid_create(s),
            ACTION_DEPOSIT => self.valid_deposit(s),
            ACTION_MARK_CHECKED_IN => self.valid_checkin(s),
            ACTION_REFUND => self.valid_refund(s),
            ACTION_CLAIM_FORFEITED => self.valid_claim(s),
            ACTION_DEACTIVATE_EVENT => self.valid_deactivate(s),
            ACTION_CLOSE_EVENT => self.valid_close_event(s),
            ACTION_CLOSE_DEPOSIT => self.valid_close_deposit(s),
            ACTION_ROLLOVER_DEPOSIT => self.valid_rollover(s),
            ACTION_INTROSPECTION => true,
            _ => false,
        }
    }

    fn valid_create(&self, s: &EscrowState) -> bool {
        !s.event_closed && !s.event_created && matches!(self.clock, ClockPhase::Active)
    }

    fn valid_deposit(&self, s: &EscrowState) -> bool {
        s.event_created && s.is_active && !s.event_closed && !s.deposit_exists
    }

    fn valid_checkin(&self, s: &EscrowState) -> bool {
        s.deposit_exists
            && !s.checked_in
            && !s.settled
            && !s.deposit_closed
            && matches!(self.clock, ClockPhase::Active)
    }

    fn valid_refund(&self, s: &EscrowState) -> bool {
        if !s.deposit_exists || s.settled || s.deposit_closed {
            return false;
        }
        match s.checked_in {
            // Checked-in attendees refund anytime after event_end.
            true => matches!(
                self.clock,
                ClockPhase::RefundWindow | ClockPhase::PostDeadline
            ),
            // No-shows must refund before the deadline.
            false => matches!(self.clock, ClockPhase::RefundWindow),
        }
    }

    fn valid_claim(&self, s: &EscrowState) -> bool {
        s.deposit_exists
            && !s.checked_in
            && !s.settled
            && !s.deposit_closed
            && matches!(self.clock, ClockPhase::PostDeadline)
    }

    fn valid_deactivate(&self, s: &EscrowState) -> bool {
        s.event_created && s.is_active && !s.event_closed
    }

    fn valid_close_event(&self, s: &EscrowState) -> bool {
        s.event_created && !s.is_active && !s.event_closed && (!s.deposit_exists || s.settled)
    }

    fn valid_close_deposit(&self, s: &EscrowState) -> bool {
        s.deposit_exists && !s.deposit_closed && (s.settled || s.event_closed)
    }

    fn valid_rollover(&self, s: &EscrowState) -> bool {
        s.deposit_exists
            && s.checked_in
            && !s.settled
            && !s.deposit_closed
            && self.config.target_event_active
    }

    /// Graded relevance weight for a valid transition, used by
    /// [`ScreeningPruner::screen`].
    ///
    /// - Settling / terminal actions → `1.0` (strong commitment).
    /// - Progressive lifecycle actions → `0.8`.
    /// - Read-only introspection → `0.5` (neutral).
    fn transition_weight(&self, token: TokenId) -> f32 {
        match token {
            ACTION_REFUND
            | ACTION_CLAIM_FORFEITED
            | ACTION_ROLLOVER_DEPOSIT
            | ACTION_CLOSE_EVENT
            | ACTION_CLOSE_DEPOSIT
            | ACTION_DEACTIVATE_EVENT => 1.0,
            ACTION_CREATE_EVENT | ACTION_DEPOSIT | ACTION_MARK_CHECKED_IN => 0.8,
            ACTION_INTROSPECTION => 0.5,
            _ => 0.0,
        }
    }
}
