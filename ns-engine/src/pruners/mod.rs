//! Concrete ConstraintPruner implementations.
//!
//! Each pruner enforces domain-specific constraints during speculative
//! decode. New domains = new pruners; the backbone stays the same.
//!
//! # Phase 2 additions
//!
//! - [`NgramScreeningPruner`] — graded n-gram probability as screen score
//! - [`RegexPruner`] — regex prefix-validity for constrained generation
//!
//! Both implement [`ScreeningPruner`](crate::traits::ScreeningPruner) and can
//! serve as arms of a [`BanditPruner`](crate::bandit::BanditPruner).
//!
//! # Phase 3 additions (behind `wasm-pruner` feature)
//!
//! - [`WasmPruner`] — WASM-interpreted pruner via the `wasmi` engine
//! - [`HotSwapPruner`] — wraps a `WasmPruner` with atomic hot-swap and
//!   BLAKE3 audit logging
//!
//! # Domain-specific pruner
//!
//! - [`BethereEscrowPruner`] — `ConstraintPruner` test oracle for the
//!   `bethere-escrow` Solana program state machine. Encodes the full
//!   deposit lifecycle (create → deposit → check-in → settle → close) as
//!   discrete, deterministic transition rules. Demonstrates the modelless
//!   thesis in a real production domain.

pub mod escrow;
pub mod ngram_screen;
pub mod no_pruner;
pub mod regex;
pub mod sudoku;

#[cfg(feature = "wasm-pruner")]
pub mod hot_swap;
#[cfg(feature = "wasm-pruner")]
pub mod wasm_pruner;

pub use escrow::{
    BethereEscrowPruner, ClockPhase, EscrowConfig, EscrowState, ACTION_CLAIM_FORFEITED,
    ACTION_CLOSE_DEPOSIT, ACTION_CLOSE_EVENT, ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT,
    ACTION_DEPOSIT, ACTION_INTROSPECTION, ACTION_MARK_CHECKED_IN, ACTION_REFUND,
    ACTION_ROLLOVER_DEPOSIT, ESCROW_ARM_ID, ESCROW_LABEL,
};
pub use ngram_screen::{NgramScreeningPruner, NGRAM_SCREEN_ARM_ID, NGRAM_SCREEN_LABEL};
pub use no_pruner::{NoPruner, NO_PRUNER_ARM_ID, NO_PRUNER_LABEL};
pub use regex::{
    RegexPruner, RegexPrunerError, MAX_DEFAULT_TOKEN as REGEX_MAX_DEFAULT_TOKEN, REGEX_ARM_ID,
    REGEX_LABEL,
};
pub use sudoku::{SudokuError, SudokuPruner, SUDOKU_ARM_ID, SUDOKU_LABEL};

#[cfg(feature = "wasm-pruner")]
pub use hot_swap::{HotSwapPruner, ReloadError, ReloadOutcome, SwapRecord};
#[cfg(feature = "wasm-pruner")]
pub use wasm_pruner::{
    WasmPruner, WasmPrunerError, DEFAULT_FUEL_PER_CALL, IS_VALID_EXPORT_NAME, MAX_PAGES,
    MEMORY_EXPORT_NAME, WASM_PAGE_SIZE,
};
