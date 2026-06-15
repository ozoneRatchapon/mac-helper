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
//!
//! # Phase 5 additions
//!
//! - [`JsonSchemaPruner`] — structural JSON prefix-validity pruner. Tracks
//!   container nesting, string state (escapes, unicode), number grammar,
//!   keyword prefixes, and structural expectations (colons, commas, close
//!   brackets) via an incremental [`JsonState`](json_schema::JsonState)
//!   state machine.
//! - [`BomberActionPruner`] — Bomberman domain legality pruner. Mirrors a
//!   full [`BomberState`] forward model (grid, bombs, chain reactions,
//!   destructible blocks) via stateless replay. First Phase 5 domain to ship
//!   BOTH a [`GameState`](crate::game::GameState) impl AND a matching
//!   [`ConstraintPruner`]. Graded screen scores actions by exit-distance
//!   reduction and bomb-placement utility.
//! - [`QuestActionPruner`] — RPG quest-tree legality pruner. Mirrors a
//!   [`QuestState`] forward model (prerequisite DAG, accept/complete/fail
//!   lifecycle, goal set) via stateless replay. Second Phase 5 domain to ship
//!   BOTH a [`GameState`](crate::game::GameState) impl AND a matching
//!   [`ConstraintPruner`]. Graded screen scores goal-path quest completion
//!   higher than off-path distraction; failing is weighted low.

pub mod bomber;
pub mod escrow;
pub mod json_schema;
pub mod ngram_screen;
pub mod no_pruner;
pub mod quest;
pub mod regex;
pub mod sudoku;

#[cfg(feature = "wasm-pruner")]
pub mod hot_swap;
#[cfg(feature = "wasm-pruner")]
pub mod wasm_pruner;

pub use bomber::{
    Bomb, BomberActionPruner, BomberConfig, BomberConfigError, BomberState, Cell, ACTION_MOVE_E,
    ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_W, ACTION_PLACE_BOMB, ACTION_WAIT, BOMBER_ARM_ID,
    BOMBER_LABEL, BOMBER_VOCAB,
};
pub use escrow::{
    BethereEscrowPruner, ClockPhase, EscrowConfig, EscrowState, ACTION_CLAIM_FORFEITED,
    ACTION_CLOSE_DEPOSIT, ACTION_CLOSE_EVENT, ACTION_CREATE_EVENT, ACTION_DEACTIVATE_EVENT,
    ACTION_DEPOSIT, ACTION_INTROSPECTION, ACTION_MARK_CHECKED_IN, ACTION_REFUND,
    ACTION_ROLLOVER_DEPOSIT, ESCROW_ARM_ID, ESCROW_LABEL,
};
pub use json_schema::{
    JsonError, JsonSchemaPruner, JsonState, JSON_SCHEMA_ARM_ID, JSON_SCHEMA_LABEL,
};
pub use ngram_screen::{NgramScreeningPruner, NGRAM_SCREEN_ARM_ID, NGRAM_SCREEN_LABEL};
pub use no_pruner::{NoPruner, NO_PRUNER_ARM_ID, NO_PRUNER_LABEL};
pub use quest::{
    decode_quest_action, encode_quest_action, QuestActionPruner, QuestConfig, QuestConfigError,
    QuestState, QuestStatus, ACTIONS_PER_QUEST, QUEST_ACTION_ACCEPT, QUEST_ACTION_COMPLETE,
    QUEST_ACTION_FAIL, QUEST_ARM_ID, QUEST_LABEL,
};
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
