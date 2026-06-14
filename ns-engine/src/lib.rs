//! ns-engine: Modelless neuro-symbolic inference engine.
//!
//! Core thesis (katopz): the draft model provides fluency, the ConstraintPruner
//! provides correctness, and the speculative decode loop connects them.
//! Intelligence evolves in the symbolic pruner layer, not in model weights.
//!
//! See `.plans/006_modelless_neuro_symbolic_scaffold.md` for the full roadmap.

pub mod bandit;
pub mod decode;
pub mod draft;
pub mod kg;
pub mod pruners;
pub mod traits;
pub mod types;

pub use bandit::{BanditPolicy, BanditPruner};
pub use decode::speculative_decode;
pub use kg::{DomainLatent, InMemoryKgStore, SchemaCentroid, ShardEmbedding};
pub use traits::{ConstraintPruner, DraftModel, KgStore, ScreeningPruner};
pub use types::{ArmId, DecodeConfig, DecodeResult, KgTriple, Logits, TokenId};

// Phase 3: WASM hot-swap pruners (behind `wasm-pruner` feature).
#[cfg(feature = "wasm-pruner")]
pub use pruners::{
    HotSwapPruner, ReloadError, ReloadOutcome, SwapRecord, WasmPruner, WasmPrunerError,
};
