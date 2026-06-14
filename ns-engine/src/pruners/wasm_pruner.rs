//! WASM-interpreted [`ConstraintPruner`].
//!
//! Loads a WebAssembly module that exports:
//!   - `memory` — linear memory used as scratch space for `parent_tokens`.
//!   - `is_valid(depth: i32, token: i32, parent_ptr: i32, parent_len: i32) -> i32`
//!     Returns 1 to accept the candidate, 0 (or any other value) to reject.
//!
//! # Sandbox safety
//!
//! - **Fuel**: each `is_valid` call resets the interpreter fuel to a fixed
//!   budget ([`DEFAULT_FUEL_PER_CALL`] or a custom value via
//!   [`WasmPruner::with_fuel`]). Runaway loops trap and are reported as
//!   rejection.
//! - **No host imports**: the WASM module runs with zero host capabilities —
//!   no I/O, no clocks, no random, no imports. The linker exposes nothing.
//! - **Memory cap**: the pruner may grow linear memory up to [`MAX_PAGES`]
//!   (16 MiB). Larger growth requests are denied and reported as rejection.
//!
//! # ABI contract
//!
//! Before each `is_valid` call, the host packs `parent_tokens` as
//! little-endian `i32` values into the WASM module's linear memory at
//! offset [`PARENT_TOKENS_OFFSET`] (0). The WASM pruner reads them via
//! `i32.load offset=4*i` from `parent_ptr`. Tokens are sign-cast on the
//! host (`u32 as i32`); the bit pattern is preserved so the WASM side may
//! reinterpret as needed.
//!
//! # Threading
//!
//! All trait methods serialize through an internal [`Mutex`], making
//! `WasmPruner` [`Send`] + [`Sync`]. The mutex is held only across the
//! fuel reset, memory write, and WASM call — never across user code.

use crate::traits::ConstraintPruner;
use crate::types::TokenId;
use std::sync::Mutex;
use wasmi::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

/// Default fuel budget per `is_valid` call. ~100k instructions covers any
/// reasonable single-token validity check while bounding runaway loops.
pub const DEFAULT_FUEL_PER_CALL: u64 = 100_000;

/// Name of the WASM linear memory export that pruner modules MUST provide.
pub const MEMORY_EXPORT_NAME: &str = "memory";

/// Name of the `is_valid` function export that pruner modules MUST provide.
pub const IS_VALID_EXPORT_NAME: &str = "is_valid";

/// WASM page size in bytes (64 KiB per spec).
pub const WASM_PAGE_SIZE: u64 = 65_536;

/// Maximum number of memory pages a WASM pruner may grow to.
/// 256 pages = 16 MiB scratch cap.
pub const MAX_PAGES: u64 = 256;

/// Scratch offset in linear memory where `parent_tokens` are written.
pub const PARENT_TOKENS_OFFSET: usize = 0;

/// Bytes per packed token in the ABI (WASM `i32`).
const TOKEN_PACK_SIZE: usize = 4;

/// Errors raised while compiling or instantiating a WASM pruner module.
#[derive(Debug)]
pub enum WasmPrunerError {
    /// WASM bytes failed validation or compilation.
    Module(String),
    /// Module does not export the required linear memory.
    MissingMemory,
    /// Module does not export the required `is_valid` function with the
    /// expected signature `(i32, i32, i32, i32) -> i32`.
    MissingIsValid,
    /// Module instantiation or start-function execution failed.
    Instantiation(String),
}

impl std::fmt::Display for WasmPrunerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Module(s) => write!(f, "WASM module compilation failed: {s}"),
            Self::MissingMemory => write!(
                f,
                "WASM module missing required memory export '{MEMORY_EXPORT_NAME}'"
            ),
            Self::MissingIsValid => write!(
                f,
                "WASM module missing required function export '{IS_VALID_EXPORT_NAME}' with signature (i32,i32,i32,i32)->i32"
            ),
            Self::Instantiation(s) => write!(f, "WASM instantiation failed: {s}"),
        }
    }
}

impl std::error::Error for WasmPrunerError {}

/// Internal mutable state, guarded by the outer [`Mutex`].
#[derive(Debug)]
struct WasmState {
    store: Store<()>,
    memory: Memory,
    is_valid: TypedFunc<(i32, i32, i32, i32), i32>,
    fuel_per_call: u64,
}

/// WASM-interpreted [`ConstraintPruner`].
///
/// Holds a compiled [`Module`] and a single live [`Store`]. All
/// [`ConstraintPruner`] trait methods serialize through an internal
/// [`Mutex`], making `WasmPruner` [`Send`] + [`Sync`].
#[derive(Debug)]
pub struct WasmPruner {
    engine: Engine,
    module: Module,
    state: Mutex<WasmState>,
}

impl WasmPruner {
    /// Compile and instantiate a WASM pruner with the default fuel budget.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmPrunerError> {
        Self::with_fuel(wasm_bytes, DEFAULT_FUEL_PER_CALL)
    }

    /// Compile and instantiate a WASM pruner with a custom fuel budget.
    ///
    /// `fuel` is the number of WASM instructions allowed per `is_valid`
    /// call. Set lower for tighter sandboxing, higher for complex pruners.
    /// The same budget also bounds any start function executed during
    /// instantiation.
    pub fn with_fuel(wasm_bytes: &[u8], fuel: u64) -> Result<Self, WasmPrunerError> {
        let mut config = Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmPrunerError::Module(format!("{e:?}")))?;

        let linker = <Linker<()>>::new(&engine);
        let mut store = Store::new(&engine, ());
        // Seed initial fuel so any start function is bounded too.
        let _ = store.set_fuel(fuel);

        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|e| WasmPrunerError::Instantiation(format!("{e:?}")))?;

        let memory = instance
            .get_memory(&store, MEMORY_EXPORT_NAME)
            .ok_or(WasmPrunerError::MissingMemory)?;
        let is_valid = instance
            .get_typed_func::<(i32, i32, i32, i32), i32>(&store, IS_VALID_EXPORT_NAME)
            .map_err(|_| WasmPrunerError::MissingIsValid)?;

        Ok(Self {
            engine,
            module,
            state: Mutex::new(WasmState {
                store,
                memory,
                is_valid,
                fuel_per_call: fuel,
            }),
        })
    }

    /// Read-only access to the compiled module.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Read-only access to the engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Update the per-call fuel budget. Applies to subsequent `is_valid`
    /// calls. Returns the previous budget.
    ///
    /// Recovers automatically from a poisoned mutex (e.g. a previous call
    /// panicked while holding the lock) by writing through the poisoned
    /// guard.
    pub fn set_fuel_per_call(&mut self, fuel: u64) -> u64 {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let prev = state.fuel_per_call;
        state.fuel_per_call = fuel;
        prev
    }

    /// Ensure linear memory has at least `needed_bytes` capacity, growing
    /// within the [`MAX_PAGES`] cap. Returns `false` if growth is required
    /// but impossible.
    fn ensure_capacity(state: &mut WasmState, needed_bytes: usize) -> bool {
        let current = state.memory.data_size(&state.store);
        if current >= needed_bytes {
            return true;
        }
        let needed_pages = needed_bytes.div_ceil(WASM_PAGE_SIZE as usize) as u64;
        let current_pages = (current / WASM_PAGE_SIZE as usize) as u64;
        let Some(delta) = needed_pages.checked_sub(current_pages) else {
            return true;
        };
        if current_pages.saturating_add(delta) > MAX_PAGES {
            return false;
        }
        state.memory.grow(&mut state.store, delta).is_ok()
    }

    /// Write `parent_tokens` into linear memory at [`PARENT_TOKENS_OFFSET`].
    /// Returns `false` on memory-access failure or capacity overflow.
    fn write_parent_tokens(state: &mut WasmState, parent_tokens: &[TokenId]) -> bool {
        let Some(parent_bytes_len) = parent_tokens.len().checked_mul(TOKEN_PACK_SIZE) else {
            return false;
        };
        if !Self::ensure_capacity(state, parent_bytes_len) {
            return false;
        }
        // Pack tokens as little-endian i32 (bit pattern identical to u32 LE).
        let mut buf: Vec<u8> = Vec::with_capacity(parent_bytes_len);
        for t in parent_tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        state
            .memory
            .write(&mut state.store, PARENT_TOKENS_OFFSET, &buf)
            .is_ok()
    }

    /// Evaluate one candidate via the WASM `is_valid` function.
    /// Refuels the store immediately before the call. Any trap (fuel
    /// exhaustion, memory fault, etc.) is reported as rejection.
    fn evaluate(state: &mut WasmState, depth: usize, token: TokenId, parent_len: usize) -> bool {
        let _ = state.store.set_fuel(state.fuel_per_call);
        let args = (
            depth as i32,
            token as i32,
            PARENT_TOKENS_OFFSET as i32,
            parent_len as i32,
        );
        matches!(state.is_valid.call(&mut state.store, args), Ok(1))
    }
}

impl ConstraintPruner for WasmPruner {
    fn is_valid(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !Self::write_parent_tokens(&mut state, parent_tokens) {
            return false;
        }
        Self::evaluate(&mut state, depth, token, parent_tokens.len())
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !Self::write_parent_tokens(&mut state, parent_tokens) {
            results.fill(false);
            return;
        }
        let n = candidates.len().min(results.len());
        for i in 0..n {
            results[i] = Self::evaluate(&mut state, depth, candidates[i], parent_tokens.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ConstraintPruner;

    /// Always-accept pruner (returns 1 for any input).
    const ALWAYS_ACCEPT_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1))
    "#;

    /// Always-reject pruner (returns 0 for any input).
    const ALWAYS_REJECT_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 0))
    "#;

    /// Accept iff token != 0.
    const NONZERO_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param $d i32) (param $t i32) (param $p i32) (param $l i32) (result i32)
            local.get $t
            i32.const 0
            i32.ne))
    "#;

    /// Accept iff parent_len > 0 (verifies parent_len is passed correctly).
    const HAS_PARENTS_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param $d i32) (param $t i32) (param $p i32) (param $l i32) (result i32)
            local.get $l
            i32.const 0
            i32.gt_u))
    "#;

    /// Accept iff first parent token == 5 (verifies memory is written).
    const FIRST_PARENT_FIVE_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param $d i32) (param $t i32) (param $p i32) (param $l i32) (result i32)
            local.get $l
            i32.eqz
            if (result i32)
              i32.const 1
            else
              local.get $p
              i32.load
              i32.const 5
              i32.eq
            end))
    "#;

    /// Accept iff depth (param 0) is even.
    const EVEN_DEPTH_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param $d i32) (param $t i32) (param $p i32) (param $l i32) (result i32)
            local.get $d
            i32.const 1
            i32.and
            i32.eqz))
    "#;

    /// Infinite loop. Should trap on fuel exhaustion and report false.
    const INFINITE_LOOP_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            loop
              br 0
            end
            i32.const 1))
    "#;

    fn parse(wat: &str) -> Vec<u8> {
        wat::parse_bytes(wat.as_bytes())
            .expect("WAT parse failure")
            .into_owned()
    }

    #[test]
    fn test_new_compiles_valid_module() {
        let bytes = parse(ALWAYS_ACCEPT_WAT);
        let pruner = WasmPruner::new(&bytes);
        assert!(
            pruner.is_ok(),
            "valid module should compile: {:?}",
            pruner.err()
        );
    }

    #[test]
    fn test_new_rejects_missing_memory() {
        let wat = r#"
            (module
              (func (export "is_valid") (param i32 i32 i32 i32) (result i32) i32.const 1))
        "#;
        let bytes = parse(wat);
        let err = WasmPruner::new(&bytes).unwrap_err();
        assert!(matches!(err, WasmPrunerError::MissingMemory), "got: {err}");
    }

    #[test]
    fn test_new_rejects_missing_is_valid() {
        let wat = r#"
            (module
              (memory (export "memory") 1))
        "#;
        let bytes = parse(wat);
        let err = WasmPruner::new(&bytes).unwrap_err();
        assert!(matches!(err, WasmPrunerError::MissingIsValid), "got: {err}");
    }

    #[test]
    fn test_new_rejects_wrong_is_valid_signature() {
        // is_valid with wrong arity
        let wat = r#"
            (module
              (memory (export "memory") 1)
              (func (export "is_valid") (param i32) (result i32) i32.const 1))
        "#;
        let bytes = parse(wat);
        let err = WasmPruner::new(&bytes).unwrap_err();
        assert!(matches!(err, WasmPrunerError::MissingIsValid), "got: {err}");
    }

    #[test]
    fn test_new_rejects_wrong_is_valid_return_type() {
        // is_valid returns i64 instead of i32
        let wat = r#"
            (module
              (memory (export "memory") 1)
              (func (export "is_valid") (param i32 i32 i32 i32) (result i64) i64.const 1))
        "#;
        let bytes = parse(wat);
        let err = WasmPruner::new(&bytes).unwrap_err();
        assert!(matches!(err, WasmPrunerError::MissingIsValid), "got: {err}");
    }

    #[test]
    fn test_new_rejects_invalid_wasm_bytes() {
        let bytes = b"not wasm";
        let err = WasmPruner::new(bytes).unwrap_err();
        assert!(matches!(err, WasmPrunerError::Module(_)), "got: {err}");
    }

    #[test]
    fn test_always_accept() {
        let pruner = WasmPruner::new(&parse(ALWAYS_ACCEPT_WAT)).unwrap();
        assert!(pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(5, 999, &[1, 2, 3]));
        assert!(pruner.is_valid(100, u32::MAX, &[u32::MAX; 200]));
    }

    #[test]
    fn test_always_reject() {
        let pruner = WasmPruner::new(&parse(ALWAYS_REJECT_WAT)).unwrap();
        assert!(!pruner.is_valid(0, 0, &[]));
        assert!(!pruner.is_valid(5, 999, &[1, 2, 3]));
    }

    #[test]
    fn test_nonzero_token() {
        let pruner = WasmPruner::new(&parse(NONZERO_WAT)).unwrap();
        assert!(!pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(0, 42, &[]));
        // 0xFFFFFFFF reinterpreted as i32 = -1, still nonzero
        assert!(pruner.is_valid(0, u32::MAX, &[]));
    }

    #[test]
    fn test_uses_parent_len() {
        let pruner = WasmPruner::new(&parse(HAS_PARENTS_WAT)).unwrap();
        assert!(!pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(0, 1, &[5]));
        assert!(pruner.is_valid(0, 1, &[5, 6, 7]));
    }

    #[test]
    fn test_uses_parent_memory() {
        let pruner = WasmPruner::new(&parse(FIRST_PARENT_FIVE_WAT)).unwrap();
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(0, 1, &[5]));
        assert!(!pruner.is_valid(0, 1, &[6]));
        assert!(pruner.is_valid(0, 1, &[5, 9, 9]));
    }

    #[test]
    fn test_uses_depth_param() {
        let pruner = WasmPruner::new(&parse(EVEN_DEPTH_WAT)).unwrap();
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(!pruner.is_valid(1, 1, &[]));
        assert!(pruner.is_valid(2, 1, &[]));
        assert!(!pruner.is_valid(3, 1, &[]));
    }

    #[test]
    fn test_fuel_limits_runaway_loop() {
        // Use a small fuel so the trap is quick.
        let pruner = WasmPruner::with_fuel(&parse(INFINITE_LOOP_WAT), 1000).unwrap();
        // The infinite loop should trap on fuel exhaustion → false (not hang).
        let result = pruner.is_valid(0, 1, &[]);
        assert!(!result, "fuel exhaustion should report as rejection");
    }

    #[test]
    fn test_fuel_per_call_isolated_between_calls() {
        // A pruner that consumes fuel each call. With budget 100k, many
        // calls should each succeed independently — fuel is reset per call.
        let wat = r#"
            (module
              (memory (export "memory") 1)
              (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
                (local $i i32)
                i32.const 0
                local.set $i
                block
                  loop
                    local.get $i
                    i32.const 100
                    i32.ge_s
                    br_if 1
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br 0
                  end
                end
                i32.const 1))
        "#;
        let pruner = WasmPruner::with_fuel(&parse(wat), 100_000).unwrap();
        for _ in 0..50 {
            assert!(pruner.is_valid(0, 1, &[]), "each call should succeed");
        }
    }

    #[test]
    fn test_batch_is_valid_amortizes() {
        let pruner = WasmPruner::new(&parse(NONZERO_WAT)).unwrap();
        let candidates = vec![0, 1, 2, 0, 3, 0, 5];
        let mut results = vec![false; candidates.len()];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        assert_eq!(results, vec![false, true, true, false, true, false, true]);
    }

    #[test]
    fn test_batch_is_valid_handles_shorter_results() {
        let pruner = WasmPruner::new(&parse(NONZERO_WAT)).unwrap();
        let candidates = vec![1, 2, 3, 4, 5];
        let mut results = vec![false; 3];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        assert_eq!(results, vec![true, true, true]);
    }

    #[test]
    fn test_batch_is_valid_uses_parent_tokens() {
        let pruner = WasmPruner::new(&parse(HAS_PARENTS_WAT)).unwrap();
        let candidates = vec![1, 2, 3];
        let mut results = vec![false; 3];
        // No parents → all rejected
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        assert_eq!(results, vec![false, false, false]);
        // With parents → all accepted
        pruner.batch_is_valid(0, &candidates, &[7], &mut results);
        assert_eq!(results, vec![true, true, true]);
    }

    #[test]
    fn test_propagate_and_on_backtrack_are_noop() {
        let mut pruner = WasmPruner::new(&parse(ALWAYS_ACCEPT_WAT)).unwrap();
        pruner.propagate(0, 1, &[]);
        pruner.on_backtrack(0, 1, &[]);
        assert!(pruner.is_valid(1, 5, &[1]));
    }

    #[test]
    fn test_manifold_score_default_binary() {
        let pruner = WasmPruner::new(&parse(NONZERO_WAT)).unwrap();
        let valid = pruner.manifold_score(0, 5, &[]);
        let invalid = pruner.manifold_score(0, 0, &[]);
        assert!((valid - 1.0).abs() < 1e-6);
        assert!((invalid - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_fuel_per_call_returns_previous() {
        let mut pruner = WasmPruner::new(&parse(ALWAYS_ACCEPT_WAT)).unwrap();
        let prev = pruner.set_fuel_per_call(5000);
        assert_eq!(prev, DEFAULT_FUEL_PER_CALL);
        let prev2 = pruner.set_fuel_per_call(2000);
        assert_eq!(prev2, 5000);
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WasmPruner>();
    }

    #[test]
    fn test_trait_object_works() {
        let pruner: Box<dyn ConstraintPruner> =
            Box::new(WasmPruner::new(&parse(ALWAYS_ACCEPT_WAT)).unwrap());
        assert!(pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_large_parent_tokens_grows_memory() {
        let pruner = WasmPruner::new(&parse(HAS_PARENTS_WAT)).unwrap();
        // Default memory is 64 KiB = 16384 i32 tokens. Push past it.
        let big: Vec<TokenId> = (0..20_000).collect();
        assert!(pruner.is_valid(0, 1, &big));
    }

    #[test]
    fn test_memory_growth_cap_rejects_oversized() {
        let pruner = WasmPruner::new(&parse(HAS_PARENTS_WAT)).unwrap();
        // MAX_PAGES = 256 → 16 MiB → ~4M tokens. Push past it.
        let huge: Vec<TokenId> = vec![0; 5_000_000];
        // Should not panic; should report false (rejected).
        assert!(!pruner.is_valid(0, 1, &huge));
    }

    #[test]
    fn test_error_display_messages() {
        let e = WasmPrunerError::Module("compilation error".to_string());
        let s = format!("{e}");
        assert!(s.contains("compilation failed"));
        assert!(s.contains("compilation error"));

        let e = WasmPrunerError::MissingMemory;
        assert!(format!("{e}").contains("memory"));

        let e = WasmPrunerError::MissingIsValid;
        assert!(format!("{e}").contains("is_valid"));

        let e = WasmPrunerError::Instantiation("trap".to_string());
        assert!(format!("{e}").contains("instantiation"));
    }
}
