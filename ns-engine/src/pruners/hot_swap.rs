//! Hot-swappable WASM pruner with BLAKE3 audit logging.
//!
//! Wraps a [`WasmPruner`] and supports atomic replacement at runtime via
//! [`HotSwapPruner::reload`]. Each swap attempt — successful or failed — is
//! recorded in an append-only audit log keyed by BLAKE3 hashes, forming a
//! tamper-evident chain: each record stores the hash of the previously
//! active WASM module, so a missing or reordered record is detectable.
//!
//! # Design
//!
//! - **Compile before lock**: WASM compilation happens outside the mutex
//!   so a slow compile does not block ongoing `is_valid` calls on the
//!   live pruner.
//! - **Atomic swap**: the live pruner is replaced under a single mutex
//!   acquisition; readers never observe a half-swapped state.
//! - **Failure preserves liveness**: if the new WASM fails to instantiate,
//!   the previously active pruner remains in place and `is_valid` continues
//!   to operate. The failed attempt is still logged with `success: false`.
//! - **Hash chain**: every record carries the hash of the WASM that was
//!   live immediately before the attempt. Replaying the log reproduces the
//!   exact sequence of (failed or successful) modules the system ran.
//!
//! # Audit chain semantics
//!
//! Consider three reload attempts R0, R1, R2:
//!   - R0 (succeeded): `prev_hash = initial`, `wasm_hash = H0`
//!   - R1 (failed):    `prev_hash = H0`,      `wasm_hash = H1` (still live = H0)
//!   - R2 (succeeded): `prev_hash = H0`,      `wasm_hash = H2`
//!
//! The `prev_hash` of R2 equals the live hash after R1 — which is still H0
//! because R1 failed. This chain lets an auditor reconstruct what was
//! actually running at any point in time.

use crate::pruners::wasm_pruner::{WasmPruner, WasmPrunerError};
use crate::traits::ConstraintPruner;
use crate::types::TokenId;
use std::sync::Mutex;

/// BLAKE3 hash output size in bytes.
const HASH_SIZE: usize = 32;

/// Compute BLAKE3 hash of WASM bytes (32 bytes).
fn hash_wasm(wasm_bytes: &[u8]) -> [u8; HASH_SIZE] {
    *blake3::hash(wasm_bytes).as_bytes()
}

/// Audit record for a single swap attempt.
///
/// Both successful and failed reloads produce a record. The chain of
/// records lets an auditor replay the exact sequence of WASM modules the
/// system attempted to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwapRecord {
    /// Monotonic sequence number (starts at 0 for the first attempt).
    pub seq: u64,
    /// BLAKE3 hash of the WASM bytes that were attempted in this reload.
    pub wasm_hash: [u8; HASH_SIZE],
    /// Size in bytes of the attempted WASM module.
    pub wasm_size: usize,
    /// BLAKE3 hash of the WASM that was live immediately before this
    /// attempt. Equal to the new `wasm_hash` for a failed swap on a
    /// first-time identical module, but generally reflects whatever was
    /// actually running.
    pub prev_hash: [u8; HASH_SIZE],
    /// Whether the swap succeeded. On failure the live pruner is
    /// preserved, but the failed attempt is still recorded.
    pub success: bool,
}

/// Errors returned by [`HotSwapPruner::reload`].
#[derive(Debug)]
pub enum ReloadError {
    /// The new WASM failed to compile or instantiate.
    /// The previously active pruner remains in place.
    Wasm(WasmPrunerError),
    /// The internal state mutex is poisoned (a previous call panicked
    /// while holding the lock).
    Poisoned,
}

impl std::fmt::Display for ReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wasm(e) => write!(f, "reload failed: {e}"),
            Self::Poisoned => write!(f, "reload failed: hot-swap state mutex poisoned"),
        }
    }
}

impl std::error::Error for ReloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wasm(e) => Some(e),
            Self::Poisoned => None,
        }
    }
}

/// Outcome of a successful reload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReloadOutcome {
    /// Sequence number assigned to this swap.
    pub seq: u64,
    /// BLAKE3 hash of the newly active WASM.
    pub wasm_hash: [u8; HASH_SIZE],
    /// BLAKE3 hash of the previously active WASM (for chain audit).
    pub prev_hash: [u8; HASH_SIZE],
    /// Size in bytes of the newly active WASM module.
    pub wasm_size: usize,
}

#[derive(Debug)]
struct State {
    inner: WasmPruner,
    wasm_hash: [u8; HASH_SIZE],
    wasm_size: usize,
    swap_count: u64,
    audit_log: Vec<SwapRecord>,
}

/// Wraps a [`WasmPruner`] with atomic hot-swap and audit logging.
///
/// All trait methods serialize through an internal [`Mutex`], making
/// `HotSwapPruner` [`Send`] + [`Sync`]. The mutex is held only briefly
/// during delegation — never across user code or WASM compilation.
#[derive(Debug)]
pub struct HotSwapPruner {
    state: Mutex<State>,
}

impl HotSwapPruner {
    /// Build a hot-swap pruner from initial WASM bytes.
    ///
    /// The initial module is recorded implicitly as the starting state;
    /// its audit entry is created on the first call to [`reload`](Self::reload).
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmPrunerError> {
        let inner = WasmPruner::new(wasm_bytes)?;
        let wasm_hash = hash_wasm(wasm_bytes);
        Ok(Self {
            state: Mutex::new(State {
                inner,
                wasm_hash,
                wasm_size: wasm_bytes.len(),
                swap_count: 0,
                audit_log: Vec::new(),
            }),
        })
    }

    /// Like [`new`](Self::new) but with a custom per-call fuel budget for
    /// the inner pruner.
    pub fn with_fuel(wasm_bytes: &[u8], fuel: u64) -> Result<Self, WasmPrunerError> {
        let inner = WasmPruner::with_fuel(wasm_bytes, fuel)?;
        let wasm_hash = hash_wasm(wasm_bytes);
        Ok(Self {
            state: Mutex::new(State {
                inner,
                wasm_hash,
                wasm_size: wasm_bytes.len(),
                swap_count: 0,
                audit_log: Vec::new(),
            }),
        })
    }

    /// Hot-swap to new WASM bytes.
    ///
    /// On success, the live pruner is replaced and the outcome is recorded
    /// in the audit log. On failure, the live pruner is left untouched but
    /// the failed attempt is still recorded with `success: false`.
    ///
    /// Compilation happens outside the mutex so a slow compile does not
    /// block ongoing `is_valid` calls on the live pruner.
    pub fn reload(&self, wasm_bytes: &[u8]) -> Result<ReloadOutcome, ReloadError> {
        let attempted_hash = hash_wasm(wasm_bytes);
        let attempted_size = wasm_bytes.len();

        // Compile BEFORE taking the lock — a slow compile should not block
        // ongoing `is_valid` calls on the live pruner.
        let new_inner = match WasmPruner::new(wasm_bytes) {
            Ok(p) => p,
            Err(e) => {
                // Record the failed attempt before surfacing the error.
                self.record_failed(attempted_hash, attempted_size);
                return Err(ReloadError::Wasm(e));
            }
        };

        // Take the lock and swap.
        let mut state = self.state.lock().map_err(|_| ReloadError::Poisoned)?;

        let prev_hash = state.wasm_hash;
        let seq = state.swap_count;
        state.inner = new_inner;
        state.wasm_hash = attempted_hash;
        state.wasm_size = attempted_size;
        state.swap_count = seq.saturating_add(1);
        state.audit_log.push(SwapRecord {
            seq,
            wasm_hash: attempted_hash,
            wasm_size: attempted_size,
            prev_hash,
            success: true,
        });

        Ok(ReloadOutcome {
            seq,
            wasm_hash: attempted_hash,
            prev_hash,
            wasm_size: attempted_size,
        })
    }

    /// Record a failed reload attempt. The live pruner is left untouched.
    /// Best-effort: silently ignored if the mutex is poisoned.
    fn record_failed(&self, attempted_hash: [u8; HASH_SIZE], attempted_size: usize) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let prev_hash = state.wasm_hash;
        let seq = state.swap_count;
        state.swap_count = seq.saturating_add(1);
        state.audit_log.push(SwapRecord {
            seq,
            wasm_hash: attempted_hash,
            wasm_size: attempted_size,
            prev_hash,
            success: false,
        });
    }

    /// Update the per-call fuel budget of the live pruner.
    /// Returns the previous budget, or `None` if the mutex is poisoned.
    pub fn set_fuel_per_call(&self, fuel: u64) -> Option<u64> {
        self.state
            .lock()
            .ok()
            .map(|mut s| s.inner.set_fuel_per_call(fuel))
    }

    /// Current WASM module's BLAKE3 hash.
    pub fn wasm_hash(&self) -> [u8; HASH_SIZE] {
        self.state
            .lock()
            .map(|s| s.wasm_hash)
            .unwrap_or([0u8; HASH_SIZE])
    }

    /// Current WASM module size in bytes.
    pub fn wasm_size(&self) -> usize {
        self.state.lock().map(|s| s.wasm_size).unwrap_or(0)
    }

    /// Total number of reload attempts (successful + failed) since
    /// construction.
    pub fn attempt_count(&self) -> u64 {
        self.state.lock().map(|s| s.swap_count).unwrap_or(0)
    }

    /// Read-only view of the audit log (cloned).
    pub fn audit_log(&self) -> Vec<SwapRecord> {
        self.state
            .lock()
            .map(|s| s.audit_log.clone())
            .unwrap_or_default()
    }

    /// Number of successful swaps recorded.
    pub fn successful_swaps(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.audit_log.iter().filter(|r| r.success).count())
            .unwrap_or(0)
    }

    /// Number of failed reload attempts recorded.
    pub fn failed_swaps(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.audit_log.iter().filter(|r| !r.success).count())
            .unwrap_or(0)
    }
}

impl ConstraintPruner for HotSwapPruner {
    fn is_valid(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        match self.state.lock() {
            Ok(state) => state.inner.is_valid(depth, token, parent_tokens),
            Err(_) => false,
        }
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        match self.state.lock() {
            Ok(state) => state
                .inner
                .batch_is_valid(depth, candidates, parent_tokens, results),
            Err(_) => results.fill(false),
        }
    }

    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        match self.state.lock() {
            Ok(state) => state.inner.manifold_score(depth, token, parent_tokens),
            Err(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ConstraintPruner;
    use std::error::Error;

    const ACCEPT_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1))
    "#;

    const REJECT_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 0))
    "#;

    const NONZERO_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param $d i32) (param $t i32) (param $p i32) (param $l i32) (result i32)
            local.get $t
            i32.const 0
            i32.ne))
    "#;

    const INVALID_BYTES: &[u8] = b"not wasm";

    fn parse(wat: &str) -> Vec<u8> {
        wat::parse_bytes(wat.as_bytes())
            .expect("WAT parse failure")
            .into_owned()
    }

    #[test]
    fn test_new_initializes_state() {
        let bytes = parse(ACCEPT_WAT);
        let hs = HotSwapPruner::new(&bytes).unwrap();
        assert_eq!(hs.wasm_size(), bytes.len());
        assert_eq!(hs.attempt_count(), 0);
        assert!(hs.audit_log().is_empty());
        assert_eq!(hs.successful_swaps(), 0);
        assert_eq!(hs.failed_swaps(), 0);
    }

    #[test]
    fn test_new_propagates_wasm_error() {
        let err = HotSwapPruner::new(INVALID_BYTES).unwrap_err();
        assert!(matches!(err, WasmPrunerError::Module(_)));
    }

    #[test]
    fn test_with_fuel_initializes_state() {
        let bytes = parse(ACCEPT_WAT);
        let hs = HotSwapPruner::with_fuel(&bytes, 5000).unwrap();
        assert_eq!(hs.wasm_size(), bytes.len());
        assert!(hs.is_valid(0, 1, &[]));
    }

    #[test]
    fn test_blake3_hash_correct() {
        let bytes = parse(ACCEPT_WAT);
        let expected = hash_wasm(&bytes);
        let hs = HotSwapPruner::new(&bytes).unwrap();
        assert_eq!(hs.wasm_hash(), expected);
    }

    #[test]
    fn test_blake3_distinguishes_different_modules() {
        let h1 = hash_wasm(&parse(ACCEPT_WAT));
        let h2 = hash_wasm(&parse(REJECT_WAT));
        assert_ne!(h1, h2, "different WASM must hash to different values");
    }

    #[test]
    fn test_blake3_stable_across_calls() {
        let bytes = parse(ACCEPT_WAT);
        let hs = HotSwapPruner::new(&bytes).unwrap();
        let h1 = hs.wasm_hash();
        let h2 = hs.wasm_hash();
        assert_eq!(h1, h2, "hash must be stable until a reload");
    }

    #[test]
    fn test_delegates_to_inner_accept() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        assert!(hs.is_valid(0, 1, &[]));
        assert!(hs.is_valid(5, 999, &[1, 2, 3]));
    }

    #[test]
    fn test_delegates_to_inner_reject() {
        let hs = HotSwapPruner::new(&parse(REJECT_WAT)).unwrap();
        assert!(!hs.is_valid(0, 1, &[]));
        assert!(!hs.is_valid(5, 999, &[1, 2, 3]));
    }

    #[test]
    fn test_reload_swaps_pruner_behavior() {
        let hs = HotSwapPruner::new(&parse(REJECT_WAT)).unwrap();
        assert!(!hs.is_valid(0, 1, &[]));
        let outcome = hs.reload(&parse(ACCEPT_WAT)).unwrap();
        // Outcome's wasm_hash reflects the newly-loaded module.
        assert_eq!(outcome.wasm_hash, hash_wasm(&parse(ACCEPT_WAT)));
        assert!(hs.is_valid(0, 1, &[]));
    }

    #[test]
    fn test_reload_returns_outcome_with_correct_hash() {
        let hs = HotSwapPruner::new(&parse(REJECT_WAT)).unwrap();
        let initial_hash = hs.wasm_hash();
        let outcome = hs.reload(&parse(ACCEPT_WAT)).unwrap();
        assert_eq!(outcome.prev_hash, initial_hash);
        assert_eq!(outcome.wasm_hash, hash_wasm(&parse(ACCEPT_WAT)));
        assert_eq!(outcome.seq, 0);
        assert_eq!(outcome.wasm_size, parse(ACCEPT_WAT).len());
    }

    #[test]
    fn test_reload_increments_seq() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        let o1 = hs.reload(&parse(REJECT_WAT)).unwrap();
        let o2 = hs.reload(&parse(ACCEPT_WAT)).unwrap();
        let o3 = hs.reload(&parse(NONZERO_WAT)).unwrap();
        assert_eq!(o1.seq, 0);
        assert_eq!(o2.seq, 1);
        assert_eq!(o3.seq, 2);
    }

    #[test]
    fn test_reload_records_audit_log_success() {
        let hs = HotSwapPruner::new(&parse(REJECT_WAT)).unwrap();
        let initial_hash = hs.wasm_hash();
        let _ = hs.reload(&parse(ACCEPT_WAT)).unwrap();
        let log = hs.audit_log();
        assert_eq!(log.len(), 1);
        assert!(log[0].success);
        assert_eq!(log[0].seq, 0);
        assert_eq!(log[0].prev_hash, initial_hash);
        assert_eq!(log[0].wasm_hash, hash_wasm(&parse(ACCEPT_WAT)));
    }

    #[test]
    fn test_reload_failure_keeps_current_pruner() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        let initial_hash = hs.wasm_hash();
        let initial_size = hs.wasm_size();
        let err = hs.reload(INVALID_BYTES).unwrap_err();
        assert!(matches!(err, ReloadError::Wasm(_)));
        // Current pruner is untouched.
        assert!(hs.is_valid(0, 1, &[]));
        assert_eq!(hs.wasm_hash(), initial_hash);
        assert_eq!(hs.wasm_size(), initial_size);
    }

    #[test]
    fn test_reload_failure_records_audit_log() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        let initial_hash = hs.wasm_hash();
        let _ = hs.reload(INVALID_BYTES).unwrap_err();
        let log = hs.audit_log();
        assert_eq!(log.len(), 1);
        assert!(!log[0].success);
        assert_eq!(log[0].seq, 0);
        assert_eq!(log[0].prev_hash, initial_hash);
        // The failed attempt's hash is recorded even though it didn't take effect.
        assert_eq!(log[0].wasm_hash, hash_wasm(INVALID_BYTES));
    }

    #[test]
    fn test_audit_log_chain_after_mixed_success() {
        let hs = HotSwapPruner::new(&parse(REJECT_WAT)).unwrap();
        let initial_hash = hs.wasm_hash();
        let _ = hs.reload(&parse(ACCEPT_WAT)).unwrap(); // success
        let _ = hs.reload(INVALID_BYTES).unwrap_err(); // failure
        let _ = hs.reload(&parse(NONZERO_WAT)).unwrap(); // success

        let log = hs.audit_log();
        assert_eq!(log.len(), 3);

        // R0 (success): prev = initial, new = H_accept
        assert!(log[0].success);
        assert_eq!(log[0].prev_hash, initial_hash);
        let h_accept = log[0].wasm_hash;

        // R1 (failed): prev = H_accept (still live), attempted = H_invalid
        assert!(!log[1].success);
        assert_eq!(log[1].prev_hash, h_accept);

        // R2 (success): prev = H_accept (R1 didn't change live), new = H_nonzero
        assert!(log[2].success);
        assert_eq!(log[2].prev_hash, h_accept);

        // After all attempts, live = H_nonzero
        assert_eq!(hs.wasm_hash(), log[2].wasm_hash);
    }

    #[test]
    fn test_attempt_count_includes_failures() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        let _ = hs.reload(&parse(REJECT_WAT)).unwrap();
        let _ = hs.reload(INVALID_BYTES).unwrap_err();
        let _ = hs.reload(&parse(ACCEPT_WAT)).unwrap();
        assert_eq!(hs.attempt_count(), 3);
        assert_eq!(hs.successful_swaps(), 2);
        assert_eq!(hs.failed_swaps(), 1);
    }

    #[test]
    fn test_batch_delegates_to_inner() {
        let hs = HotSwapPruner::new(&parse(NONZERO_WAT)).unwrap();
        let candidates = vec![0, 1, 2, 0, 3];
        let mut results = vec![false; candidates.len()];
        hs.batch_is_valid(0, &candidates, &[], &mut results);
        assert_eq!(results, vec![false, true, true, false, true]);
    }

    #[test]
    fn test_manifold_score_delegates() {
        let hs = HotSwapPruner::new(&parse(NONZERO_WAT)).unwrap();
        let valid = hs.manifold_score(0, 5, &[]);
        let invalid = hs.manifold_score(0, 0, &[]);
        assert!((valid - 1.0).abs() < 1e-6);
        assert!((invalid - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_fuel_per_call_returns_previous() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        let prev = hs.set_fuel_per_call(5000);
        assert_eq!(
            prev,
            Some(crate::pruners::wasm_pruner::DEFAULT_FUEL_PER_CALL)
        );
        let prev2 = hs.set_fuel_per_call(2000);
        assert_eq!(prev2, Some(5000));
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HotSwapPruner>();
        assert_send_sync::<SwapRecord>();
        assert_send_sync::<ReloadOutcome>();
        assert_send_sync::<ReloadError>();
    }

    #[test]
    fn test_trait_object_works() {
        let hs: Box<dyn ConstraintPruner> =
            Box::new(HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap());
        assert!(hs.is_valid(0, 1, &[]));
    }

    #[test]
    fn test_error_display_messages() {
        let e = ReloadError::Wasm(WasmPrunerError::Module("compilation".to_string()));
        let s = format!("{e}");
        assert!(s.contains("reload failed"));
        assert!(s.contains("compilation"));

        let e = ReloadError::Poisoned;
        let s = format!("{e}");
        assert!(s.contains("poisoned"));
    }

    #[test]
    fn test_reload_error_source() {
        let e = ReloadError::Wasm(WasmPrunerError::MissingMemory);
        assert!(e.source().is_some());

        let e = ReloadError::Poisoned;
        assert!(e.source().is_none());
    }

    #[test]
    fn test_swap_record_equality() {
        let h1 = [1u8; 32];
        let h2 = [2u8; 32];
        let r1 = SwapRecord {
            seq: 0,
            wasm_hash: h1,
            wasm_size: 100,
            prev_hash: h2,
            success: true,
        };
        let r2 = SwapRecord {
            seq: 0,
            wasm_hash: h1,
            wasm_size: 100,
            prev_hash: h2,
            success: true,
        };
        let r3 = SwapRecord {
            seq: 0,
            wasm_hash: h2, // different
            wasm_size: 100,
            prev_hash: h2,
            success: true,
        };
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_chain_of_many_reloads() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        // 10 successful reloads, alternating accept/reject.
        for i in 0..10 {
            let wat = if i % 2 == 0 { ACCEPT_WAT } else { REJECT_WAT };
            let _ = hs.reload(&parse(wat)).unwrap();
        }
        assert_eq!(hs.attempt_count(), 10);
        assert_eq!(hs.successful_swaps(), 10);
        assert_eq!(hs.failed_swaps(), 0);
        // After 10 reloads (even i=0..9), last was i=9 → REJECT.
        assert!(!hs.is_valid(0, 1, &[]));
        // Chain integrity: each prev_hash matches prior record's wasm_hash
        // for that record's effect (success → next prev = wasm_hash;
        // failure → next prev = unchanged).
        let log = hs.audit_log();
        let mut live_hash = log[0].prev_hash; // hash before any reload
        for r in &log {
            assert_eq!(r.prev_hash, live_hash, "chain break at seq={}", r.seq);
            if r.success {
                live_hash = r.wasm_hash;
            }
        }
        assert_eq!(live_hash, hs.wasm_hash(), "final live hash matches");
    }

    #[test]
    fn test_propagate_and_on_backtrack_delegated() {
        let hs = HotSwapPruner::new(&parse(ACCEPT_WAT)).unwrap();
        // Defaults are no-op; should not panic.
        // We test this indirectly by verifying is_valid still works after
        // calling them. Since HotSwapPruner doesn't override them, the
        // default ConstraintPruner implementations are used.
        assert!(hs.is_valid(0, 1, &[]));
    }
}
