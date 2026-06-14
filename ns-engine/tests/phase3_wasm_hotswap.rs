//! Phase 3 end-to-end proof: WASM pruner compile → load → hot-swap → regression.
//!
//! This file closes out Phase 3 of plan 006. It exercises three things the
//! existing unit tests in `pruners/wasm_pruner.rs` and `pruners/hot_swap.rs`
//! do NOT cover:
//!
//! 1. **Compile a real Rust pruner source to `wasm32-unknown-unknown`** and
//!    feed the bytes to `HotSwapPruner`. The deliverable source is
//!    `wasm-examples/regex_pruner.rs`; this test compiles it via `rustc` at
//!    test time and asserts the resulting WASM loads and enforces its
//!    `a*b*c*` constraint correctly.
//!
//! 2. **Hot-swap a pruner mid-decode** between drafts and verify the new
//!    pruner takes effect on the *next* draft (not the in-flight one).
//!    A sequential variant proves the cross-draft transition; a concurrent
//!    variant proves the atomic-swap + compile-outside-lock design does not
//!    deadlock or corrupt an in-flight decode.
//!
//! 3. **Regression suite runs after hot-swap** — a battery of invariant
//!    checks inspired by katgpt-rs's hot-load infrastructure, asserting the
//!    swapped pruner leaves the engine's contracts intact.
//!
//! ## Environment guard
//!
//! These tests compile Rust → `wasm32-unknown-unknown` via `rustc`. They are
//! skipped (with a stderr note) if that target is not installed, rather than
//! failing. On a machine with the target installed (the documented Phase 3
//! toolchain) they run in full.

#![cfg(feature = "wasm-pruner")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ns_engine::{
    draft::UniformDraftModel, speculative_decode, ConstraintPruner, DecodeConfig, DraftModel,
    HotSwapPruner, TokenId, WasmPruner,
};

// ── Fixtures ───────────────────────────────────────────────────

/// Target triple the example pruner is compiled for.
const WASM_TARGET: &str = "wasm32-unknown-unknown";

/// A second, real Rust → WASM pruner compiled at test time. Accepts ONLY
/// token `7`, regardless of context. This is deliberately disjoint from the
/// `regex_pruner.rs` alphabet `{0,1,2}` so a swap between them produces an
/// observable, unambiguous change in decode behavior.
const PRUNER_B_SOURCE: &str = r#"
#![no_std]

/// Accept only token 7. Ignores parents (stateless screening).
#[no_mangle]
pub extern "C" fn is_valid(
    _depth: i32,
    token: i32,
    _parent_ptr: i32,
    _parent_len: i32,
) -> i32 {
    match token == 7 {
        true => 1,
        false => 0,
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
"#;

/// Resolve the deliverable example pruner source path inside the crate.
fn regex_pruner_source_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wasm-examples")
        .join("regex_pruner.rs")
}

// ── Toolchain guards + compile helpers ─────────────────────────

/// True iff the `wasm32-unknown-unknown` target is installed via `rustup`.
/// Tests that need real Rust → WASM compilation skip gracefully when this
/// returns false.
fn wasm32_target_installed() -> bool {
    let output = match Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return false,
    };
    output.status.success() && String::from_utf8_lossy(&output.stdout).contains(WASM_TARGET)
}

/// Skip helper for `Result`-returning tests. Returns `Ok(())` (no-op skip)
/// when the target is missing; otherwise returns `Ok(())`-shaped sentinel
/// the caller ignores. We model it as an early-return macro-free helper
/// returning a bool so the caller can `if !continue_if_target() { return Ok(()); }`.
fn skip_if_no_wasm_target() -> bool {
    if wasm32_target_installed() {
        true
    } else {
        eprintln!("skipping: {WASM_TARGET} target not installed (rustup target add {WASM_TARGET})");
        false
    }
}

/// Compile a Rust source FILE to `wasm32-unknown-unknown` cdylib bytes.
///
/// Uses `rustc` directly (not `cargo`) to avoid recursive cargo invocations
/// and to keep the compile self-contained. Warnings are treated as errors
/// (`-D warnings`) so a degrading example source fails the test, not silently
/// ships a broken module.
fn compile_wasm_file(src: &Path, tag: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let out = std::env::temp_dir().join(format!("ns_engine_phase3_{tag}.wasm"));
    let out_str = out
        .to_str()
        .ok_or_else(|| format!("non-utf8 temp path for tag {tag}"))?;

    let output = Command::new("rustc")
        .args([
            "--target",
            WASM_TARGET,
            "--crate-type",
            "cdylib",
            "-O",
            "-D",
            "warnings",
        ])
        .arg(src)
        .args(["-o", out_str])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "rustc {WASM_TARGET} compile failed for {tag}\n--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}"
        )
        .into());
    }

    let bytes = fs::read(&out)?;
    Ok(bytes)
}

/// Compile a Rust source STRING to `wasm32-unknown-unknown` cdylib bytes.
/// Writes the source to a temp `.rs` first, then delegates to
/// [`compile_wasm_file`].
fn compile_wasm_source(source: &str, tag: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let src = std::env::temp_dir().join(format!("ns_engine_phase3_{tag}.rs"));
    fs::write(&src, source)?;
    let bytes = compile_wasm_file(&src, tag)?;
    // Best-effort cleanup; ignore failure.
    let _ = fs::remove_file(&src);
    Ok(bytes)
}

/// BLAKE3 of `bytes` — used to assert the swap took effect via hash equality.
fn blake3_of(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Build the two real compiled pruner payloads used across these tests.
/// Returns `(regex_bytes, pruner_b_bytes)`.
fn build_pruner_payloads() -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let regex_bytes = compile_wasm_file(&regex_pruner_source_path(), "regex")?;
    let pruner_b_bytes = compile_wasm_source(PRUNER_B_SOURCE, "only7")?;
    Ok((regex_bytes, pruner_b_bytes))
}

// ── Shared decode helpers ──────────────────────────────────────

/// Greedy decode using ONLY `&self` pruner methods (`batch_is_valid`).
///
/// This mirrors `speculative_decode`'s greedy path but never calls
/// `propagate`/`on_backtrack` (which require `&mut self`). That makes it
/// safe to run against a `&HotSwapPruner` shared across threads — exactly
/// what the concurrent mid-decode test needs to exercise the atomic-swap
/// design under live decode traffic.
///
/// `HotSwapPruner`'s `propagate`/`on_backtrack` are default no-ops, so
/// skipping them is equivalent for this pruner family.
fn greedy_decode_shared(
    pruner: &HotSwapPruner,
    draft: &dyn DraftModel,
    max_tokens: usize,
) -> Vec<TokenId> {
    let vocab = draft.vocab_size();
    // Uniform-ish candidate set: all vocab tokens in index order. Matches
    // `top_k_indices` on a uniform logit distribution with `top_k == vocab`.
    let candidates: Vec<TokenId> = (0..vocab as TokenId).collect();
    let mut tokens: Vec<TokenId> = Vec::with_capacity(max_tokens);
    while tokens.len() < max_tokens {
        let depth = tokens.len();
        let mut mask = vec![false; candidates.len()];
        pruner.batch_is_valid(depth, &candidates, &tokens, &mut mask);
        match candidates
            .iter()
            .zip(mask.iter())
            .find(|(_, &valid)| valid)
            .map(|(&t, _)| t)
        {
            Some(token) => tokens.push(token),
            None => break,
        }
    }
    tokens
}

/// Small greedy-only decode config for the Phase 3 fixtures.
fn greedy_config(max_tokens: usize) -> DecodeConfig {
    DecodeConfig {
        max_tokens,
        top_k: 10,
        seed: 42,
        backtrack: false,
        max_attempts: 0,
    }
}

// ═══════════════════════════════════════════════════════════════
// CHECKBOX 1 — Compile example pruner to wasm32-unknown-unknown
// ═══════════════════════════════════════════════════════════════

/// The deliverable Rust source (`wasm-examples/regex_pruner.rs`) compiles to
/// `wasm32-unknown-unknown` via `rustc`, the resulting WASM exports `memory`
/// and `is_valid`, and both `WasmPruner` and `HotSwapPruner` load it.
///
/// The pruner enforces the regular language `a*b*c*` (alphabet `{0,1,2}`,
/// non-decreasing). We verify the constraint logic directly against the
/// loaded module — this proves the full Rust → WASM → host path end-to-end.
#[test]
fn checkbox1_compile_example_pruner_to_wasm32_and_loads() -> Result<(), Box<dyn Error>> {
    if !skip_if_no_wasm_target() {
        return Ok(());
    }

    let bytes = compile_wasm_file(&regex_pruner_source_path(), "regex_checkbox1")?;

    // Non-trivial WASM module, BLAKE3-stable identity.
    let hash = blake3_of(&bytes);
    assert_ne!(hash, [0u8; 32], "compiled wasm must hash to non-zero");
    assert!(
        bytes.len() > 64,
        "compiled wasm suspiciously small: {len} bytes",
        len = bytes.len()
    );

    // Loads as a bare WasmPruner. Instantiation succeeding already proves the
    // module exports `memory` and `is_valid` (those are validated inside
    // `WasmPruner::with_fuel`); exercising the module confirms the ABI works.
    let direct = WasmPruner::new(&bytes)?;
    assert!(
        direct.is_valid(0, 0, &[]),
        "bare WasmPruner serves is_valid"
    );

    // Loads wrapped in HotSwapPruner (the swap-capable surface).
    let pruner = HotSwapPruner::new(&bytes)?;
    assert_eq!(pruner.wasm_size(), bytes.len());
    assert_eq!(pruner.wasm_hash(), hash);
    assert_eq!(pruner.attempt_count(), 0);

    // ── a*b*c* constraint logic (read by the host through wasmi) ──
    // Alphabet membership at depth 0.
    assert!(pruner.is_valid(0, 0, &[]), "a (0) at depth 0");
    assert!(pruner.is_valid(0, 1, &[]), "b (1) at depth 0");
    assert!(pruner.is_valid(0, 2, &[]), "c (2) at depth 0");
    assert!(!pruner.is_valid(0, 3, &[]), "3 is out of alphabet");
    assert!(!pruner.is_valid(0, 7, &[]), "7 is out of alphabet");

    // Non-decreasing rule vs last committed parent.
    assert!(
        !pruner.is_valid(1, 0, &[1]),
        "b then a is decreasing → reject"
    );
    assert!(pruner.is_valid(1, 1, &[1]), "b then b is ok");
    assert!(pruner.is_valid(1, 2, &[1]), "b then c is ok");
    assert!(
        !pruner.is_valid(2, 0, &[1, 2]),
        "c then a is decreasing → reject"
    );
    assert!(pruner.is_valid(2, 2, &[1, 2]), "c then c is ok");

    // Batch path agrees with per-item path.
    let candidates = vec![0u32, 1, 2, 3, 7];
    let parents = vec![1u32];
    let mut batch = vec![false; candidates.len()];
    pruner.batch_is_valid(1, &candidates, &parents, &mut batch);
    // parents=[b], candidates 0..3,7 → only b(1) and c(2) are >= b
    assert_eq!(
        batch,
        vec![false, true, true, false, false],
        "batch must match per-item a*b*c* decisions"
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// CHECKBOX 2 — Hot-swap pruner mid-decode
// ═══════════════════════════════════════════════════════════════

/// **Sequential, between drafts.** Draft 1 runs under pruner A (`a*b*c*`);
/// between drafts we `reload` to pruner B (accept-only-7); draft 2 must
/// reflect B. The in-flight draft 1 is NOT retroactively disturbed — it
/// completed under A, and its tokens still respect A's constraint.
///
/// This is the cleanest "next draft takes effect, not the in-flight one"
/// proof: draft 1's tokens are all in `{0,1,2}` (A's alphabet, no 7),
/// draft 2's tokens are all `7` (B's sole accepted token).
#[test]
fn checkbox2_mid_decode_swap_between_drafts() -> Result<(), Box<dyn Error>> {
    if !skip_if_no_wasm_target() {
        return Ok(());
    }

    let (regex_bytes, only7_bytes) = build_pruner_payloads()?;
    let regex_hash = blake3_of(&regex_bytes);
    let only7_hash = blake3_of(&only7_bytes);
    assert_ne!(regex_hash, only7_hash, "fixtures must differ");

    let draft = UniformDraftModel::new(10);
    let config = greedy_config(5);
    let mut pruner = HotSwapPruner::new(&regex_bytes)?;

    // ── Pre-swap: pruner A (regex a*b*c*) is live ──
    assert_eq!(pruner.wasm_hash(), regex_hash);
    assert!(
        !pruner.is_valid(0, 7, &[]),
        "regex pruner must reject token 7"
    );

    let draft1 = speculative_decode(&draft, &mut pruner, &config);
    assert!(
        draft1.verified,
        "draft 1 (regex) must verify; got hash={h:?}",
        h = draft1.hash
    );
    assert!(
        draft1.tokens.iter().all(|&t| t < 3),
        "draft 1 tokens must be inside regex alphabet {{0,1,2}}; got {t:?}",
        t = draft1.tokens
    );
    assert!(
        !draft1.tokens.contains(&7),
        "draft 1 must contain no token 7 (regex rejects it)"
    );
    let draft1_snapshot = draft1.tokens.clone();

    // ── Hot-swap to pruner B (accept-only-7), between drafts ──
    let outcome = pruner.reload(&only7_bytes)?;
    assert_eq!(outcome.prev_hash, regex_hash, "reload prev_hash must be A");
    assert_eq!(outcome.wasm_hash, only7_hash, "reload new hash must be B");
    assert_eq!(pruner.wasm_hash(), only7_hash, "live pruner is now B");
    assert_eq!(pruner.successful_swaps(), 1, "one successful swap recorded");

    // Swap took effect immediately for subsequent is_valid calls.
    assert!(
        pruner.is_valid(0, 7, &[]),
        "after reload to B, token 7 must be valid"
    );
    assert!(
        !pruner.is_valid(0, 0, &[]),
        "after reload to B, token 0 must be rejected"
    );

    // ── Draft 1 was NOT retroactively disturbed by the swap ──
    assert_eq!(
        draft1_snapshot, draft1.tokens,
        "draft 1's committed tokens are immutable post-swap"
    );

    // ── Post-swap: draft 2 must reflect B ──
    let draft2 = speculative_decode(&draft, &mut pruner, &config);
    assert!(
        draft2.verified,
        "draft 2 (only-7) must verify; got hash={h:?}",
        h = draft2.hash
    );
    assert!(
        draft2.tokens.iter().all(|&t| t == 7),
        "draft 2 tokens must all be 7 under pruner B; got {t:?}",
        t = draft2.tokens
    );
    assert_ne!(
        draft1.tokens, draft2.tokens,
        "drafts must differ — the swap changed behavior"
    );

    Ok(())
}

/// **Concurrent, in-flight.** A decode thread runs many `batch_is_valid` calls
/// against an `Arc<HotSwapPruner>` while another thread `reload`s it. This
/// validates the two non-obvious design properties of `HotSwapPruner`:
///
/// 1. **Compile-outside-lock**: WASM compilation happens before the mutex is
///    taken, so the decode thread's `is_valid` calls are never blocked by a
///    slow compile — and never observe a half-swapped state.
/// 2. **Atomic swap under live traffic**: the mutex-guarded swap is safe
///    under concurrent readers; no deadlock, no panic, no torn reads.
///
/// Determinism: a `started` flag guarantees the reload fires only after the
/// decode thread has begun stepping, so the overlap is real without relying
/// on wall-clock timing.
#[test]
fn checkbox2_mid_decode_swap_concurrent_in_flight() -> Result<(), Box<dyn Error>> {
    if !skip_if_no_wasm_target() {
        return Ok(());
    }

    let (regex_bytes, only7_bytes) = build_pruner_payloads()?;
    let only7_hash = blake3_of(&only7_bytes);

    let pruner = Arc::new(HotSwapPruner::new(&regex_bytes)?);
    let draft = UniformDraftModel::new(10);

    let started = Arc::new(AtomicBool::new(false));

    // Decode thread: long enough that reload is guaranteed to land mid-flight.
    let decode_pruner = Arc::clone(&pruner);
    let decode_started = Arc::clone(&started);
    // Clone the draft model into the thread so the original remains usable on
    // this thread for the post-swap decode check.
    let decode_draft = draft.clone();
    let decode_handle = thread::spawn(move || -> (Vec<TokenId>, usize) {
        let mut steps = 0usize;
        // Step through many short decodes; each step calls batch_is_valid,
        // exercising the shared mutex against the reload thread.
        let mut all_tokens: Vec<TokenId> = Vec::new();
        for _ in 0..2000 {
            let tokens = greedy_decode_shared(&decode_pruner, &decode_draft, 4);
            steps += 1;
            all_tokens = tokens;
            if !decode_started.load(Ordering::SeqCst) {
                decode_started.store(true, Ordering::SeqCst);
            }
        }
        (all_tokens, steps)
    });

    // Wait until the decode thread is provably mid-flight, then swap.
    while !started.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_micros(50));
    }
    // Tiny extra delay so the decode thread is solidly mid-loop, not at the
    // boundary of the flag store.
    thread::sleep(Duration::from_millis(2));

    let outcome = pruner.reload(&only7_bytes)?;
    assert_eq!(outcome.wasm_hash, only7_hash);

    // Both threads must complete without deadlock or panic.
    let (final_tokens, steps) = decode_handle.join().map_err(|_| "decode thread panicked")?;
    assert!(steps > 0, "decode thread must have executed steps");

    // After join, the live pruner is B (accept-only-7) for any new decode.
    assert_eq!(
        pruner.wasm_hash(),
        only7_hash,
        "live pruner is B after swap"
    );
    assert!(
        pruner.is_valid(0, 7, &[]),
        "post-swap is_valid must reflect B"
    );

    // A fresh decode under the swapped pruner must produce only 7s.
    let post = greedy_decode_shared(&pruner, &draft, 4);
    assert!(
        post.iter().all(|&t| t == 7),
        "post-swap decode must be all 7s; got {post:?}"
    );

    // The in-flight concurrent decode must not have panicked/corrupted:
    // whatever tokens it returned are valid for SOME pruner in {A, B}, i.e.,
    // either in {0,1,2} (A) or equal to 7 (B). Mixed is allowed because the
    // swap landed somewhere mid-sequence.
    let plausible = final_tokens.iter().all(|&t| t < 3 || t == 7);
    assert!(
        plausible,
        "in-flight tokens must be valid under A or B; got {final_tokens:?}"
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// CHECKBOX 3 — Regression suite runs after hot-swap
// ═══════════════════════════════════════════════════════════════

/// Regression battery inspired by katgpt-rs hot-load infrastructure: after a
/// successful hot-swap, re-validate every engine invariant that could have
/// been disturbed by replacing the live pruner.
///
/// Returns `(passed, total)` so the caller can assert `passed == total`.
/// Each sub-check records its name so a failure points at the broken
/// invariant rather than a bare assertion line.
fn regression_suite_after_swap(
    pruner: &mut HotSwapPruner,
    draft: &dyn DraftModel,
    initial_hash: [u8; 32],
    expected_hash: [u8; 32],
    expected_size: usize,
) -> (usize, usize) {
    // Evaluate each invariant into a named bool first, then assemble the
    // result vector in one literal. This keeps the `&mut` speculative_decode
    // borrow separate from the shared `&self` reads (no interleaved borrows
    // inside a `vec![]` macro) and satisfies clippy's vec-init-then-push lint.

    // 1. Live identity matches the swapped-in WASM.
    let hash_ok = pruner.wasm_hash() == expected_hash;
    let size_ok = pruner.wasm_size() == expected_size;

    // 2. is_valid matches pruner B's (accept-only-7) policy exactly.
    let valid_7_at_root = pruner.is_valid(0, 7, &[]);
    let valid_0_at_root = !pruner.is_valid(0, 0, &[]);
    let valid_7_ignores_parents = pruner.is_valid(3, 7, &[1, 2, 3]);
    let rejects_non7_with_parents = !pruner.is_valid(3, 8, &[7, 7]);

    // 3. batch_is_valid is consistent with per-item is_valid.
    let candidates = vec![0u32, 7, 2, 7, 9];
    let parents = vec![7u32, 7];
    let mut batch = vec![false; candidates.len()];
    pruner.batch_is_valid(2, &candidates, &parents, &mut batch);
    let per_item: Vec<bool> = candidates
        .iter()
        .map(|&t| pruner.is_valid(2, t, &parents))
        .collect();
    let batch_consistent = batch == per_item && batch == vec![false, true, false, true, false];

    // 4. manifold_score is the default binary (1.0 valid / 0.0 invalid).
    let score_valid = pruner.manifold_score(1, 7, &[7]);
    let score_invalid = pruner.manifold_score(1, 0, &[7]);
    let score_valid_ok = (score_valid - 1.0).abs() < 1e-6;
    let score_invalid_ok = (score_invalid - 0.0).abs() < 1e-6;

    // 5. Full decode under the swapped pruner produces a verified, B-consistent
    //    sequence. Uses the REAL `speculative_decode` path (it requires
    //    `&mut dyn ConstraintPruner`; the auto-reborrow keeps `pruner` usable
    //    afterwards). This exercises the same loop production code uses.
    let config = greedy_config(5);
    let result = speculative_decode(draft, pruner, &config);
    let decode_verified = result.verified && result.tokens.len() == config.max_tokens;
    let decode_all_seven = result.tokens.iter().all(|&t| t == 7);

    // 6. Audit log is well-formed: seq is monotonic from 0, the first record's
    //    prev_hash equals the initial live module's hash, and the swap record
    //    carries the expected wasm_hash with success == true. This is the
    //    tamper-evident chain contract from HotSwapPruner.
    let log = pruner.audit_log();
    let seq_monotonic = log.iter().enumerate().all(|(i, r)| r.seq == i as u64);
    let first = log.first();
    let chain_initial_ok = first.is_some_and(|r| r.prev_hash == initial_hash);
    let swap_record_ok = first.is_some_and(|r| r.success && r.wasm_hash == expected_hash);
    let swap_record_full_ok = swap_record_ok && pruner.successful_swaps() >= 1;

    // 7. Trait-object usability post-swap (Send+Sync contract intact). A
    //    scoped immutable reborrow as `&dyn ConstraintPruner` must still
    //    dispatch `is_valid` to the swapped pruner.
    let trait_ok = {
        let trait_ref: &dyn ConstraintPruner = &*pruner;
        trait_ref.is_valid(0, 7, &[])
    };

    let checks: Vec<(&str, bool)> = vec![
        ("live wasm_hash == expected", hash_ok),
        ("live wasm_size == expected", size_ok),
        ("is_valid(0,7,[]) == true", valid_7_at_root),
        ("is_valid(0,0,[]) == false", valid_0_at_root),
        (
            "is_valid ignores parents (3,7,[1,2,3])",
            valid_7_ignores_parents,
        ),
        (
            "is_valid rejects non-7 with parents (3,8,[7,7])",
            rejects_non7_with_parents,
        ),
        ("batch_is_valid matches per-item is_valid", batch_consistent),
        ("manifold_score(valid) == 1.0", score_valid_ok),
        ("manifold_score(invalid) == 0.0", score_invalid_ok),
        ("post-swap speculative_decode verifies", decode_verified),
        ("post-swap decode produces only-7", decode_all_seven),
        ("audit log seq monotonic from 0", seq_monotonic),
        (
            "audit log first record prev_hash == initial live hash",
            chain_initial_ok,
        ),
        (
            "audit log records swap (success, expected wasm_hash)",
            swap_record_full_ok,
        ),
        ("usable as &dyn ConstraintPruner after swap", trait_ok),
    ];

    let total = checks.len();
    let passed = checks.iter().filter(|(_, ok)| *ok).count();
    if passed != total {
        for (name, ok) in &checks {
            if !ok {
                eprintln!("regression FAIL: {name}");
            }
        }
    }
    (passed, total)
}

/// Run the full regression suite immediately after a hot-swap. Every
/// invariant must pass — a regression here means the swap corrupted the
/// engine's contracts. This is the Phase 3 gate "regression suite 100% pass
/// after swap".
#[test]
fn checkbox3_regression_suite_passes_after_hotswap() -> Result<(), Box<dyn Error>> {
    if !skip_if_no_wasm_target() {
        return Ok(());
    }

    let (regex_bytes, only7_bytes) = build_pruner_payloads()?;
    let regex_hash = blake3_of(&regex_bytes);
    let only7_hash = blake3_of(&only7_bytes);
    let only7_size = only7_bytes.len();

    let mut pruner = HotSwapPruner::new(&regex_bytes)?;
    let outcome = pruner.reload(&only7_bytes)?;
    assert_eq!(outcome.wasm_hash, only7_hash);

    let draft = UniformDraftModel::new(10);
    let (passed, total) =
        regression_suite_after_swap(&mut pruner, &draft, regex_hash, only7_hash, only7_size);

    assert_eq!(
        passed, total,
        "regression suite after hot-swap: {passed}/{total} passed (see stderr for failures)"
    );
    Ok(())
}

/// A failed reload must NOT regress the engine — the live pruner keeps
/// serving, and the audit log records the failed attempt with `success: false`.
/// This complements the positive regression test by asserting liveness on
/// failure (a core `HotSwapPruner` design promise).
#[test]
fn checkbox3_regression_failed_reload_preserves_liveness() -> Result<(), Box<dyn Error>> {
    if !skip_if_no_wasm_target() {
        return Ok(());
    }

    let (regex_bytes, _only7_bytes) = build_pruner_payloads()?;
    let regex_hash = blake3_of(&regex_bytes);

    let pruner = HotSwapPruner::new(&regex_bytes)?;

    // Genuinely invalid WASM bytes — instantiation must fail.
    let bad_bytes = [0u8, 97, 115, 109, 0xFF, 0xFF];
    let reload_result = pruner.reload(&bad_bytes);
    assert!(
        reload_result.is_err(),
        "reload of malformed wasm must error"
    );

    // Live pruner is still A (regex). Liveness preserved.
    assert_eq!(
        pruner.wasm_hash(),
        regex_hash,
        "failed reload must leave the live pruner in place"
    );
    assert!(
        pruner.is_valid(0, 1, &[]),
        "regex pruner still serves is_valid"
    );
    assert!(
        !pruner.is_valid(0, 7, &[]),
        "regex pruner still rejects 7 (its alphabet is {{0,1,2}})"
    );

    // Audit log records the failed attempt without breaking the chain.
    let log = pruner.audit_log();
    let failures = log.iter().filter(|r| !r.success).count();
    assert_eq!(failures, 1, "exactly one failed attempt recorded");
    assert_eq!(pruner.failed_swaps(), 1);
    assert_eq!(pruner.successful_swaps(), 0);
    // The failed record's prev_hash is the live module (regex) — chain intact.
    match log.last() {
        Some(record) => assert_eq!(
            record.prev_hash, regex_hash,
            "failed attempt's prev_hash is the still-live module"
        ),
        None => panic!("audit log must be non-empty after a failed reload attempt"),
    }

    Ok(())
}
