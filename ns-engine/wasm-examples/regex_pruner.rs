//! # Example WASM regex pruner (`a*b*c*`)
//!
//! A minimal `#![no_std]` Rust pruner that compiles to
//! `wasm32-unknown-unknown` and is loaded by `ns-engine`'s `WasmPruner` /
//! `HotSwapPruner`. It enforces the regular language `a*b*c*` — the decoded
//! sequence must be drawn from the alphabet `{a, b, c}` (token IDs `0, 1, 2`)
//! AND be non-decreasing: zero or more `a`s, then `b`s, then `c`s.
//!
//! This is a real prefix-validity check: a candidate is accepted only if it
//! could extend the committed prefix toward a word in the language. The
//! pruner reads the committed parent tokens from linear memory to decide.
//!
//! ## ABI
//!
//! Exports exactly what `WasmPruner` requires:
//!   - `memory`              — linear memory (auto-exported for `cdylib`).
//!   - `is_valid(depth: i32, token: i32, parent_ptr: i32, parent_len: i32) -> i32`
//!     returning `1` to accept, `0` to reject.
//!
//! ## Build
//!
//! ```sh
//! rustc --target wasm32-unknown-unknown --crate-type cdylib -O \
//!     regex_pruner.rs -o regex_pruner.wasm
//! ```
//!
//! The resulting `regex_pruner.wasm` can be fed directly to
//! `HotSwapPruner::new` / `HotSwapPruner::reload`.
//!
//! ## Why all four ABI parameters are read
//!
//! - `depth`      — used to cap the sequence length at `MAX_DEPTH` (a
//!                  defensive bound; large enough for any realistic decode).
//! - `token`      — the candidate being screened against the regex.
//! - `parent_ptr` — base address of the parent-token array in linear memory.
//! - `parent_len` — number of committed parents; determines whether we read
//!                  a previous token and which one is "last".

#![no_std]

/// Maximum accepted decode depth. A defensive cap that also demonstrates that
/// the `depth` ABI parameter is consulted (and not silently ignored).
const MAX_DEPTH: i32 = 1024;

/// Token values that form the alphabet `{a, b, c}`. Anything outside this
/// range is rejected — the candidate is not part of the language.
const TOK_A: i32 = 0;
const TOK_B: i32 = 1;
const TOK_C: i32 = 2;

/// Bytes per packed parent token (WASM `i32`, little-endian). Must match the
/// host-side packing in `WasmPruner`.
const TOKEN_PACK_SIZE: usize = 4;

/// WASM validity entry point. See the module docs for the ABI contract.
///
/// # Safety
///
/// Reads from linear memory at `parent_ptr + (parent_len - 1) * 4` when
/// `parent_len > 0`. The host (`WasmPruner`) guarantees that `parent_ptr`
/// points to a region large enough to hold `parent_len` packed `i32` values,
/// so the read is always in-bounds.
#[no_mangle]
pub extern "C" fn is_valid(depth: i32, token: i32, parent_ptr: i32, parent_len: i32) -> i32 {
    // Depth guard: refuse to extend an over-long sequence. This consults the
    // `depth` ABI parameter so a host can observe it being read.
    if depth < 0 || depth >= MAX_DEPTH {
        return 0;
    }

    // Alphabet guard: only {0, 1, 2} are in the language.
    if !is_in_alphabet(token) {
        return 0;
    }

    // Non-decreasing guard: the candidate must be >= the last committed
    // token. With no parents yet, any alphabet token is acceptable.
    match parent_len {
        0 => 1,
        n if n > 0 => {
            // SAFETY: The host (`WasmPruner`) writes the full parent array
            // at `parent_ptr` before invoking `is_valid`, so the read of the
            // last entry at `parent_ptr + (n-1)*4` is in-bounds linear memory.
            let last = unsafe { read_last_parent(parent_ptr, n) };
            bool_to_i32(token >= last)
        }
        // Negative parent_len is impossible from a well-behaved host; treat
        // defensively as a rejection rather than reading garbage.
        _ => 0,
    }
}

/// Membership test for the `{a, b, c}` alphabet.
fn is_in_alphabet(token: i32) -> bool {
    matches!(token, TOK_A | TOK_B | TOK_C)
}

/// Read the last committed parent token (`parent[n - 1]`) from linear memory.
///
/// `parent_ptr` is the base address; the last entry is at offset
/// `(n - 1) * TOKEN_PACK_SIZE`. We compute the address in `usize` to avoid
/// signed-overflow edge cases for very large (or hostile) `n`.
///
/// # Safety
///
/// Caller guarantees `n >= 1` and that `[parent_ptr, parent_ptr + n*4)` is
/// valid, readable linear memory — a precondition the host satisfies by
/// writing the parent array before each call.
unsafe fn read_last_parent(parent_ptr: i32, n: i32) -> i32 {
    let base = parent_ptr as usize;
    let index = (n as usize).saturating_sub(1);
    let addr = base + index * TOKEN_PACK_SIZE;
    // `read_volatile` compiles to a single `i32.load` on wasm32 and prevents
    // the optimizer from assuming the value (we genuinely want the host's
    // packed bytes).
    unsafe { core::ptr::read_volatile(addr as *const i32) }
}

/// Map a `bool` to the WASM return convention (`1` = accept, `0` = reject).
fn bool_to_i32(b: bool) -> i32 {
    match b {
        true => 1,
        false => 0,
    }
}

/// Panic handler. A constrained pruner never panics on the happy path; this
/// handler exists only to satisfy `#![no_std]` and traps if something truly
/// unexpected happens (e.g. arithmetic overflow under panic=unwind, which is
/// not generated in a release `cdylib` build).
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // An `abort`-style trap. `loop {}` lowers to an unreachable loop; the
    // release build has no panic sites in this module, so this is only
    // reached if a future edit introduces one.
    loop {}
}
