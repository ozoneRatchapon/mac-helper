# ns-engine — Agent Implementation Guidelines

> Rules and contracts for any agent (or human) implementing work in this crate.
> Read this BEFORE writing code. It is enforced by review and CI.
>
> Source plan: `.plans/006_modelless_neuro_symbolic_scaffold.md`
> Latest handover: `.handovers/` (check INDEX for current state)

---

## 1. Thesis (the WHY — read first)

This crate is a **modelless neuro-symbolic inference engine** (katopz / katgpt-rs lineage).

- The draft model provides **fluency only** (logits over vocab).
- The symbolic pruner layer provides **correctness** (constraints, DFA, schema).
- The bandit layer provides **adaptation** (which pruner to fire per context).
- KG triples provide **structured memory** (Phase 4, pending).
- WASM hot-swap provides **self-modification** (Phase 3, pending).

> Intelligence evolves by **updating the trial log and swapping pruner code**,
> NOT by gradient descent on weights.

If your change moves intelligence into the draft model, you are violating the thesis.
Stop and reconsider.

---

## 2. Core Loop (immutable)

```
draft.log_probs(context)
    │
    ▼ top-k candidates
pruner.batch_is_valid(depth, candidates, parents, &mut results)
    │
    ▼ valid-only candidates
(bandit) pick arm with highest policy score (UCB1 / ε-greedy / Thompson)
    │
    ▼ best valid candidate
verifier: deterministic check (or p/q rejection if a real verifier exists)
    │
    ▼ accept → emit token, pruner.propagate(+reward)
    │
    ▼ reject → backtrack, pruner.on_backtrack(-1.0)
```

Do not change the loop's shape. Add behavior at the extension points
(`propagate`, `on_backtrack`, new pruner arms).

---

## 3. Enforced Design Rules (NON-NEGOTIABLE)

These apply to every file you touch.

### File hygiene
- Files **MUST** stay under **1024 lines**. Split before merging if needed.
- `mod.rs` is **index only**: `pub mod`, `pub use`. No logic, no impls.
- `types.rs` holds structs/enums/impls, **decoupled** from trait defs.
- `lib.rs` is minimal: module decls and `pub use` re-exports.

### Rust style
- `snake_case` for all functions, methods, variables, modules.
- `PascalCase` for types, traits, enums.
- `match` over chained `if`. Early returns for guards.
- `format!("{var}")` — never `format!("{}", var)`.
- `Option`/`Result` over nulls and panics.
- `unwrap`/`expect` are **banned** in non-test code. Use `?` or explicit matches.
- No `TODO`, `FIXME`, `unimplemented!`, `todo!`, placeholder bodies.
- No mock objects. Use real implementations or test fixtures.

### Hashing and IDs
- **BLAKE3** for all hashing (audit keys, pattern keys, WASM module digests).
  Never SHA-1, SHA-256, or MD5 unless an external protocol requires it.
- When IDs are introduced, use `Uuid::now_v7()` (time-ordered), never `v4()`.

### Dependencies
- Add a dependency only if a real consumer exists in `src/`.
- Prefer crates already in `Cargo.lock` as transitive deps before adding new ones.
- `wasmi` is gated behind the `wasm-pruner` feature (Phase 3).

### Workflow
- Run `cargo clippy --fix --allow-dirty` before commits.
- Clippy MUST pass with `-D warnings` — zero warnings allowed.
- Run the specific failed test: `cargo test -p ns-engine --test <name>`.
- Add new tests under `tests/` (integration) or `#[cfg(test)] mod tests` (unit).
- RUST_LOG=info, build with --quiet for clean logs.

---

## 4. Trait Contracts (do not break)

### `ConstraintPruner` (Phase 1, base trait)
```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth, token, parents) -> bool;
    fn batch_is_valid(&self, depth, candidates, parents, results);
    fn manifold_score(&self, depth, token, parents) -> f32;
    fn propagate(&mut self, depth, token, parents);            // default no-op
    fn on_backtrack(&mut self, depth, token, parents);         // default no-op
}
```

### `ScreeningPruner` (Phase 2, subsumes ConstraintPruner)
```rust
pub trait ScreeningPruner: ConstraintPruner {
    type ArmId; // or use the shared ArmId alias
    fn screen(&self, depth, token, parents) -> f32;            // graded relevance
    fn arm_id(&self) -> ArmId;
    fn arm_label(&self) -> &str;
    // batch variants of screen()
}
```

**Hierarchy rule:** Any `ScreeningPruner` is automatically a candidate bandit arm.
New pruners that should compete in the bandit MUST implement `ScreeningPruner`.

### `DraftModel`
- `vocab_size() -> usize`
- `log_probs(context) -> Logits`

Draft models are **fluency-only**. Do not put constraint logic in them.

---

## 5. Reward Signal Design (the engine of adaptation)

This is how the bandit learns. Do not change the signs without a design discussion.

| Event | Reward | Where called |
|---|---|---|
| Token accepted by verifier | `+screen_score(arm)` | `propagate` |
| Token leads to dead-end (backtrack) | `-1.0` | `on_backtrack` |
| No acceptance signal yet | `0.0` (no update) | — |

The decode loop maintains a `committed: Vec<(depth, arm_index)>` stack.
On backtrack, pop the top entry and attribute `-1.0` to that arm.

**Why this works:** arms that screen aggressively get high positive reward when
right, but pay the full `-1.0` penalty when their choice causes a dead-end.
The asymmetry pushes the bandit toward *calibrated* screening, not greedy screening.

---

## 6. Determinism Contract (critical, easy to break)

Stochastic policies (`ε-greedy`, `Thompson`) MUST select the **same arm** when:
- called from `batch_is_valid(&self)` (the read path), AND
- re-derived in `propagate(&mut self)` (the write path).

**Why it matters:** if selection differs between read and write, the bandit
rewards an arm that didn't actually fire → silent corruption of the trial log.

**Solution (enforced):** derive a per-call seed from `(pattern_key, total_pulls)`
via BLAKE3. **Never** store mutable RNG state on the pruner.

If you add a new stochastic policy, you MUST implement it seed-derivable.
Verify determinism with a property test before merging.

---

## 7. Verification Loop (definition of done per phase)

A phase is NOT done until ALL of:

1. `cargo test --all` — every test passes (unit + integration).
2. `cargo clippy --all-targets -- -D warnings` — zero warnings.
3. File inventory — every touched file under 1024 lines.
4. Phase-specific proof in `tests/` (e.g., `phase2_bandit.rs`).
5. For adaptive work: Arena proof showing **bandit > static > random**.
6. No new `unwrap`/`expect`/`TODO` introduced.

Phase-specific gates:

| Phase | Gate |
|---|---|
| 1 | Sudoku solved with 100% verified solution, backtracking within budget |
| 2 | All three policies (UCB1/ε/Thompson) prove bandit > static > random |
| 3 | WASM hot-swap < 10ms, regression suite 100% pass after swap |
| 4 | KG NIAH retrieval 100%, facts recalled verbatim |
| 5 | ≥3 domains, each proving adaptive > static > random |

---

## 8. Anti-Patterns (learned the hard way — do not repeat)

### Regex prefix-validity
- **Wrong:** `regex` crate cannot do true prefix-validity (no DFA state exposure).
- **Right:** use `regex-automata` — `start_state_forward`, `next_state`,
  `is_dead_state`. A non-dead DFA after consuming the prefix = completable.

### Regex anchoring
- **Wrong:** patterns like `a*` silently accept `b` (empty match at pos 0).
- **Right:** every pattern is anchored at BOTH ends. Auto-wrap as
  `(?:<pattern>)$` at compile time. Document this in the pruner.

### `top_k` vs `vocab_size`
- **Wrong:** `top_k: 9` with `vocab=10` excludes token 9 → Sudoku unsolvable
  when digit 9 is needed.
- **Right:** for full-coverage tests, `top_k == vocab_size`. Document the
  exclusion semantics otherwise.

### Format string indexing
- **Wrong:** `format!("{batch_scores[i]}")` — Rust forbids indexing inside `{}`.
- **Right:** `let s = batch_scores[i]; format!("{s}")`.

### Byte literal casts
- **Wrong:** `vec![b'a' as TokenId, b'b', b'c']` — `b'b'` infers `u8`, not `TokenId`.
- **Right:** cast every element: `vec![b'a' as TokenId, b'b' as TokenId, ...]`.

### Trait method disambiguation
- **Wrong:** `bandit.arm_id(index)` is ambiguous (inherent lookup vs trait method).
- **Right:** call via fully-qualified syntax: `ScreeningPruner::arm_id(&bandit)`.

### Bandit on mutable RNG
- **Wrong:** storing `fastrand::Rng` on the pruner and stepping it inline.
- **Right:** derive seed from `(pattern_key, total_pulls)` via BLAKE3 (see §6).

### Intelligence in the draft model
- **Wrong:** teaching the n-gram model to avoid invalid tokens.
- **Right:** keep the draft model uninformative; let pruners enforce correctness.

---

## 9. Testing Strategy

### Pyramid
1. **Unit** — `#[cfg(test)] mod tests` in each module. Fast, isolated.
2. **Integration** — `tests/phase<N>_<name>.rs`. End-to-end proofs.
3. **Arena** — adaptive proofs: bandit vs static vs random. Required for any
   change that touches arm selection or reward.

### Required test patterns
- **Property tests** for determinism (§6) — same seed ⇒ same selection.
- **Regression tests** for every bug fixed (anti-patterns in §8).
- **Coverage tests** for edge tokens (0, vocab-1, givens, boundary depths).

### Naming
- Integration: `tests/phase<N>_<topic>.rs` (e.g., `phase2_bandit.rs`).
- Unit: descriptive `fn <behavior>_<condition>()` (e.g., `arm_select_ties_break_lowest_index`).

---

## 10. Phase Status & Roadmap

| Phase | Status | Proof |
|---|---|---|
| 1 — Minimal loop (Sudoku) | ✅ Complete | `tests/phase1_sudoku.rs` |
| 2 — BanditPruner (UCB1/ε/Thompson) | ✅ Complete | `tests/phase2_bandit.rs` (incl. decode-loop attribution, group 6) |
| 3 — WASM hot-swap (wasmi) | ✅ Complete | `tests/phase3_wasm_hotswap.rs` (feature-gated — run command below) |
| 4 — KG triple injection | ✅ Complete | `tests/phase4_kg_grounding.rs` |
| 5 — Domain generalization | ✅ Complete | `tests/phase5_arena.rs` + `tests/phase5_benchmark.rs` |

**Only open item:** Phase 2 hardening (Absorb inline, Beta Thompson).
It touches arm selection and reward attribution (§5) — requires a design
discussion before implementation; do not guess at reward-sign changes.

**Feature-gated tests:** `tests/phase3_wasm_hotswap.rs` is behind the
non-default `wasm-pruner` feature, so a plain `cargo test` runs **0 tests**
from it. Run it explicitly to see the Phase 3 proof (requires the
`wasm32-unknown-unknown` target installed):

```sh
cargo test --features wasm-pruner --test phase3_wasm_hotswap
```

---

## 11. Before You Start Any Task

1. Read `.plans/006_modelless_neuro_symbolic_scaffold.md` (the source of truth).
2. Check `.handovers/` for the latest INDEX (current state, blockers, next step).
3. Read the module you're touching — do not guess its shape.
4. Run `cargo test --all` to confirm green baseline before changes.
5. After changes: `cargo clippy --fix`, `cargo test --all`, then write the proof.

---

## 12. When You Finish a Task

1. All gates in §7 pass.
2. No anti-patterns from §8 reintroduced.
3. Update `.handovers/{INDEX}_{TITLE}.md` — what happened, where the code/test
   is, what's struggling/solved, what remains, how to dev/test.
4. If new issue surfaced: CRUD `.issues/{INDEX}_{TITLE}.md`.
5. Suggest a conventional commit (feat/fix/refactor). Use git rebase, not merge.

---

## 13. Quick Reference — Commands

```sh
# Build & test
cargo build --quiet
cargo test --all
cargo test -p ns-engine --test phase2_bandit    # specific integration test

# Lint (must be clean)
cargo clippy --all-targets -- -D warnings
cargo clippy --fix --allow-dirty

# Logs
RUST_LOG=info cargo test -- --nocapture
```

---

## References

- katopz / katgpt-rs — neuro-symbolic micro-transformer reference
- Andrej Karpathy / microgpt — minimal transformer inspiration
- Leviathan et al. — speculative decoding (p/q rejection)
- "Screening Is Enough" (arXiv:2604.01178) — graded relevance pruning
- Matthews et al. — LEO all-goals Q-learning (Phase 2 inspiration)

---

**Version:** 1.1 (post-Phase 5)
**Maintainer of record:** see `.handovers/` latest entry