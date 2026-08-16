# AGENTS.md — mac helper

Rules for any agent working in this repo, including **local models** (gemma4:26b,
qwen3-coder:30b via Ollama in Zed). Read this first. Keep it open while you work.

---

## 1. Local-first rule (non-negotiable)

**All LLM inference in this repo's experiments runs on local Ollama at
`http://127.0.0.1:11434`.** Never call a cloud LLM API from code, tests, or
benchmarks here. Cloud-only models (e.g. `*:cloud` tags) are out of scope.

Check the server is up before anything that needs it:
```sh
curl -s http://127.0.0.1:11434/          # -> "Ollama is running"
ollama list                              # installed models
```

Default models: **`gemma4:26b`** general work, **`qwen3-coder:30b`** code,
`qwen3.8` multimodal/thinking, `nomic-embed-text` embeddings.

---

## 2. Layout — and which rules apply where

| Path | What it is | Extra rules |
|---|---|---|
| `ns-engine/` | Neuro-symbolic decode/search engine | **Read `ns-engine/AGENTS.md` first — it is strict and enforced** |
| `katgpt-rs/` | Pure-Rust micro-transformer runtime | **Read `katgpt-rs/AGENTS.md` first (modelless mandate)** |
| `ollama-lab/` | Local-LLM integration lab (RAG, bridge, chat) | This file only |
| `*.sh` | macOS maintenance scripts | §5 below |
| `.logs/` | Scratch artifacts, repros. Not shipped. | — |

If you are editing inside `ns-engine/` or `katgpt-rs/`, that crate's AGENTS.md
**overrides** this file where they differ.

---

## 3. Build and test

Cargo writes to a **global** target dir (`~/.cargo/target`, set in
`~/.cargo/config.toml`). `./target/` does not exist — do not look for binaries
there. Use `cargo run` instead of running a path.

```sh
# ns-engine
cd ns-engine && cargo test                       # full suite (809 tests)
cargo test --test <name>                         # one integration test

# ollama-lab (needs Ollama running)
cd ollama-lab && cargo run -- health
cargo run -- rag "your question"
cargo run -- bridge gemma4:26b --big

# katgpt-rs
cd katgpt-rs && cargo run --release -p katgpt-cli -- bench --suite intent-router --format json
```

Shell scripts: check syntax with `bash -n <script>.sh` before running.

---

## 4. Hard rules

1. **Never commit unless asked.** Show the diff and stop.
2. **Run the tests you can affect** before saying you are done. Report real
   output; if something fails, say so and paste it.
3. **No `unwrap()`/`expect()` in non-test Rust.** Use `?` or an explicit match.
   (`ns-engine/AGENTS.md` §3 enforces this.)
4. **Do not weaken a test to make it pass.** If an assertion is wrong, say why
   in a comment and record the history.
5. **Read the file before editing it.** Do not guess a function's shape.
6. **No new dependencies** without a real consumer already in the tree.
7. Every bug you fix gets a regression test that fails before and passes after.

---

## 5. Shell script conventions (`port-audit.sh`, `adobe-cleanup.sh`)

- Destructive scripts are **dry-run by default**; deletion requires `--apply`.
- Globs must be passed **unquoted** or expanded in a `shopt -s nullglob` loop —
  a quoted glob never expands, so `[[ -e ]]` silently reports "absent".
- String matching against accumulated multi-line output must be **anchored to a
  line start** (prefix the newline), or `80` matches inside `8080`.
- Colors are TTY-gated. Keep `bash -n` clean.

---

## 6. What to hand a local model (and what not to)

**Good fit** — concrete, verifiable, one file at a time:
- Write or extend a test for behavior that already exists.
- Fix a compile error or clippy warning.
- Rename, extract a helper, remove duplication.
- Write docs/comments for code you have read in full.
- Run a benchmark and report the numbers verbatim.

**Escalate instead of guessing** — say "I need help with X" and stop:
- Changing the decode/generate loop shape or trait contracts.
- Anything touching reward signals or arm selection (`ns-engine/AGENTS.md` §5–6).
- Cross-crate refactors, or a change spanning more than ~3 files.
- Any task where you cannot run something that proves you are right.

**Never**: invent benchmark numbers, claim a test passed without running it, or
delete files to make a build succeed.

---

## 7. Definition of done

1. It builds.
2. The relevant tests run **and you pasted the real result**.
3. No new `unwrap`/`expect`/`TODO`/placeholder.
4. You said plainly what you changed, what you verified, and what you did not.

Uncertainty is fine and useful. Silent guessing is not.
