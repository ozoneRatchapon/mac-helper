# Ollama Overnight Plan — 2026-08-16

**Hard constraint:** every LLM inference in these experiments runs against local Ollama
(`http://127.0.0.1:11434`). No cloud LLM APIs (no Claude/OpenAI/Ollama-cloud calls) inside
any experiment, benchmark, or scaffold built tonight.

**Machine:** MacBook Pro M5 Pro, 48 GB RAM, 1.1 TB free disk. Ollama 0.32.13 (brew),
server bound to 127.0.0.1 only. Keep-awake: `caffeinate -dims` running for the session.

**State tracking:** each phase appends results to `ollama-night-report.md` (repo root).
If the session restarts, resume from the first unchecked box below.

---

## Phase 0 — qwen3.8 online — DONE 01:45
- [x] `ollama pull qwen3.8` completes (17 GB on disk)
- [x] Smoke test: prompt responds; `ollama ps` shows RAM footprint (17 GB, 100% GPU)
- [x] Baseline perf: cold load 7.7s, warm 0.14s, ~25 tok/s generation → see report

## Phase 1 — Model bench & eval
Pull sequentially, bench each, then free memory (`ollama stop <model>`):
- [x] `qwen3.8:27b` (already pulled in Phase 0)
- [x] `gemma4:26b` (17 GB, benched — 75 tok/s, all probes pass)
- [x] `qwen3-coder:30b` (18 GB, benched — 90 tok/s, fn correct / own test wrong)
- [x] `gpt-oss:20b` (13 GB, benched — 65 tok/s, format=json FAIL)
- [x] `nomic-embed-text` (274 MB, embed test green)

Per model record: download size, load time, RAM (`ollama ps`), eval tok/s, and quality on
fixed probes: (a) Rust code generation, (b) multi-step reasoning, (c) strict-JSON output
adherence (ties into ns-engine JsonSchemaPruner), (d) long-context recall sanity check.
Output: comparison matrix in `ollama-night-report.md` with a recommendation.

## Phase 2 — Rust integration scaffold (`ollama-lab/`)
New crate in this repo, following the rustify.rs article but corrected to current versions:
- [x] `cargo new ollama-lab` with `ollama-rs = "0.3.6"` (article says 0.2 — outdated)
- [x] Non-streaming generate (plus reqwest used in health check)
- [x] Streaming generation, token-by-token to stdout (tested green vs gemma4:26b)
- [x] Multi-turn chat with history (tested green)
- [x] Embeddings via `nomic-embed-text` (green: 0.451 related vs 0.337 unrelated)
- [x] Health-check + graceful "server down" handling (article's mistakes section)
- [x] All examples run green (winner: gemma4:26b for general, qwen3-coder:30b for code)
Note: builds land in ~/.cargo/target (global target-dir in ~/.cargo/config.toml) — use `cargo run`.

## Phase 3 — ns-engine ↔ Ollama bridge (design exploration + prototype)
Ollama's API exposes no per-token logits, so the classic pruner-in-the-decode-loop shape
doesn't fit. Explore generate-and-validate instead:
- [x] Read ns-engine Phase 5 traits (`GameState`, `SpeculativeGenerator`, `speculative_generate`)
- [x] Design note: LLM fills the DraftModel seat as a RANKER (rank→pseudo-logits); pruner keeps legality — see report
- [x] Prototype adapter in `ollama-lab/src/bridge.rs`: OllamaDraftModel implements the real
      ns_engine::DraftModel; quest-diamond solved optimally, 0/18 parse failures
- [x] Write up findings: mechanically sound; pays off only on high-branching domains — see report

## Phase 4 — Local RAG / embeddings
Fully offline semantic search:
- [x] Corpus: markdown across ~/mac helper (46 files, 224 chunks)
- [x] Chunk → embed with `nomic-embed-text` → in-memory index (3.2 s to index)
- [x] Query loop: top-4 retrieval → gemma4:26b grounded answer (accurate on test query)
- [x] Record retrieval quality observations in the report

## Morning deliverables
1. `ollama-night-report.md` — bench matrix, model recommendation, per-phase findings
2. `ollama-lab/` — runnable Rust crate (scaffold + RAG + bridge prototype)
3. ns-engine bridge design note (inside the report)
