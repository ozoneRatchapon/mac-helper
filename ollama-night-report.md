# Ollama Overnight Report — 2026-08-16

Machine: MacBook Pro M5 Pro, 48 GB RAM. Ollama 0.32.13, server on 127.0.0.1:11434 (localhost-only).
All inference in this report is fully local.

## Morning TL;DR (finished ~03:45)

1. **All 4 planned phases complete** + bonus experiments. Everything ran locally; zero cloud LLM calls.
2. **Model picks:** `gemma4:26b` as daily driver (75 tok/s gen, 1104 tok/s prompt, aced every probe);
   `qwen3-coder:30b` for coding speed (90 tok/s — but it wrote a wrong test assertion once);
   `qwen3.8:27b` for multimodal/thinking; avoid `gpt-oss:20b` for structured output (format=json fail).
   Zed is wired to qwen3.8 via the Ollama provider — consider adding gemma4:26b there too.
3. **⭐ Found and FIXED a real ns-engine bug** (fix user-approved in the morning): backtrack mode
   explored draft candidates in *inverted* preference order (worst-first). Cost the LLM bridge
   10,000 attempts vs 17 post-fix. Fix applied + regression tests added + two luck-dependent
   tests re-tuned; suite green at 807. Details in the headline section below.
4. **New crate `ollama-lab/`:** health/generate/stream/chat/embed/bridge/rag subcommands, all
   green. `bridge` runs your real `speculative_generate` with an LLM as `DraftModel`; `rag` does
   fully-local retrieval-augmented Q&A over this repo's markdown (46 files, 3.2 s to index).
5. LLM-as-ranker comparison on the decoy domain: gemma4 19 attempts (optimal-ish), qwen3-coder 32,
   uniform 26 — the ranker only beats symbolic search when its rankings are good AND ordering
   semantics are right; each LLM call costs ~1–2 s vs microseconds for uniform.

## Phase 0 — qwen3.8 baseline (01:0x–01:4x)

`ollama pull qwen3.8` → 17 GB on disk, tag `qwen3.8:latest` (= 27b), 256K context, text+image, thinking enabled by default.

| Metric | Value |
|---|---|
| RAM while loaded | 17 GB, 100% GPU (Metal) |
| Cold load | 7.7 s |
| Warm load | 0.14 s |
| Prompt eval | 47–57 tok/s |
| Generation | 24–26 tok/s |
| Default context in `ollama ps` | 32768 |

Notes:
- Thinking mode is on by default — it "thinks" even for trivial prompts (adds latency for simple tasks; can be disabled per-request with `"think": false` in the API).
- Code smoke test: produced a correct, idiomatic Rust `reverse(s: &str) -> String` with doc comment.
- Model auto-unloads after ~4 min idle (default keep-alive); warm load is instant while resident.

## Phase 1 — Model bench & eval

### qwen3.8:27b probes
- **Rust codegen**: asked for `top_k` word-frequency fn + test, no-markdown constraint honored;
  output compiled first try with `rustc --test` and its own test passed. 27.7 tok/s with `think:false`.
- **Tool calling** (OpenAI-compatible `/v1/chat/completions`): correctly emitted a
  `run_shell("ls -la")` tool call with `finish_reason: tool_calls` — agent-harness capable.
- Remaining probes (reasoning, strict-JSON, long-context) pending; will run the same battery on gemma4:26b.

### gemma4:26b probes (02:0x)
- 17 GB on disk / 17 GB RAM, 100% GPU. Cold load 3.3 s. **75 tok/s generation** (~3× qwen3.8).
- **Rust codegen**: compiled first try, test passed (same battery as qwen3.8).
- **Tool calling**: correct `run_shell("ls")` call, `finish_reason: tool_calls`.

### Head-to-head battery (both with think:false)
| Probe | qwen3.8:27b | gemma4:26b |
|---|---|---|
| Rust codegen compile+test | pass | pass |
| Tool calling (OpenAI endpoint) | pass | pass |
| Multi-step arithmetic ($67.50) | correct, 25 s | correct, 12 s |
| JSON schema adherence (unassisted) | pass | pass |
| JSON schema adherence (format=json) | pass | pass |
| Generation speed | 25–29 tok/s | **75 tok/s** |

Early read: quality tied on this battery; gemma4:26b wins decisively on speed at equal RAM.
qwen3.8 retains edges elsewhere: 65K+ usable context configured in Zed, native image+video, thinking mode.

### qwen3-coder:30b and gpt-oss:20b probes (02:4x)
(qwen3-coder-next was considered but its smallest quant is 52 GB — too big for 48 GB RAM)

## Phase 1 final matrix

| Probe | qwen3.8:27b | gemma4:26b | qwen3-coder:30b | gpt-oss:20b |
|---|---|---|---|---|
| Disk / RAM | 17 GB | 17 GB | 18 GB | 13 GB |
| Cold load | 7.7 s | 3.3 s | 5.4 s | 3.9 s |
| Generation speed | 25–29 tok/s | 75 tok/s | **90 tok/s** | 65 tok/s |
| Rust codegen compile+test | pass | pass | fn correct, **its own test assertion wrong** | pass |
| Multi-step arithmetic | correct | correct | correct | correct |
| JSON adherence (unassisted) | pass | pass | pass | pass |
| JSON adherence (format=json) | pass | pass | pass | **PARSE FAIL** |
| Tool calling (OpenAI endpoint) | pass | pass | pass | pass |
| Context / extras | 256K, image+video, thinking | 256K, image, thinking, fast | 256K, agentic-coding tuned | 128K, reasoning-effort levels |

**Recommendations:**
- **Daily driver / Zed agent:** `gemma4:26b` — same quality on this battery as qwen3.8 at ~3× the speed; or `qwen3-coder:30b` for coding sessions (fastest, agentic-tuned, but write your own tests).
- **Multimodal / long-thinking tasks:** `qwen3.8:27b`.
- **Caution:** `gpt-oss:20b` failed constrained JSON (`format=json`) — avoid it for structured-output pipelines.
- `nomic-embed-text` embeddings sane: related concepts 0.451 vs unrelated 0.337 cosine.
- **Long-context recall** (needle at 70% of a ~16K-token log haystack, num_ctx 32768): both
  qwen3.8 and gemma4:26b recalled the code exactly. Prompt-processing speed differs sharply:
  gemma4 **1104 tok/s** vs qwen3.8 306 tok/s (22 s vs 60 s wall) — gemma4 is the better choice
  for long-document / RAG workloads.

## Phase 2 — ollama-lab scaffold (02:1x–02:3x)
`ollama-lab/` crate created; **compiled first try against ollama-rs 0.3.6** (article's 0.2 API
carried over almost unchanged: `ChatMessage::system/user` constructors instead of
`ChatMessage::new(MessageRole::…)` is the main surface difference).
- `health` — reqwest GET with a clear "start `ollama serve`" error instead of raw connection-refused
- `generate` / `stream` — both green against gemma4:26b; token-by-token stdout streaming works
- `chat` — multi-turn with history green; model answers stayed consistent across turns
- `embed` — written, waiting for nomic-embed-text
Model picked per-run via `OLLAMA_MODEL` env (default qwen3.8). Build artifacts go to
`~/.cargo/target` (global target-dir).

## Phase 3 — ns-engine ↔ Ollama bridge (02:5x)

**Design.** Ollama exposes no per-token logits, so the LLM cannot sit inside the decode loop the
classic way. But ns-engine's `DraftModel::log_probs` only requires "any ordering signal" — so the
LLM fits the DRAFT seat as a *ranker*: prompt it with the domain rules + trajectory so far, ask for
a full ranking of the action vocabulary as constrained JSON (`format=json`, temperature 0), convert
rank → pseudo-logits. Legality stays with `QuestActionPruner`, dynamics with `QuestState` — the
ownership boundary ns-engine already draws (draft = fluency, pruner = correctness) maps cleanly
onto "LLM = fluency, symbolic layer = correctness". Parse failures degrade gracefully to uniform
logits (all-ties), so the loop never stalls on a bad LLM response.

**Prototype.** `ollama-lab/src/bridge.rs` — `OllamaDraftModel` implements the real
`ns_engine::DraftModel` trait (ns-engine as path dep); ran `speculative_generate` on the 5-quest
diamond domain (0 → {1,2} → 3 → 4, goal = quest 4), backtracking mode:

| Draft model | goal | steps | attempts | LLM calls | parse failures |
|---|---|---|---|---|---|
| UniformDraftModel | yes | 10 | 14 | — | — |
| gemma4:26b (ranking) | yes | 10 | 26 | 18 | 0 |

Both found the optimal 10-action trajectory.

**Findings / verdict.**
- The bridge is *mechanically sound*: constrained JSON ranking over an action vocabulary works
  reliably (0/18 parse failures with gemma4:26b), and rank→pseudo-logit satisfies the trait contract.
- On a small, densely-pruned domain the LLM draft is strictly worse than uniform: more attempts
  (its ranking sometimes promotes illegal actions the pruner then rejects) and ~1–2 s of wall-clock
  per `log_probs` call vs microseconds for uniform. Symbolic DFS alone already crushes this domain.
- Where it should pay off: domains with a **large legal branching factor where most legal moves are
  strategically bad** — the pruner can't help choose among legal moves; the LLM can. Worth a
  follow-up experiment with a bigger quest graph (e.g. 20+ quests, decoy branches) or Bomber maps.
- If pursued for real: cache rankings by state hash (`GameState::hash` exists for exactly this),
  batch the ranking ask ("top 5 next actions" instead of full vocabulary), and consider asking the
  LLM only at high-branching decision points, falling back to uniform elsewhere.

## ⭐ Headline finding — ns-engine backtrack mode inverts draft preference (03:1x–03:3x)

The decoy-domain bridge experiment (12 quests, goal chain 0→4→8→11, 8 decoys) produced an
inversion that turned out to be a real ns-engine behavior, not an LLM failure:

| Run | goal | attempts | LLM calls |
|---|---|---|---|
| uniform draft | yes | 26 | — |
| gemma4 ranking, best→highest logit (per trait docs) | **no** | 10,000 (budget) | 5,009 |
| gemma4 ranking, deliberately INVERTED | yes | **19, zero backtracks** | 19 |

**Mechanism.** `valid_candidates` returns candidates best-first (its docs say so), but
`generate_with_backtrack` / `decode_with_backtrack` consume them with `untried[depth].pop()` —
from the END. Backtracking DFS therefore explores the draft's WORST-ranked candidate first.
Greedy mode uses `valid.first()` — correct. So the two modes assign opposite meanings to the
same logits, and `DraftModel`'s "higher values = more likely" contract only holds in greedy mode.

**Why no test ever caught it:** every prior backtrack test used `UniformDraftModel` (all logits
tied → order fully shuffled). The one non-uniform case — `BacktrackDraft` in `tests/bomber.rs` —
**documents the inversion and is hand-tuned to exploit it** ("tuned for the engine's worst-first
backtracking search": WAIT ranked highest so it's popped last, moves scored with `+distance` so
nearest-to-exit pops first).

**Resolution (morning, user-approved): best-first fix ADOPTED.**
- Two-line fix applied to `decode.rs` + `game.rs` (reverse `valid` before pushing onto `untried`,
  so `pop()` really takes the best candidate).
- `tests/backtrack_ordering.rs` added: rigged non-uniform draft asserts best-first exploration and
  greedy/backtrack agreement.
- `BacktrackDraft` in `tests/bomber.rs` re-tuned to best-first scoring (moves `1000-distance` >
  PLACE_BOMB > reverse-of-last > WAIT); doc comments updated.
- `test_generate_backtrack_respects_max_tokens` in `tests/game_traits.rs` had been passing by
  tie-shuffle luck while violating the pruner's canonical-origin invariant (starts at D, pruner
  replays from A); now uses `RightFirstDraft` so it's deterministic.
- **Full suite green: 807 tests.** The `.logs/` patch artifacts are superseded by the applied fix.

**Post-fix bridge result** (natural best→highest mapping, no workaround): gemma4:26b solved the
decoy domain in **17 attempts, zero backtracks**, while uniform DFS took 2,682 attempts on the
same run (tie-shuffle variance) — with correct semantics the LLM draft decisively beats
uninformed search on strategy-heavy domains.

## Follow-up (morning) — MLX variants vs current GGUF builds

User asked whether `qwen3.8:27b-mlx` / `gemma4:31b-mlx` beat the current builds. Same battery,
server restarted with `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` (≈ speed-neutral for
generation, halves KV memory — kept).

| | qwen3.8 GGUF | qwen3.8:27b-mlx | gemma4:26b GGUF | gemma4:31b-mlx |
|---|---|---|---|---|
| Generation | 25–29 tok/s | **45 tok/s** | **68–75 tok/s** | 35–37 tok/s |
| Prompt eval (8K–16K ctx) | 306 tok/s | 105 tok/s | **1071 tok/s** | 84 tok/s |
| Quality battery (code/math/JSON/tools) | pass | pass | pass | pass |
| Long-context needle | pass | pass | pass | pass (99 s for 8K) |

**Verdict:**
- `gemma4:31b-mlx` — strictly worse than gemma4:26b here: half the generation speed, **13× slower
  prompt ingestion** (the 39K-token needle test timed out at 10 min vs 22 s on GGUF). Skip it.
- `qwen3.8:27b-mlx` — real +60% generation over qwen GGUF, but ~3× slower prompt processing;
  fine for short-prompt chat, wrong for RAG/agent contexts.
- **`gemma4:26b` (GGUF) remains the daily driver** — it beats every variant tested on both axes.
- Ollama's MLX runtime also handled `num_ctx` differently (processed the full ~39K-token prompt
  where GGUF windowed to ~16K) and its prefill speed looks immature relative to upstream mlx-lm;
  worth re-testing in a few Ollama releases.

## Follow-up (morning) — speculative decoding experiment (llama.cpp)

Goal: 2× generation via draft-model speculation on qwen3.8:27b, using the GGUF blobs already in
`~/.ollama/models`.

- **Draft-model path blocked:** qwen3.8 uses a new 248,320-token vocab; qwen3:0.6b has 151,936.
  llama.cpp requires near-identical vocabs, and no small qwen3.8-family model exists yet. Revisit
  when Alibaba ships a qwen3.8 mini.
- **Model-free n-gram speculation** (llama.cpp b10360 `--spec-type`, via llama-server) on an
  echo-heavy code-rewrite task, temp 0, n=300:

| Mode | gen tok/s | drafted → accepted |
|---|---|---|
| none (baseline) | 15.5 | — |
| ngram-simple | **18.5 (+19%)** | 144 → 66 (46%) |
| ngram-map-k | 18.5 (+19%) | 144 → 66 |
| ngram-mod | 17.2 | 128 → 43 |
| ngram-cache | 12.9 (slower!) | 194 → 97 |

- **Caveat that decides everything:** brew's llama.cpp generates at only 15.5 tok/s on the same
  blob where Ollama does 26 — so even +19% (18.5) loses to plain Ollama today. The speculation
  mechanism is validated (46% acceptance on edit-style output), but it only pays once the baseline
  runtime matches Ollama (source-built llama.cpp with better Metal tuning, or Ollama exposing
  speculation itself). Also note llama-cli b10360 ignores `-no-cnv` (use `llama-completion`), and
  `llama-speculative` requires `-md` even for ngram modes (use `llama-server`).

`ollama-lab rag "<question>" [model]` — walks `~/mac helper` for markdown (skipping target/,
dot-dirs), paragraph-aligned ~1200-char chunks, batch-embeds with nomic-embed-text, cosine top-4,
answers with a local chat model constrained to the retrieved context.

- Corpus: 46 files → 224 chunks, **indexed in 3.2 s** entirely locally.
- Test query ("ns-engine core thesis + what ConstraintPruner owns"): retrieval ranked
  `ns-engine/AGENTS.md` first (0.698), pulled related katgpt-pruners README cross-repo; gemma4:26b's
  answer was accurate and stayed within the provided context.
- Observation: retrieval quality is good enough for repo-doc Q&A at zero cost and no data egress;
  the whole loop (index + retrieve + answer) is fast enough to rebuild the index per invocation,
  so no persistence layer was needed at this corpus size.

## Follow-up (2026-08-17) — how far can Zed's context actually go?

Zed's Ollama provider passes `max_tokens` straight through as `num_ctx` (confirmed in Zed docs and
visible in `ollama ps`). Both models declare a 262,144-token native ceiling. Question was whether
48 GB of RAM can back that.

**Load test — all six configs loaded, 100% GPU, no CPU spill:**

| Model | num_ctx 131072 | 200000 | 262144 |
|---|---|---|---|
| gemma4:26b | 17 GB, 5 s | 17 GB, 5 s | 17 GB, 3 s |
| qwen3.8 | 17 GB, 5 s | 18 GB, 6 s | 18 GB, 6 s |

Ollama does **not** preallocate the KV cache — it grows with tokens actually used. An earlier
estimate in this file ("256K needs ~52 GB, won't fit") described a *full* context, not the load
footprint, and was wrong as a practical limit.

**Fill test — gemma4:26b at num_ctx=200000, a real ~743 KB prompt:**

| Metric | Value |
|---|---|
| Prompt tokens processed | 100,003 |
| Resident size / processor | **17 GB, stayed 100% GPU** |
| Prompt eval rate | **286 tok/s** (vs 1071 tok/s at ~8K context) |
| Wall clock, single query | **360 s** |
| Needle recall (planted at 70% depth) | correct (`ORION-4417`) |

**Verdict: the binding constraint is TIME, not RAM.** Resident size never left 17 GB, never spilled
to CPU, and recall at depth stayed correct — but prompt throughput degraded ~3.7x and one
100K-token query took six minutes. Since Zed fills whatever window it is given, a large
`max_tokens` mostly buys slow turns rather than outright failures.

Settings now: both Ollama models at `max_tokens: 200000` (matching the GLM entry's number; backup
at `~/.config/zed/settings.json.bak.pre-ctx200k`). The *shape* cannot be copied from GLM — that is
an `openai_compatible` provider with `max_output_tokens` / `max_completion_tokens` / `capabilities`,
while the Ollama provider supports only `name`, `display_name`, `max_tokens`, `keep_alive`,
`supports_tools`, `supports_thinking`, `supports_images`.

For long-context work prefer **gemma4:26b** — sliding-window attention (window 1024) keeps its KV
cache far cheaper than qwen3.8's full attention over 65 layers (~133 KB/token).

## Follow-up (2026-08-19) — does quantization cost us anything?

Every local model here runs Q4_K_M. That raised a question the suite could not answer from
inside: when a model fails a task, is that the model, or the 4-bit compression?

A free community endpoint (HF Space `victor/Qwen3.8-27B-free-endpoint`, 1×H200 / vLLM) serves
**unquantized BF16 Qwen3.8-27B** — the same model as local `qwen3.8:latest`, at full precision,
no key required. `ollama-lab eval` gained an `--openai <base-url>` flag so the identical task
definitions and scorers could measure it; running two implementations would have made the
numbers incomparable.

| Task | BF16 (hosted) | Q4_K_M (local) |
|---|---|---|
| rust_codegen | 1.00 | 1.00 |
| json_unassisted / json_format | 1.00 | 1.00 |
| arithmetic | 1.00 | 1.00 |
| tool_call | 1.00 | 1.00 |
| needle | 1.00 | 1.00 |
| multi_turn | 1.00 | 1.00 |
| **arith_words** | **0.50** | **0.50** |
| codegen_strict | 1.00 | 1.00 |
| json_deep | 1.00 | 1.00 |
| tool_choice | 1.00 | 1.00 |
| needle_deep | 1.00 | 1.00 |
| multi_turn_update | 1.00 | 1.00 |
| **overall** | **0.96** | **0.96** |

**Identical across all thirteen tasks.** Q4_K_M compresses the model roughly fourfold and costs
nothing measurable here.

The one shared failure makes the point more strongly than the matching totals do: both miss the
same bakery word problem, but *differently* — Q4 answered 447, BF16 answered 4325, correct is
460. A quantization artifact would look like BF16 succeeding where Q4 fails. Two different wrong
answers to the same question is a model limitation with multi-step arithmetic carrying a
distractor value, at any precision. The fix is a calculator tool or a different model, not a
better quant.

Also settled: `qwen3.8` Q4 scores 0.96, exactly matching `gemma4:26b`, and both miss only
arith_words. `qwen3-coder:30b` sits at 0.90 — its needle_deep failure (grabbing a decoy) is
specific to that model, not a general long-context weakness, since qwen3.8 scores 1.00 on the
same task.

Result files carry a `backend` field: `ollama-native` rows form the local ratchet, `openai` rows
came from a remote endpoint and must not be compared against them as though the setup were the
same. The endpoint's own timings ranged from 1.5 s to 245 s — it is shared and rate-limited, and
its author says it will be retired. Useful as a one-off reference, not as a dependency.

## Zed ↔ Ollama handoff (01:5x)
Added `ollama` provider to `~/.config/zed/settings.json` (backup: `settings.json.bak.pre-ollama`):
qwen3.8 exposed as "Qwen 3.8 27B (local)", 65536-token context, tools/thinking/images on,
30 min keep-alive. Select it in Zed's Agent Panel model picker under **Ollama**.
