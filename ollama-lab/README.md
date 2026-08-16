# ollama-lab

Local-LLM integration lab. Every command talks only to the local Ollama server at
`127.0.0.1:11434` — nothing leaves the machine.

Built during the 2026-08-16 overnight session; see `../ollama-night-report.md` for benchmarks
and findings, `../ollama-night-plan.md` for the phase plan.

## Commands

```bash
cargo run -- health                      # server reachability check
cargo run -- generate "prompt"           # one-shot completion
cargo run -- stream "prompt"             # token-by-token streaming
cargo run -- chat                        # scripted multi-turn chat with history
cargo run -- embed                       # nomic-embed-text cosine-similarity demo
cargo run -- bridge [model]              # ns-engine DraftModel backed by an Ollama ranker
cargo run -- rag "question" [model]      # fully-local RAG over ~/mac helper markdown
```

Model selection: `OLLAMA_MODEL` env var for generate/stream/chat (default `qwen3.8`);
`bridge` and `rag` take the model as a positional arg (default `gemma4:26b`).

## Notes

- `ollama-rs` 0.3.6 (the rustify.rs article's 0.2 API carries over almost unchanged).
- Build artifacts land in `~/.cargo/target` (global `target-dir` in `~/.cargo/config.toml`).
- `bridge` runs the sync ns-engine loop on a plain OS thread because it uses
  `reqwest::blocking`, which must not run on a tokio runtime thread.
