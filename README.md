# Mac Helper

> A real-world guide to optimizing macOS for developers — from a full Intel → Apple Silicon migration, 77 GB of cleanup, and benchmarking every tool.

## Why This Exists

After migrating from an Intel MacBook to an M5 Pro via Apple Migration Assistant, everything was *technically* working — but carrying 8+ years of Intel baggage: stale binaries, duplicate runtimes, dead app data, and shell configs from 4 different eras.

This project documents what was actually found, what was fixed, and what was measured — so you can do the same on your machine without the guesswork.

---

## Results at a Glance

| Metric | Before | After |
|--------|--------|-------|
| Disk used | ~89 GB | **12 GB** |
| Shell startup | 335 ms | **34 ms** (10× faster) |
| `find` | 515 ms | **17 ms** (30× faster, via `fd`) |
| `grep` | 740 ms | **29 ms** (25× faster, via `rg`) |
| Python runtimes | 2 (3.12 + 3.14) | **1** (Homebrew 3.14.5) |
| Node runtimes | 3 (NVM + old binary + fnm) | **1** (fnm v24.16.0 + v25.9.0) |
| Go runtime | 1 (x86_64, unused) | **0** (removed) |
| `.zshrc` | 80+ lines, duplicates | **25 lines, clean** |
| `/usr/local/` | 8.4 GB | **195 MB** |
| Dead app data | 9.7 GB | **0 GB** |
| Build artifacts | 46 GB | **0 GB** (regenerates on build) |

---

## What Was Done (5 Phases)

### Phase 1: Fix Broken Tools (~7.4 GB freed)

After Migration Assistant, many tools were broken or running Intel binaries via Rosetta.

- Replaced NVM → **fnm** (Rust-based Node manager, 8× faster shell startup)
- Fixed Rust toolchain (was targeting `x86_64` → `aarch64`)
- Cleaned npm cache (5.5 GB, root-owned files from migration)
- Removed 9 stale LaunchAgents (php, adobe, ipfs, virtualbox, postgres, etc.)
- Consolidated `.zshrc` (4 files with duplicates → 1 clean file)
- Set up SSH key in Keychain, global `.gitignore`, firewall

### Phase 2: Upgrade CLI Tools to Rust

Benchmarked 13 Rust CLI tools against BSD originals on real hardware:

| Tool | Replaces | Speedup | Worth It? |
|------|----------|---------|-----------|
| `fd` | `find` | **30×** | Yes — must have |
| `rg` (ripgrep) | `grep` | **25×** | Yes — must have |
| `fnm` | `nvm` | **10×** (shell) | Yes — must have |
| `bat` | `cat` | 3.6× slower | Yes — UX is worth 3ms |
| `eza` | `ls` | 3× slower | Yes — icons, colors, tree |
| `dust` | `du` | ~same | Yes — visual tree |
| `sd` | `sed` | 1.5× slower | Marginal — simpler syntax |
| `delta` | git pager | N/A | Yes — side-by-side diffs |
| `bottom` | `top` | N/A | Yes — modern TUI |
| `hyperfine` | `time` | N/A | Yes — statistical benchmarking |

**Security**: All 13 tools verified — code-signed, arm64 native, zero CVEs, zero network connections, from Homebrew bottles.

Install all at once:
```bash
brew install fd ripgrep bat eza dust sd procs bottom delta tokei hyperfine zellij fnm
```

### Phase 3: Deep Cleanup (~8.2 GB freed)

The first pass cleaned user-level configs but missed system-level leftovers:

- `/usr/local/` had 8.4 GB of Intel artifacts (logs, .NET, Go, old Homebrew, dead Python libs)
- Removed python.org Python 3.12 (duplicate of Homebrew Python 3.14)
- Removed Go (zero Go projects)
- Removed old `/usr/local/bin/node` (v22.12.0, shadowing fnm's v24.16.0)

### Phase 4: Dead App Data + Legacy Frameworks (~13.3 GB freed)

- 6 orphaned Application Support dirs (Cisco Spark, WebEx, Edge, VisualStudio, TabNine, NATURAL8)
- Mono.framework (1 GB, legacy .NET), OSXFUSE, empty Python.framework
- 3 stale Solana releases
- Cargo registry cache (regenerates on `cargo build`)

### Phase 5: Build Artifacts (~47.3 GB freed)

- Rust `target/` dirs across 4 projects (debug builds, incremental caches)
- x86_64 cross-compilation target (wrong architecture)
- DMG installers in Downloads

---

## Reproduce on Your Machine

### 1. Find what's wasting space

```bash
# Home directory breakdown
du -sh ~/*/ 2>/dev/null | sort -rh | head -20

# Dead app data (compare with installed apps)
du -sh ~/Library/Application\ Support/* 2>/dev/null | sort -rh | head -20

# Build artifacts
find ~ -name "target" -type d -path "*/Rust/*" -o -name "target" -type d -path "*cargo*" 2>/dev/null | while read d; do du -sh "$d"; done | sort -rh

# Old caches
du -sh ~/Library/Caches/* 2>/dev/null | sort -rh | head -10
```

### 2. Clean build artifacts (regenerates on build)

```bash
# Cargo debug builds (safe to remove, keeps release/)
rm -rf ~/your-project/target/debug/incremental

# Or full clean
cargo clean --manifest-dir ~/your-project/Cargo.toml

# npm cache
npm cache clean --force

# Homebrew cache
brew cleanup --prune=all
```

### 3. Check for duplicate runtimes

```bash
# How many Pythons?
which -a python3

# How many Nodes?
which -a node

# How many Rust toolchains?
rustup toolchain list
```

### 4. Benchmark your tools

```bash
brew install hyperfine
hyperfine 'fd . /Users --type f --max-depth 5' 'find /Users -maxdepth 5 -type f'
hyperfine 'rg -c "fn main" ~/projects' 'grep -rc "fn main" ~/projects'
```

### 5. Verify security of installed tools

```bash
# Check architecture (should be arm64, not x86_64)
file $(which fd bat eza rg)

# Check code signing
codesign -vvv --strict $(which fd bat eza rg)

# Check for quarantine attributes
xattr $(which fd bat eza rg)
```

---

## Key Lessons

1. **Migration Assistant copies everything** — including Intel binaries, stale LaunchAgents, and dead configs. Plan a cleanup session after migration.

2. **`/usr/local/` is the graveyard** — on Apple Silicon, Homebrew uses `/opt/homebrew/`. Anything in `/usr/local/` is likely from the Intel era.

3. **Measure before replacing tools** — some Rust alternatives are *slower* for simple operations (bat, eza) but the UX improvement is worth the 3-5ms overhead.

4. **Build artifacts are the biggest silent consumer** — a single Rust workspace can accumulate 15+ GB of `target/` files. Clean debug builds periodically.

5. **One runtime per language** — multiple Python/Node versions cause path conflicts and confusion. Use version managers (fnm, pyenv) but pin to one default.

6. **Shell startup time matters** — 335ms → 34ms means every terminal window opens instantly. Replace NVM with fnm, clean PATH duplicates.

---

## Quick Start

```bash
# Audit your Mac (safe, read-only)
curl -s https://raw.githubusercontent.com/<you>/mac-helper/main/mac-audit.sh | bash

# Or clone and run
git clone https://github.com/<you>/mac-helper.git
cd mac-helper
./mac-audit.sh
```

## File Structure

```
mac helper/
├── README.md                                    ← You are here
├── mac-audit.sh                                 ← Audit script (run on any Mac)
├── .handovers/
│   └── 001_intel_to_apple_silicon_migration.md  ← Full technical reference
└── .issues/                                     ← Issue tracking
```

## For the Full Technical Details

See [`.handovers/001_intel_to_apple_silicon_migration.md`](.handovers/001_intel_to_apple_silicon_migration.md) for:
- All 18 issues found & fixed (with exact commands)
- Benchmark methodology & raw numbers
- Security audit details
- Config backup locations
- Revert instructions

---

## Remaining

- [ ] Set up Time Machine (buy external SSD)
- [ ] Consider a dotfiles repo (`.zshrc` is clean at 25 lines — good time to version-control)
- [ ] `brew update && brew upgrade` periodically

---

*Based on a real Intel → M5 Pro migration on 2026-06-13. Total freed: ~77 GB.*
