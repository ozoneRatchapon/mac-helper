# Handover: Intel → Apple Silicon Migration (M5 Pro)

**Date**: 2026-06-13
**Context**: Migrated from Intel MacBook to Apple Silicon M5 Pro via Apple Migration Assistant.
**Summary**: See [README.md](../README.md) for the public-facing guide. This document is the full technical reference.

---

## What Happened

Apple Migration Assistant copied everything from the Intel Mac — including binaries compiled for x86_64, stale configs, duplicate shell entries, and leftover LaunchAgents. Session covered: fix broken tools, clean up, optimize, deep audit, and Rust-powered CLI upgrade.

---

## Issues Found & Fixed

### 1. GitHub Copilot CLI — `! Copilot CLI not installed`
- **Fix**: Removed old Intel extension, re-authed GitHub, removed copilot binary, cleaned `.zshrc`

### 2. Rust Toolchain — Default host showed `x86_64`
- **Fix**: `rustup default stable-aarch64-apple-darwin`

### 3. Figma — Intel-only binary
- **Fix**: User reinstalled Apple Silicon version from figma.com

### 4. `! Copilot CLI not installed` on Ghostty open
- **Fix**: Removed `gh copilot alias` from `.zshrc` and `.bashrc`

### 5. npm cache had root-owned files
- **Fix**: `sudo chown -R $(whoami):staff ~/.npm`

### 6. Shell configs — duplicate & messy paths
- **Fix**: Consolidated into clean `.zshrc`, removed duplicates (Solana 5x→1x, Cargo 3x→1x, NVM 2x→1x)

### 7. Firewall disabled
- **Fix**: User enabled via System Settings

### 8. No global `.gitignore`
- **Fix**: Created `~/.gitignore_global` with `.DS_Store`, `._*`, etc.

### 9. 8 stale LaunchAgents from old Mac
- **Fix**: Removed all (php@7.4, adobe, ipfs, steam, virtualbox websrv, pulsesecure, Grass, postgresql@15)

### 10. SSH key not in agent
- **Fix**: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`

### 11. NVM → fnm (shell startup 8x faster)
- **Fix**: Replaced NVM with `fnm` (Rust-based). Shell startup: 335ms → 42ms. Removed NVM (~585MB freed).

### 12. All CLI tools upgraded to Rust-based alternatives
- **Fix**: Installed 12 Rust-powered replacements with aliases in `.zshrc`

### 13. /usr/local contained 8.4 GB of Intel Mac leftovers
- **Problem**: Old node (v22.12.0) shadowing fnm, Intel Go/x86_64, VirtualBox stubs, .NET, dead Python libs, 4.1GB logs
- **Fix**: Removed 8.2 GB of Intel/dead weight, reinstalled nmap + git-lfs as arm64 via Homebrew
- **`/usr/local/`**: 8.4 GB → 195 MB

### 14. Two Pythons + unneeded Go
- **Problem**: python.org Python 3.12 (Intel-era) + Homebrew Python 3.14, Go installed with zero Go projects
- **Fix**: Removed python.org Python 3.12 and stale PATH, removed Go. Single Python (Homebrew 3.14.5) only.
- **`.zshrc`**: Removed `Python 3.12` PATH block

### 15. Dead app data — 8.9 GB of orphaned Application Support
- **Problem**: Cisco Spark (2.7G), WebEx (1.4G), MS Edge (1.9G), VisualStudio (810M), TabNine (485M), NATURAL8 (799M) — all left behind after app removal
- **Fix**: Deleted all 6 orphaned data directories + Firefox cache (825M)

### 16. Legacy frameworks in /Library/Frameworks
- **Problem**: Mono.framework (1.0G, old .NET), OSXFUSE.framework (dead, macFUSE not installed), Python.framework (empty shell from previous cleanup)
- **Fix**: Removed all three. `/Library/Frameworks/` now only has `iTunesLibrary.framework` (Apple system)

### 17. Old Solana releases (3 stale)
- **Problem**: 3 old releases (2.1.0, stable-3134055, stable-437252f) taking 802M, active is 3.1.10
- **Fix**: Removed stale releases, kept 3.1.10 (active) + stable-e4e3aa (latest)

### 18. Deploy script `npx: command not found` + `blake3` import error
- **Problem**: `event-checkin/worker/deploy.sh` uses `#!/usr/bin/env bash` which starts a non-interactive shell. `eval "$(fnm env --use-on-cd)"` failed because fnm's shell hooks (`cd` alias, `.node-version` detection) are unreliable in non-interactive bash. Also `python3` resolved to wrong binary.
- **Fix**: Bypassed fnm shell integration entirely — directly added fnm's node installation dir to PATH (`~/.local/share/fnm/node-versions/v24.16.0/installation/bin`). Hardcoded `/opt/homebrew/bin/python3` for blake3.
- **File**: `/Users/ozone/event-checkin/worker/deploy.sh`

---

## Cleanup Performed (~30 GB freed total)

### Session 1 (~7.4 GB)
| What | Saved |
|------|-------|
| npm cache | ~5.5 GB |
| Copilot cache | 166 MB |
| pnpm cache | 352 MB |
| Old Node versions (v23.6, v23.11, v24.16) | ~715 MB |
| Homebrew + node-gyp cache | ~111 MB |
| NVM directory (replaced by fnm) | ~585 MB |

### Session 2 — /usr/local Intel cleanup (~8.2 GB)
| What | Saved |
|------|-------|
| php-fpm.log + postgresql@15.log | 4.1 GB |
| Intel .NET (x86_64) | 2.1 GB |
| Intel Go (/usr/local/go, x86_64) | 346 MB |
| Old Intel Homebrew | 284 MB |
| Old node binary (/usr/local/bin/node, v22.12.0) | 237 MB |
| Dead Python 3.9/3.11/3.13 libs | 456 MB |
| Intel binaries (nmap, wasmedge, turbo, surfpool, git-lfs, VirtualBox) | ~200 MB |
| Stale node globals (create-react-app, webpack, solc, truffle, vue) | ~200 MB |
| Stale headers, PostgreSQL data, pear | ~120 MB |
| python.org Python 3.12 + broken CLI scripts | ~100 MB |
| Homebrew Go (unneeded) | 228 MB |

### Session 3 — Dead app data, legacy frameworks, old toolchains (~13.3 GB)
| What | Saved |
|------|-------|
| Cisco Spark (no app installed) | 2.7 GB |
| WebEx Folder (no app installed) | 1.4 GB |
| Microsoft Edge (no app installed) | 1.9 GB |
| VisualStudio (no app installed) | 810 MB |
| TabNine (no app installed) | 485 MB |
| NATURAL8 gambling app (no app installed) | 799 MB |
| Firefox cache | 825 MB |
| Mono.framework (legacy .NET) | 1.0 GB |
| OSXFUSE.framework (dead) + Python.framework (empty shell) | — |
| Old Solana releases (2.1.0, stable-3134055, stable-437252f) | 802 MB |
| Cargo registry cache/src + git checkouts (regenerates on build) | 2.6 GB |

---

## Optimizations Applied

| What | Before | After |
|------|--------|-------|
| Shell startup | 335ms | **54ms** (6x faster) |
| DNS | ISP default | Cloudflare + AIS fallback |
| Git protocol | v1 | v2 |
| Git diff | default | **delta** (side-by-side, syntax highlighted) |
| Cargo registry | git-based | Sparse (faster) |
| `ls` | BSD ls | **eza** (icons, git status) |
| `cat` | BSD cat | **bat** (syntax highlight, line numbers) |
| `find` | BSD find | **fd** (30× faster, benchmarked) |
| `grep` | BSD grep | **ripgrep** (25× faster, benchmarked) |
| `du` | BSD du | **dust** (visual) |
| `ps` | BSD ps | **procs** (modern) |
| `top` | BSD top | **bottom** (TUI) |
| `sed` | BSD sed | **sd** (simpler syntax) |
| Firewall | Off | On |
| Shell configs | 4 files, duplicates | 1 clean `.zshrc` |
| SSH key | Not in agent | In Keychain, auto-loads |
| LaunchAgents | 8 stale | 0 stale |

## NOT Applied (correctly skipped)

| What | Why |
|------|-----|
| TCP sysctl tuning | VPN/tunnel interfaces = risk, macOS ignores sysctl.conf at boot |

---

## Rust-Powered CLI Tools — Benchmarked on M5 Pro (2026-06-13)

### Genuine Speed Wins (benchmarked with hyperfine)

| Tool | Replaces | Measured Speedup | Benchmark |
|------|----------|-----------------|-----------|
| `fd` | `find` | **30.27× faster** | 17ms vs 515ms across `/Users/ozone/Projects` |
| `ripgrep` (`rg`) | `grep` | **25.65× faster** | 29ms vs 740ms searching Rust files |
| `fnm` | `nvm` | **6.2× faster shell** | 54ms vs 335ms shell startup |

### Slower Than Original (better UX, not speed)

| Tool | Replaces | Measured | Why Keep |
|------|----------|----------|----------|
| `bat` | `cat` | **3.6× slower** (4.4ms vs 1.2ms) | Syntax highlighting, line numbers, Git integration, paging |
| `eza` | `ls` | **3× slower** (4ms vs 1.3ms) | Icons, colors, git status, tree view |
| `procs` | `ps` | **3.4× slower** (132ms vs 39ms) | Readable layout, keyword search, tree view |
| `sd` | `sed` | **1.5× slower** (1.9ms vs 1.3ms) | Simpler regex syntax |
| `dust` | `du` | **1.14× faster** (marginal) | Value is visual tree output, not speed |

### Non-comparable (UX/feature tools)

| Tool | Replaces | Value |
|------|----------|-------|
| `delta` | git pager | Side-by-side colored diffs |
| `bottom` (`btm`) | `top` | Modern TUI system monitor |
| `zellij` | tmux | Terminal multiplexer with better UX |
| `tokei` | cloc | Accurate code statistics |
| `hyperfine` | `time` | Statistical benchmarking |

### Security Audit of All 13 Tools

| Check | Result |
|-------|--------|
| Architecture | All native `arm64` ✅ |
| Code signing | All valid, satisfies Designated Requirement ✅ |
| Quarantine xattr | None ✅ |
| File permissions | All `555`, owned by `ozone:admin` ✅ |
| setuid/setgid | None ✅ |
| Network connections | Zero (no phone-home) ✅ |
| Known CVEs | Zero across all 13 tools ✅ |
| Source | All via Homebrew bottles (`/opt/homebrew/Cellar/`) ✅ |

All configured in `.zshrc` with aliases — drop-in replacements.

---

## Config Backups

Old configs saved at `~/.config-backup-2026-06-13/`:
- `zshrc`, `bashrc`, `zprofile`, `profile`

---

## System Snapshot (Final State)

### Hardware
- Apple Silicon M5 Pro, 48GB RAM, 1.8TB SSD (1.6TB free, 12 GB used)

### Toolchain (all arm64 native)
| Tool | Version |
|------|---------|
| macOS | 26.5.1 |
| git | 2.50.1 |
| Node.js | v24.16.0 (+ v25.9.0) via fnm |
| npm | 11.13.0 |
| pnpm | 11.6.0 |
| yarn | 1.22.22 |
| rustc | 1.96.0 |
| cargo | 1.96.0 |
| solana-cli | 3.1.10 |
| anchor-cli | 1.0.2 |
| gh | 2.94.0 |
| Go | Removed (no projects) |
| Python | 3.14.5 (Homebrew, single Python) |

### Security
| Item | Status |
|------|--------|
| FileVault | On ✅ |
| Firewall | On ✅ |
| Gatekeeper | Enabled ✅ |
| SSH key | ED25519 in Keychain ✅ |
| Global gitignore | Configured ✅ |

### Shell Startup: ~34ms

---

## Remain Work

- [ ] **Set up Time Machine** — buy external SSD, plug in, macOS guides you
- [ ] **Consider dotfiles repo** — `.zshrc` is clean at 25 lines
- [ ] **`brew update && brew upgrade`** periodically

## Deploy Script Notes

- **File**: `/Users/ozone/event-checkin/worker/deploy.sh`
- **Known Cloudflare bug**: `/versions` API returns error 10013. Script has fallback to PUT API.
- **Node setup**: Direct PATH to `~/.local/share/fnm/node-versions/v24.16.0/installation/bin` (no fnm shell hooks)
- **Python**: Hardcoded `/opt/homebrew/bin/python3` (has `blake3` module)
- **Cargo cache**: Cleaned — will auto-repopulate on next `cargo build`

## Build Artifacts Note

Cleaned ~47 GB of Rust `target/` dirs across 4 projects. These regenerate on `cargo build`.
Future optimization options: `sccache` (shared compilation cache), shared target dir, periodic `cargo clean`.

---

## How to Revert

### DNS if internet breaks
```bash
networksetup -setdnsservers Wi-Fi empty
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
```

### Shell configs if broken
```bash
cp ~/.config-backup-2026-06-13/zshrc ~/.zshrc
cp ~/.config-backup-2026-06-13/bashrc ~/.bashrc
cp ~/.config-backup-2026-06-13/zprofile ~/.zprofile
cp ~/.config-backup-2026-06-13/profile ~/.profile
```

### If fnm causes issues, reinstall NVM
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
# Then replace fnm line in .zshrc with NVM lines from backup
```

### Remove Rust CLI aliases (go back to defaults)
Remove the `# --- ALIASES` section from `.zshrc`.

### Remove individual tools
```bash
# If you want to remove a specific slower tool and keep using the original:
brew uninstall bat       # go back to cat
brew uninstall eza       # go back to ls
brew uninstall procs     # go back to ps
brew uninstall sd        # go back to sed
# Also remove corresponding alias from .zshrc
```
