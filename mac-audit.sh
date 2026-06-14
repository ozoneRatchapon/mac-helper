#!/usr/bin/env bash
# mac-audit.sh — Audit your macOS developer machine for health, waste, and optimization opportunities.
# Usage: curl -s <url>/mac-audit.sh | bash
#        or: ./mac-audit.sh
#
# Safe: read-only, no modifications. Reports what to clean, doesn't clean it.
# Requires: bash 3.2+ (macOS default), no external dependencies

set -uo pipefail

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  DIM='\033[2m'
  RESET='\033[0m'
else
  RED='' GREEN='' YELLOW='' CYAN='' BOLD='' DIM='' RESET=''
fi

# ── Helpers ──

header() {
  echo ""
  echo -e "${BOLD}${CYAN}$1${RESET}"
  echo -e "${DIM}$(printf '─%.0s' $(seq 1 60))${RESET}"
}

ok()   { echo -e "  ${GREEN}✅${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠️${RESET}  $1"; }
bad()  { echo -e "  ${RED}❌${RESET} $1"; }
info() { echo -e "  ${DIM}$1${RESET}"; }

bytes_to_human() {
  local bytes=$1
  if [ "$bytes" -ge 1073741824 ]; then
    echo "$(echo "scale=1; $bytes / 1073741824" | bc) GB"
  elif [ "$bytes" -ge 1048576 ]; then
    echo "$(echo "scale=1; $bytes / 1048576" | bc) MB"
  elif [ "$bytes" -ge 1024 ]; then
    echo "$(echo "scale=0; $bytes / 1024" | bc) KB"
  else
    echo "${bytes} B"
  fi
}

# ── Banner ──

echo ""
echo -e "${BOLD}🔍 Mac Audit Report${RESET}"
echo -e "${DIM}$(date '+%Y-%m-%d %H:%M') · $(sw_vers -productName) $(sw_vers -productVersion) · $(uname -m)${RESET}"

# ── 1. Disk ──

header "Disk Usage"

DISK_INFO=$(df -h / | tail -1)
DISK_USED=$(echo "$DISK_INFO" | awk '{print $3}')
DISK_AVAIL=$(echo "$DISK_INFO" | awk '{print $4}')
DISK_TOTAL=$(echo "$DISK_INFO" | awk '{print $2}')
DISK_PCT=$(echo "$DISK_INFO" | awk '{print $5}' | tr -d '%')

echo "  ${DISK_USED} used / ${DISK_TOTAL} total (${DISK_AVAIL} free)"

if [ "$DISK_PCT" -gt 90 ]; then
  bad "Disk is ${DISK_PCT}% full — clean up urgently"
elif [ "$DISK_PCT" -gt 75 ]; then
  warn "Disk is ${DISK_PCT}% full — consider cleaning up"
else
  ok "Disk usage at ${DISK_PCT}%"
fi

# ── 2. Architecture ──

header "Architecture"

if [ "$(uname -m)" = "arm64" ]; then
  ok "Apple Silicon (arm64)"
else
  warn "Intel (x86_64) — consider migrating to Apple Silicon"
fi

# Check Homebrew location
if [ -d "/opt/homebrew" ]; then
  ok "Homebrew at /opt/homebrew (Apple Silicon native)"
elif [ -d "/usr/local/Homebrew" ]; then
  if [ "$(uname -m)" = "arm64" ]; then
    bad "Homebrew at /usr/local (Intel) — should be /opt/homebrew on Apple Silicon"
  else
    ok "Homebrew at /usr/local (Intel, correct)"
  fi
else
  warn "Homebrew not found"
fi

# Check /usr/local for Intel leftovers
if [ "$(uname -m)" = "arm64" ] && [ -d "/usr/local" ]; then
  USR_LOCAL_SIZE=$(du -sm /usr/local 2>/dev/null | awk '{print $1}')
  if [ "$USR_LOCAL_SIZE" -gt 500 ]; then
    warn "/usr/local/ is $(bytes_to_human "$((USR_LOCAL_SIZE * 1048576))") — likely Intel leftovers on Apple Silicon"
  fi
fi

# ── 3. Shell Performance ──

header "Shell Performance"

# Measure zsh startup (subshell to avoid env pollution)
STARTUP_MS=$( (TIMEFORMAT='%R'; time zsh -i -c exit 2>/dev/null) 2>&1 | tail -1 | awk '{printf "%.0f", $1 * 1000}')

if [ -n "$STARTUP_MS" ] && [ "$STARTUP_MS" -gt 0 ] 2>/dev/null; then
  if [ "$STARTUP_MS" -gt 500 ]; then
    bad "Shell startup: ${STARTUP_MS}ms — slow! Check .zshrc for heavy init (nvm, pyenv, etc.)"
  elif [ "$STARTUP_MS" -gt 200 ]; then
    warn "Shell startup: ${STARTUP_MS}ms — could be faster"
  else
    ok "Shell startup: ${STARTUP_MS}ms"
  fi
else
  info "Could not measure shell startup"
fi

# Check which node manager
if command -v fnm &>/dev/null; then
  ok "Node manager: fnm (fast)"
elif [ -s "$HOME/.nvm/nvm.sh" ]; then
  warn "Node manager: nvm (slow — consider fnm for 8× faster shell startup)"
fi

# ── 4. Runtimes ──

header "Runtimes"

# Python
PYTHON_PATHS=$(which -a python3 2>/dev/null | sort -u)
PYTHON_COUNT=$(echo "$PYTHON_PATHS" | wc -l | tr -d ' ')
PYTHON_PATH=$(which python3 2>/dev/null)
if [ -n "$PYTHON_PATH" ]; then
  PYTHON_VER=$(python3 --version 2>/dev/null)
  echo "  Python: ${PYTHON_VER} (${PYTHON_PATH})"
  if [ "$PYTHON_COUNT" -gt 1 ]; then
    warn "Multiple python3 found:"
    echo "$PYTHON_PATHS" | while read p; do echo "    $p"; done
  else
    ok "Single Python runtime"
  fi
else
  warn "No python3 found"
fi

# Node (check fnm-managed paths too)
eval "$(fnm env 2>/dev/null)"
NODE_PATHS=$(which -a node 2>/dev/null | sort -u)
NODE_COUNT=$(echo "$NODE_PATHS" | wc -l | tr -d ' ')
NODE_PATH=$(which node 2>/dev/null)
if [ -n "$NODE_PATH" ]; then
  NODE_VER=$(node --version 2>/dev/null)
  echo "  Node: ${NODE_VER} (${NODE_PATH})"
  if [ "$NODE_COUNT" -gt 1 ]; then
    warn "Multiple node found — may cause conflicts"
  else
    ok "Single Node runtime"
  fi
else
  warn "No node found"
fi

# Go
GO_PATH=$(which go 2>/dev/null)
if [ -n "$GO_PATH" ]; then
  GO_VER=$(go version 2>/dev/null | awk '{print $3}')
  echo "  Go: ${GO_VER} (${GO_PATH})"
  GO_PROJECTS=$(find ~ -maxdepth 3 -name "go.mod" -not -path "*/.*" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$GO_PROJECTS" -eq 0 ]; then
    warn "Go installed but no go.mod found — consider removing if unused"
  fi
fi

# Rust
RUSTC_PATH=$(which rustc 2>/dev/null)
if [ -n "$RUSTC_PATH" ]; then
  RUST_VER=$(rustc --version 2>/dev/null)
  echo "  Rust: ${RUST_VER}"
  TOOLCHAINS=$(rustup toolchain list 2>/dev/null | wc -l | tr -d ' ')
  if [ "$TOOLCHAINS" -gt 3 ]; then
    warn "${TOOLCHAINS} Rust toolchains installed — remove unused with 'rustup toolchain uninstall <name>'"
  fi
fi

# ── 5. Dead App Data ──

header "Dead App Data"

DEAD_TOTAL=0
DEAD_COUNT=0

# Check specific known dead apps (bash 3 compatible — no associative arrays)
check_dead_app() {
  local app_name="$1"
  local app_check="$2"
  local app_dir="$HOME/Library/Application Support/${app_name}"

  [ -d "$app_dir" ] || return

  # Check if app is installed
  if mdfind "kMDItemKind == 'Application'" 2>/dev/null | grep -qi "$app_check"; then
    return
  fi

  dir_size=$(du -sm "$app_dir" 2>/dev/null | awk '{print $1}')
  if [ -n "$dir_size" ] && [ "$dir_size" -gt 10 ]; then
    warn "${app_name}: $(bytes_to_human "$((dir_size * 1048576))") (app not installed)"
    DEAD_TOTAL=$((DEAD_TOTAL + dir_size))
    DEAD_COUNT=$((DEAD_COUNT + 1))
  fi
}

check_dead_app "Cisco Spark" "webex"
check_dead_app "Microsoft Edge" "edge"
check_dead_app "VisualStudio" "visual studio"
check_dead_app "TabNine" "tabnine"
check_dead_app "WebEx Folder" "webex"
check_dead_app "NATURAL8" "natural8"
check_dead_app "TeamViewer" "teamviewer"
check_dead_app "Skype" "skype"
check_dead_app "Zoom" "zoom.us"
check_dead_app "Notion" "notion"

# Generic scan: large Application Support dirs (>500MB) without matching app
while IFS= read -r dir; do
  [ -z "$dir" ] && continue
  app_name=$(basename "$dir")
  dir_size=$(du -sm "$dir" 2>/dev/null | awk '{print $1}')
  [ -z "$dir_size" ] && continue
  [ "$dir_size" -lt 500 ] && continue

  # Skip known apps + system dirs (avoid false positives)
  # Note: Slack/Code/Cursor removed — they get caught as orphans if uninstalled.
  # 'Caches' is a system dir literally named Caches, not an app.
  case "$app_name" in
    "Microsoft"|"Google"|"Firefox"|"discord"|"Steam"|"Zed"|"BraveSoftware"|"Trae"|"Adobe"|"Apple"|"Caches"|"com.apple"*)
      continue ;;
  esac

  # Already reported above
  case "$app_name" in
    "Cisco Spark"|"Microsoft Edge"|"VisualStudio"|"TabNine"|"WebEx Folder"|"NATURAL8")
      continue ;;
  esac

  if ! mdfind "kMDItemKind == 'Application'" 2>/dev/null | grep -qi "$app_name"; then
    warn "${app_name}: $(bytes_to_human "$((dir_size * 1048576))") (no matching app found)"
    DEAD_TOTAL=$((DEAD_TOTAL + dir_size))
    DEAD_COUNT=$((DEAD_COUNT + 1))
  fi
done < <(ls -d ~/Library/Application\ Support/*/ 2>/dev/null)

if [ "$DEAD_COUNT" -eq 0 ]; then
  ok "No dead app data found"
else
  echo ""
  bad "${DEAD_COUNT} orphaned dirs totaling $(bytes_to_human "$((DEAD_TOTAL * 1048576))")"
fi

# ── 6. Build Artifacts ──

header "Build Artifacts"

TARGET_TOTAL=0
TARGET_COUNT=0

while IFS= read -r target_dir; do
  [ -z "$target_dir" ] && continue
  dir_size=$(du -sm "$target_dir" 2>/dev/null | awk '{print $1}')
  if [ -n "$dir_size" ] && [ "$dir_size" -gt 100 ]; then
    parent=$(dirname "$target_dir")
    project=$(basename "$parent")
    warn "${project}/target/: $(bytes_to_human "$((dir_size * 1048576))")"
    TARGET_TOTAL=$((TARGET_TOTAL + dir_size))
    TARGET_COUNT=$((TARGET_COUNT + 1))
  fi
done < <(find ~ -maxdepth 5 -name "target" -type d 2>/dev/null | grep -v ".rustup" | grep -v "Library" | grep -v "mac\ helper" | head -20)

if [ "$TARGET_COUNT" -eq 0 ]; then
  ok "No large Rust target/ dirs found"
else
  echo ""
  info "${TARGET_COUNT} Rust target/ dirs totaling $(bytes_to_human "$((TARGET_TOTAL * 1048576))")"
  info "Clean with: cargo clean --manifest-dir <path>/Cargo.toml"
fi

# Node modules
NODE_MODULES_TOTAL=0
NODE_MODULES_COUNT=0

while IFS= read -r nm_dir; do
  [ -z "$nm_dir" ] && continue
  dir_size=$(du -sm "$nm_dir" 2>/dev/null | awk '{print $1}')
  if [ -n "$dir_size" ] && [ "$dir_size" -gt 100 ]; then
    parent=$(dirname "$nm_dir")
    project=$(basename "$parent")
    warn "${project}/node_modules/: $(bytes_to_human "$((dir_size * 1048576))")"
    NODE_MODULES_TOTAL=$((NODE_MODULES_TOTAL + dir_size))
    NODE_MODULES_COUNT=$((NODE_MODULES_COUNT + 1))
  fi
done < <(find ~ -maxdepth 4 -name "node_modules" -type d 2>/dev/null | grep -v ".fnm" | grep -v ".nvm" | grep -v "Library" | head -20)

if [ "$NODE_MODULES_COUNT" -gt 0 ]; then
  info "${NODE_MODULES_COUNT} large node_modules/ totaling $(bytes_to_human "$((NODE_MODULES_TOTAL * 1048576))")"
fi

# ── 7. Caches ──

header "Caches"

CACHE_TOTAL=$(du -sm ~/Library/Caches 2>/dev/null | awk '{print $1}')
if [ -n "$CACHE_TOTAL" ] && [ "$CACHE_TOTAL" -gt 500 ]; then
  warn "~/Library/Caches/: $(bytes_to_human "$((CACHE_TOTAL * 1048576))")"
  du -sm ~/Library/Caches/*/ 2>/dev/null | sort -rn | head -5 | while read size name; do
    echo "    $(bytes_to_human "$((size * 1048576))")  $(basename "$name")"
  done
else
  ok "~/Library/Caches/: $(bytes_to_human "$((CACHE_TOTAL * 1048576))")"
fi

# Cargo cache
if [ -d "$HOME/.cargo/registry" ]; then
  CARGO_CACHE=$(du -sm ~/.cargo/registry 2>/dev/null | awk '{print $1}')
  if [ -n "$CARGO_CACHE" ] && [ "$CARGO_CACHE" -gt 500 ]; then
    warn "Cargo registry cache: $(bytes_to_human "$((CARGO_CACHE * 1048576))") (regenerates on cargo build)"
  fi
fi

# ── 8. Security ──

header "Security"

# FileVault
FV_STATUS=$(fdesetup status 2>/dev/null | head -1)
if echo "$FV_STATUS" | grep -q "On"; then
  ok "FileVault: On"
else
  bad "FileVault: Off — enable in System Settings > Privacy & Security"
fi

# Firewall
FW_STATUS=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null)
if echo "$FW_STATUS" | grep -q "enabled"; then
  ok "Firewall: On"
else
  bad "Firewall: Off — enable with: sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on"
fi

# SSH key
if [ -f "$HOME/.ssh/id_ed25519" ] || [ -f "$HOME/.ssh/id_rsa" ]; then
  ok "SSH key: Found"
  if ssh-add -l 2>/dev/null | grep -q "ED25519\|RSA"; then
    ok "SSH key: In agent"
  else
    warn "SSH key: Not in agent — run: ssh-add --apple-use-keychain ~/.ssh/id_ed25519"
  fi
else
  warn "No SSH key found — generate with: ssh-keygen -t ed25519"
fi

# Global gitignore
if git config --global core.excludesfile &>/dev/null; then
  ok "Global .gitignore: Configured"
else
  warn "No global .gitignore — create ~/.gitignore_global and run: git config --global core.excludesfile ~/.gitignore_global"
fi

# ── 9. LaunchAgents ──

header "LaunchAgents"

STALE_AGENTS=0
for agent in ~/Library/LaunchAgents/*.plist; do
  [ -f "$agent" ] || continue
  agent_name=$(basename "$agent")
  binary=$(defaults read "$agent" ProgramArguments 2>/dev/null | head -2 | tail -1 | tr -d '"' | sed 's/^[[:space:]]*//')
  if [ -n "$binary" ] && [ ! -f "$binary" ]; then
    warn "${agent_name} — binary not found: ${binary}"
    STALE_AGENTS=$((STALE_AGENTS + 1))
  fi
done

if [ "$STALE_AGENTS" -eq 0 ]; then
  ok "No stale LaunchAgents"
fi

# ── 10. Login Items ──
# Catches the root cause of sleep leaks (e.g. LINE auto-starting and holding
# 46 PreventUserIdleSystemSleep assertions). Best run interactively — the
# weekly LaunchAgent may report 'none readable' if bash lacks System Events
# automation permission.

header "Login Items"

LOGIN_ITEMS=$(osascript -e 'tell application "System Events" to get the name of every login item' 2>/dev/null)

if [ -z "$LOGIN_ITEMS" ]; then
  ok "No login items (or none readable)"
else
  items_normalized=$(echo "$LOGIN_ITEMS" | tr ',' '\n' | sed 's/^ *//;s/ *$//')
  item_count=$(echo "$items_normalized" | grep -c '.')

  echo "  ${item_count} login item(s):"
  echo "$items_normalized" | while read item; do
    [ -z "$item" ] && continue
    echo "    • ${item}"
  done

  # Flag known leak-prone apps (exact match)
  if echo "$items_normalized" | grep -qx "LINE"; then
    warn "LINE in Login Items — known sleep-leak culprit (46 assertions). Remove via System Settings → General → Login Items."
  fi

  if [ "$item_count" -gt 5 ]; then
    warn "${item_count} login items — many auto-start apps slow boot time"
  fi
fi

# ── 11. CLI Tools ──

header "CLI Tools (Rust alternatives)"

MODERN_TOOLS="fd rg bat eza dust sd procs delta btm hyperfine"
REPLACES="fd:find rg:grep bat:cat eza:ls dust:du sd:sed procs:ps delta:git-pager btm:top hyperfine:time"

INSTALLED_MODERN=0
MISSING_TOOLS=""

for tool in $MODERN_TOOLS; do
  if command -v "$tool" &>/dev/null; then
    INSTALLED_MODERN=$((INSTALLED_MODERN + 1))
  else
    replaces=""
    for pair in $REPLACES; do
      key="${pair%%:*}"
      val="${pair##*:}"
      if [ "$key" = "$tool" ]; then replaces="$val"; break; fi
    done
    MISSING_TOOLS="$MISSING_TOOLS
    $tool (replaces $replaces)"
  fi
done

echo "  Modern tools: ${INSTALLED_MODERN}/10 installed"
if [ -n "$MISSING_TOOLS" ]; then
  echo "  Missing:$MISSING_TOOLS"
  info "Install all: brew install fd ripgrep bat eza dust sd procs bottom delta hyperfine"
fi

# Check architecture of installed tools
WRONG_ARCH=0
for tool in $MODERN_TOOLS; do
  tool_path=$(which "$tool" 2>/dev/null)
  if [ -n "$tool_path" ]; then
    if file "$tool_path" 2>/dev/null | grep -q "x86_64" && [ "$(uname -m)" = "arm64" ]; then
      warn "${tool} is x86_64 — should be arm64"
      WRONG_ARCH=$((WRONG_ARCH + 1))
    fi
  fi
done

if [ "$WRONG_ARCH" -eq 0 ] && [ "$INSTALLED_MODERN" -gt 0 ]; then
  ok "All installed tools are native $(uname -m)"
fi

# ── 12. PATH Issues ──

header "PATH Health"

# Check for duplicates
DUPES=$(echo "$PATH" | tr ':' '\n' | sort | uniq -d)
if [ -n "$DUPES" ]; then
  warn "Duplicate PATH entries:"
  echo "$DUPES" | while read p; do echo "    $p"; done
else
  ok "No duplicate PATH entries"
fi

# Check for broken paths
echo "$PATH" | tr ':' '\n' | while read p; do
  if [ -n "$p" ] && [ ! -d "$p" ]; then
    warn "PATH contains non-existent dir: $p"
  fi
done

# Check for Intel-era paths on Apple Silicon
if [ "$(uname -m)" = "arm64" ]; then
  echo "$PATH" | tr ':' '\n' | while read p; do
    case "$p" in
      /Library/Frameworks/Python.framework/*)
        warn "Intel Python in PATH: $p" ;;
      /usr/local/go/bin)
        warn "Intel Go in PATH: $p" ;;
      */.nvm/versions/*)
        warn "NVM in PATH (consider switching to fnm): $p" ;;
    esac
  done
fi

# ── 13. Downloads Cleanup ──

header "Downloads Cleanup"

DMG_FOUND=0
find ~/Downloads -maxdepth 1 \( -name "*.dmg" -o -name "*.pkg" -o -name "*.zip" \) -type f 2>/dev/null | while read f; do
  size=$(stat -f%z "$f" 2>/dev/null || echo 0)
  echo "  $(bytes_to_human "$size")  $(basename "$f")"
  DMG_FOUND=$((DMG_FOUND + 1))
done

if [ "$DMG_FOUND" -eq 0 ]; then
  ok "No installer files in Downloads"
fi

# ── Summary ──

header "Summary"

TOTAL_WASTE=$((DEAD_TOTAL + TARGET_TOTAL + NODE_MODULES_TOTAL))

if [ "$TOTAL_WASTE" -gt 0 ]; then
  echo ""
  echo -e "  ${BOLD}Potential space to reclaim: $(bytes_to_human "$((TOTAL_WASTE * 1048576))")${RESET}"
  echo ""
  if [ "$DEAD_TOTAL" -gt 0 ]; then
    echo "  • Dead app data: $(bytes_to_human "$((DEAD_TOTAL * 1048576))")"
  fi
  if [ "$TARGET_TOTAL" -gt 0 ]; then
    echo "  • Rust target/ dirs: $(bytes_to_human "$((TARGET_TOTAL * 1048576))") (regenerates on build)"
  fi
  if [ "$NODE_MODULES_TOTAL" -gt 0 ]; then
    echo "  • node_modules/: $(bytes_to_human "$((NODE_MODULES_TOTAL * 1048576))")"
  fi
else
  ok "No major waste found — your Mac is clean!"
fi

echo ""
echo -e "${DIM}Re-run anytime: ./mac-audit.sh${RESET}"
echo ""
