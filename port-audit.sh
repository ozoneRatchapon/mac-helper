#!/usr/bin/env bash
# port-audit.sh — Audit which network ports your Mac is listening on, and which are exposed to the LAN.
# Usage: ./port-audit.sh              # audit + compare against saved baseline
#        ./port-audit.sh --save       # save current state as the new baseline
#        ./port-audit.sh --scan       # also verify from the LAN side with rustscan (if installed)
#
# Safe: read-only, no modifications. Reports exposure, doesn't change it.
# Requires: bash 3.2+ (macOS default), lsof (built-in). rustscan optional, for --scan.

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

BASELINE_DIR="$HOME/.cache/mac-helper"
BASELINE="$BASELINE_DIR/port-baseline.txt"

SAVE_BASELINE=0
DO_SCAN=0
for arg in "$@"; do
  case "$arg" in
    --save) SAVE_BASELINE=1 ;;
    --scan) DO_SCAN=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 1 ;;
  esac
done

# Why a port matters. Returns "RISK|explanation", or empty if unremarkable.
# Classified by service first (a debug port is dangerous on any port number),
# then by well-known port. Keep the risky cases loud and everything else quiet.
classify() {
  local port=$1 proc=$2
  case "$proc" in
    AnyDesk*|TeamViewer*|RustDesk*|*VNC*|ScreenSharing*|*rdesktop*)
      echo "HIGH|remote desktop — full control of this Mac if credentials leak"; return ;;
  esac
  case "$port" in
    22)    echo "HIGH|SSH — remote shell" ;;
    5900)  echo "HIGH|VNC / Screen Sharing — remote control" ;;
    9222|9229)
           echo "HIGH|remote debugging (Chrome DevTools / Node inspector) — grants full browser or process control, incl. cookies and sessions" ;;
    3306|5432|27017|6379|5984|9200|11211)
           echo "HIGH|database / datastore — usually should never leave loopback" ;;
    139|445)
           echo "MED|SMB file sharing" ;;
    548)   echo "MED|AFP file sharing" ;;
    3000|3001|4200|5173|8000|8080|8081|8888)
           echo "MED|dev server — often unauthenticated" ;;
    5000|7000)
           echo "LOW|AirPlay Receiver (macOS built-in)" ;;
    *)     echo "" ;;
  esac
}

# ── Banner ──

echo ""
echo -e "${BOLD}🔌 Port Exposure Report${RESET}"
echo -e "${DIM}$(date '+%Y-%m-%d %H:%M') · $(sw_vers -productName) $(sw_vers -productVersion) · $(uname -m)${RESET}"

# ── 1. Firewall posture ──

header "Firewall Posture"

FW_STATUS=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null)
if echo "$FW_STATUS" | grep -q "enabled"; then
  ok "Application firewall: On"
else
  bad "Application firewall: Off — sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on"
fi

STEALTH=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getstealthmode 2>/dev/null)
if echo "$STEALTH" | grep -q "enabled"; then
  ok "Stealth mode: On (does not answer probes)"
else
  warn "Stealth mode: Off — this Mac answers ping and shows up in network discovery"
  info "Enable: sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode on"
fi

LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
if [ -n "$LAN_IP" ]; then
  info "LAN address: $LAN_IP"
else
  info "No active Wi-Fi/Ethernet address — exposed ports are unreachable right now"
fi

# ── 2. Collect listening sockets ──
#
# lsof prints one row per socket, so the same service shows up twice (IPv4 + IPv6).
# Dedup on port+process, and split by bind address: "*" or a routable IP is reachable
# from the network, 127.0.0.1 / [::1] is not.

SOCKETS=$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $1, $2, $(NF-1)}')

EXPOSED=""   # port<TAB>proc<TAB>pid
LOOPBACK=""
# Dedup matches must anchor to a line start, otherwise port 80 is
# swallowed by an existing 8080 row for the same process.
NL='
'

while read -r proc pid addr; do
  [ -z "${addr:-}" ] && continue
  port="${addr##*:}"
  bind="${addr%:*}"
  case "$bind" in
    127.0.0.1|\[::1\]|localhost) bucket=LOOPBACK ;;
    *)                           bucket=EXPOSED ;;
  esac
  # lsof escapes spaces in process names as \x20; make them readable
  proc=$(printf '%s' "$proc" | sed 's/\\x20/ /g')
  line="$port	$proc	$pid"
  if [ "$bucket" = EXPOSED ]; then
    case "$NL$EXPOSED" in *"$NL$port	$proc	"*) continue ;; esac
    EXPOSED="$EXPOSED$line
"
  else
    case "$NL$LOOPBACK" in *"$NL$port	$proc	"*) continue ;; esac
    LOOPBACK="$LOOPBACK$line
"
  fi
done <<EOF
$SOCKETS
EOF

EXPOSED=$(printf '%s' "$EXPOSED" | grep -v '^$' | sort -n)
LOOPBACK=$(printf '%s' "$LOOPBACK" | grep -v '^$' | sort -n)

# ── 3. Exposed to the network ──

header "Reachable From The Network"

HIGH_COUNT=0
EXPOSED_COUNT=0

if [ -z "$EXPOSED" ]; then
  ok "Nothing listening on a routable interface — minimal attack surface"
else
  while IFS='	' read -r port proc pid; do
    [ -z "${port:-}" ] && continue
    EXPOSED_COUNT=$((EXPOSED_COUNT + 1))
    verdict=$(classify "$port" "$proc")
    risk="${verdict%%|*}"
    why="${verdict#*|}"
    case "$risk" in
      HIGH) bad  "$(printf '%-6s' "$port") $proc ${DIM}(pid $pid)${RESET}"
            info "   $why"
            HIGH_COUNT=$((HIGH_COUNT + 1)) ;;
      MED)  warn "$(printf '%-6s' "$port") $proc ${DIM}(pid $pid)${RESET}"
            info "   $why" ;;
      LOW)  info "$(printf '%-6s' "$port") $proc — $why" ;;
      *)    warn "$(printf '%-6s' "$port") $proc ${DIM}(pid $pid)${RESET}" ;;
    esac
  done <<EOF
$EXPOSED
EOF
fi

# ── 4. Loopback only ──

header "Loopback Only (not reachable from the network)"

LOOPBACK_COUNT=0
LOCAL_DEBUG=""

if [ -z "$LOOPBACK" ]; then
  info "None"
else
  while IFS='	' read -r port proc pid; do
    [ -z "${port:-}" ] && continue
    LOOPBACK_COUNT=$((LOOPBACK_COUNT + 1))
    case "$port" in
      9222|9229) LOCAL_DEBUG="$LOCAL_DEBUG $port ($proc)" ;;
    esac
    info "$(printf '%-6s' "$port") $proc"
  done <<EOF
$LOOPBACK
EOF
fi

if [ -n "$LOCAL_DEBUG" ]; then
  echo ""
  warn "Debug ports open on loopback:$LOCAL_DEBUG"
  info "   Safe from the network, but ANY local process — including an untrusted"
  info "   npm/pip dependency — can drive the browser and read your sessions."
fi

# ── 5. Baseline diff ──

header "Changes Since Last Audit"

CURRENT=$(printf '%s\n%s\n' "$EXPOSED" "$LOOPBACK" | grep -v '^$' | cut -f1,2 | sort -u)

if [ "$SAVE_BASELINE" -eq 1 ]; then
  mkdir -p "$BASELINE_DIR"
  printf '%s\n' "$CURRENT" > "$BASELINE"
  ok "Baseline saved to $BASELINE"
elif [ ! -f "$BASELINE" ]; then
  info "No baseline yet — run ./port-audit.sh --save to record the current state"
else
  ADDED=$(comm -13 "$BASELINE" <(printf '%s\n' "$CURRENT"))
  REMOVED=$(comm -23 "$BASELINE" <(printf '%s\n' "$CURRENT"))
  if [ -z "$ADDED" ] && [ -z "$REMOVED" ]; then
    ok "No change since baseline"
  else
    if [ -n "$ADDED" ]; then
      while IFS='	' read -r port proc; do
        [ -z "${port:-}" ] && continue
        warn "NEW    $(printf '%-6s' "$port") $proc"
      done <<EOF
$ADDED
EOF
    fi
    if [ -n "$REMOVED" ]; then
      while IFS='	' read -r port proc; do
        [ -z "${port:-}" ] && continue
        info "closed $(printf '%-6s' "$port") $proc"
      done <<EOF
$REMOVED
EOF
    fi
    info "Accept these as normal: ./port-audit.sh --save"
  fi
fi

# ── 6. Outside-in verification ──

if [ "$DO_SCAN" -eq 1 ]; then
  header "Outside-In Scan"
  if ! command -v rustscan >/dev/null 2>&1; then
    warn "rustscan not installed — brew install rustscan"
  elif [ -z "$LAN_IP" ]; then
    warn "No LAN address to scan"
  else
    info "Scanning all 65535 ports on $LAN_IP from this machine..."
    SCAN=$(rustscan -a "$LAN_IP" -r 1-65535 -b 3000 --no-banner -g --scripts none 2>/dev/null | tail -1)
    if [ -n "$SCAN" ]; then
      ok "Actually reachable: ${SCAN#*-> }"
      info "Should match the 'Reachable From The Network' section above."
    else
      ok "No ports answered — nothing reachable"
    fi
  fi
fi

# ── Summary ──

header "Summary"

echo ""
echo -e "  ${BOLD}$EXPOSED_COUNT${RESET} port(s) reachable from the network · ${BOLD}$LOOPBACK_COUNT${RESET} loopback-only"

if [ "$HIGH_COUNT" -gt 0 ]; then
  echo ""
  echo -e "  ${RED}${BOLD}$HIGH_COUNT high-risk service(s) exposed.${RESET} Close what you don't actively use:"
  echo ""
  echo "  • Remote desktop  → quit the app, or restrict it to a whitelist / disable unattended access"
  echo "  • AirPlay         → System Settings > General > AirDrop & Handoff > AirPlay Receiver"
  echo "  • File sharing    → System Settings > General > Sharing"
  echo "  • Dev servers     → bind to 127.0.0.1 instead of 0.0.0.0"
else
  echo ""
  ok "No high-risk services exposed to the network"
fi

echo ""
echo -e "${DIM}Re-run anytime: ./port-audit.sh [--scan] [--save]${RESET}"
echo ""
