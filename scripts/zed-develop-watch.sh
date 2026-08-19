#!/bin/bash
# What changed on katopz/zed develop since we last looked?
#
# Read-only. Fetches, then reports new fork-authored commits and any new
# .plans/ entries. Does NOT merge — the two lines diverged in April 2026 and a
# merge is a real merge, not a fast-forward (see ollama-night-report.md).
#
#   scripts/zed-develop-watch.sh            # since the last run
#   scripts/zed-develop-watch.sh 7.days     # since a date/duration
set -euo pipefail

REPO="${ZED_DEVELOP_REPO:-$HOME/projects/zed-develop}"
STATE="$HOME/.cache/zed-develop-watch.sha"

[ -d "$REPO/.git" ] || { echo "not a git repo: $REPO" >&2; exit 1; }
cd "$REPO"

git fetch origin develop --quiet
NEW=$(git rev-parse origin/develop)

if [ $# -ge 1 ]; then
  RANGE="origin/develop --since=$1"
  LABEL="since $1"
elif [ -f "$STATE" ] && OLD=$(cat "$STATE") && git cat-file -e "$OLD" 2>/dev/null; then
  RANGE="$OLD..origin/develop"
  LABEL="since last check ($(echo "$OLD" | cut -c1-8))"
else
  RANGE="origin/develop --since=7.days"
  LABEL="last 7 days (no prior state)"
fi

echo "katopz/zed develop — $LABEL"
echo "head: $(git log -1 --format='%h %ci' origin/develop | cut -c1-28)"
echo

# Fork-authored work only; upstream merges are noise for this purpose.
COMMITS=$(git log $RANGE --no-merges --format='%h|%an|%s' 2>/dev/null | grep -iE '\|katopz\|' || true)
if [ -z "$COMMITS" ]; then
  echo "no new fork-authored commits."
else
  echo "$COMMITS" | awk -F'|' '{printf "  %s  %s\n", $1, substr($3,1,88)}'
fi

echo
echo "new/changed plans:"
PLANS=$(git diff --name-only $RANGE -- .plans/ 2>/dev/null | sort -u || true)
[ -z "$PLANS" ] && echo "  (none)" || echo "$PLANS" | sed 's/^/  /'

echo "$NEW" > "$STATE"
