#!/usr/bin/env bash
# Optional claude-colab helper: run any command on Colab inside the repo, with TA_ROOT set to the
# Drive project folder. On your own machine just run the command directly.
# Usage: bash scripts/colab_run.sh [--bg] <command...>
#   --bg  run under nohup (for jobs longer than claude-colab's 600 s limit); output goes to
#         $TA_COLAB_ROOT/logs/<time>-<command>.log
# Examples:
#   bash scripts/colab_run.sh ta data preview --prepared /content/detrac_prepared --split val
#   bash scripts/colab_run.sh --bg ta data prepare --root /content/detrac --out /content/detrac_prepared
# Arguments are quoted for the remote shell, so $VARS are not expanded on Colab.
# Env:   TA_COLAB_ROOT  Drive project folder on Colab (default /content/drive/MyDrive/traffic-analytics)
set -euo pipefail

COLAB_ROOT="${TA_COLAB_ROOT:-/content/drive/MyDrive/traffic-analytics}"
REPO=/content/TrafficAnalysisSystem

BG=""
if [ "${1:-}" = --bg ]; then BG=1; shift; fi
[ $# -gt 0 ] || { echo "usage: $0 [--bg] <command...>" >&2; exit 1; }

CMD="$(printf '%q ' "$@")"
PREFIX="cd $REPO && TA_ROOT='$COLAB_ROOT'"

if [ -z "$BG" ]; then
  exec uv run claude-colab exec "$PREFIX $CMD" --timeout 600
fi

LOG="$COLAB_ROOT/logs/$(date +%Y%m%d-%H%M%S)-$(basename "$1")${2:+-$(basename "$2")}.log"
uv run claude-colab exec "mkdir -p '$COLAB_ROOT/logs' && $PREFIX nohup $CMD < /dev/null > '$LOG' 2>&1 &
echo \"started pid \$!; log: $LOG\""
echo "check: bash scripts/colab_run.sh tail -n 20 '$LOG'"
