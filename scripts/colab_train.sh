#!/usr/bin/env bash
# Optional claude-colab helper: start, watch and stop `ta train <model>` on Colab from the dev
# container. On your own GPU machine just run `ta train` directly.
# claude-colab exec does not stream output, so training runs under nohup and logs to Drive.
# Usage:
#   bash scripts/colab_train.sh start  yolo [--resume] [config]
#   bash scripts/colab_train.sh status yolo [config] [--follow]
#   bash scripts/colab_train.sh log    yolo [config] [N]
#   bash scripts/colab_train.sh stop
# Env:   TA_COLAB_ROOT  Drive project folder on Colab (default /content/drive/MyDrive/traffic-analytics);
#        passed to Colab as TA_ROOT
set -euo pipefail

COLAB_ROOT="${TA_COLAB_ROOT:-/content/drive/MyDrive/traffic-analytics}"
export TA_ROOT="$COLAB_ROOT" # for resolving run paths from the config below
REPO=/content/TrafficAnalysisSystem
MATCH="'[t]a train '" # brackets: the remote shell's own command line doesn't match

die() { echo "error: $*" >&2; exit 1; }
remote() { uv run claude-colab exec "$1" --timeout "${2:-300}"; }

CMD="${1:-}"
[ -n "$CMD" ] || die "usage: $0 start|status|log|stop <model> [options]"
shift

if [ "$CMD" = stop ]; then
  remote "pid=\$(pgrep -o -f $MATCH) || { echo 'not running'; exit 0; }
kill -INT \$pid
for _ in \$(seq 30); do kill -0 \$pid 2>/dev/null || { echo \"stopped pid \$pid (SIGINT)\"; exit 0; }; sleep 1; done
pkill -KILL -f $MATCH; echo \"killed pid \$pid (SIGKILL after 30 s)\"" 60
  exit
fi

MODEL="${1:-}"
[ -n "$MODEL" ] || die "missing model (supported: yolo)"
shift

# Per-model settings; add RF-DETR here when `ta train rfdetr` exists.
case "$MODEL" in
  yolo)
    CFG=configs/train/yolo_s.yaml
    metrics() { # header + last epochs of Ultralytics' results.csv
      echo "f='$1/results.csv'; if [ -f \"\$f\" ]; then { head -n 1 \"\$f\"; tail -n +2 \"\$f\" | tail -n 3; } | column -s, -t; else echo '(no epochs yet)'; fi"
    }
    ;;
  *) die "unknown model '$MODEL' (supported: yolo)" ;;
esac

RESUME="" FOLLOW="" N=30
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=--resume ;;
    --follow) FOLLOW=1 ;;
    *[!0-9]*) CFG="$arg" ;;
    *) N="$arg" ;;
  esac
done
[ -f "$CFG" ] || die "config not found: $CFG"

read -r PROJECT NAME < <(uv run python -c '
import sys
from pathlib import Path
from traffic_analytics.config import load_config
c = load_config(Path(sys.argv[1]), ("project", "name"))
print(c["project"], c["name"])' "$CFG")
LOG="$PROJECT/$NAME.log"
RUN="$PROJECT/$NAME"
LOG_LINES="tr '\r' '\n' < '$LOG' | grep -v '^[[:space:]]*\$'" # \r progress-bar frames -> lines

status_once() {
  remote "if pid=\$(pgrep -o -f $MATCH); then echo \"running (pid \$pid)\"; else echo 'not running'; fi
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
if [ -f '$LOG' ]; then $LOG_LINES | tail -n 1; else echo 'no log: $LOG'; fi
$(metrics "$RUN")
true"
}

case "$CMD" in
  start)
    remote "if pid=\$(pgrep -o -f $MATCH); then echo \"error: ta train already running (pid \$pid)\"; exit 1; fi
cd $REPO && mkdir -p '$PROJECT'
TA_ROOT='$COLAB_ROOT' nohup ta train $MODEL --config '$CFG' --device 0 --prepared /content/detrac_prepared $RESUME < /dev/null >> '$LOG' 2>&1 &
echo \"started pid \$!; log: $LOG\""
    ;;
  status)
    while true; do
      out="$(status_once)"
      [ -z "$FOLLOW" ] || clear
      echo "$out"
      [ -n "$FOLLOW" ] && ! grep -q '^not running' <<<"$out" || break
      sleep 60
    done
    ;;
  log) remote "$LOG_LINES | tail -n $N" ;;
  *) die "unknown command '$CMD' (start|status|log|stop)" ;;
esac
