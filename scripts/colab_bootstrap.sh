#!/usr/bin/env bash
# Optional claude-colab helper. Start-of-session setup for the Colab runtime, run from the dev container:
# unzip UA-DETRAC from Drive to /content/detrac, then clone/pull this repo and install it.
# Usage: bash scripts/colab_bootstrap.sh   (Drive must be mounted in the Colab server notebook)
# Env:   TA_COLAB_ROOT  Drive project folder on Colab (default /content/drive/MyDrive/traffic-analytics)
set -euo pipefail

COLAB_ROOT="${TA_COLAB_ROOT:-/content/drive/MyDrive/traffic-analytics}"
DATA="$COLAB_ROOT/data"

# uv run claude-colab exec "mkdir -p /content/detrac && unzip -q -o '$DATA/ua-detrac-orig.zip' -d /content/detrac && ls /content/detrac"
uv run claude-colab exec "mkdir -p /content/detrac_prepared && cd /content/detrac_prepared && unzip -q -o '$DATA/detrac_yolo.zip' && unzip -q -o '$DATA/detrac_coco.zip' && ls"
uv run claude-colab exec 'cd /content && (git -C TrafficAnalysisSystem pull || git clone https://github.com/Mage-Knight/TrafficAnalysisSystem.git) && pip install -q -e "TrafficAnalysisSystem[yolo]"'

echo "Train: bash scripts/colab_train.sh start yolo   (then: status yolo --follow | log yolo | stop)"
echo "Other commands: bash scripts/colab_run.sh [--bg] ta data ..."
