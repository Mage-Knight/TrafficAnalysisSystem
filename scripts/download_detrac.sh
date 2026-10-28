#!/usr/bin/env bash
# Download UA-DETRAC (Kaggle mirror) on Colab; optionally keep a copy on Drive.
# Usage: bash scripts/download_detrac.sh [kaggle-owner/dataset-slug]
# Env:   COPY_TO_DRIVE=0 to skip the Drive copy (re-download from Kaggle each session)
set -euo pipefail

SLUG="${1:-bratjay/ua-detrac-orig}"
COPY_TO_DRIVE="${COPY_TO_DRIVE:-1}"
DL_DIR=/content/dl
DATA_DIR=/content/detrac
DRIVE_DIR=/content/drive/MyDrive/traffic-analytics/data

: "${KAGGLE_USERNAME:?KAGGLE_USERNAME not set (Colab secret)}"
: "${KAGGLE_KEY:?KAGGLE_KEY not set (Colab secret)}"

pip install -q kaggle
mkdir -p "$DL_DIR" "$DATA_DIR"
echo "Free space on /content:"; df -h /content | tail -1

# Download to fast local disk; writing many small files to Drive is slow.
kaggle datasets download "$SLUG" -p "$DL_DIR"

cd "$DL_DIR"
sha256sum ./*.zip | tee SHA256SUMS
echo "https://www.kaggle.com/datasets/$SLUG" > SOURCE

echo "Unzipping to $DATA_DIR ..."
unzip -q -o ./*.zip -d "$DATA_DIR"
echo "Top-level contents:"; ls "$DATA_DIR"

if [ "$COPY_TO_DRIVE" = "1" ]; then
  [ -d /content/drive/MyDrive ] || { echo "Drive is not mounted"; exit 1; }
  mkdir -p "$DRIVE_DIR"
  cp ./*.zip SHA256SUMS SOURCE "$DRIVE_DIR/"
  echo "Saved to Drive:"; ls -lh "$DRIVE_DIR"
fi

echo "DONE"