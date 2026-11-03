#!/usr/bin/env bash
# Download UA-DETRAC (Kaggle mirror), unzip it, and optionally copy the zips to another folder
# (e.g. Google Drive on Colab, so later sessions skip the download).
# Usage: bash scripts/download_detrac.sh [--dest DIR] [--drive DIR] [kaggle-owner/dataset-slug]
#   --dest   unzip target (default ./data/detrac); zips go to <dest>/../downloads
#   --drive  also copy the zips, SHA256SUMS and SOURCE there
# Credentials: KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json
set -euo pipefail

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; }

DEST=./data/detrac
DRIVE=""
SLUG=bratjay/ua-detrac-orig
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="${2:?--dest needs a folder}"; shift 2 ;;
    --drive) DRIVE="${2:?--drive needs a folder}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option $1" >&2; usage >&2; exit 1 ;;
    *) SLUG="$1"; shift ;;
  esac
done

if [ -z "${KAGGLE_USERNAME:-}${KAGGLE_KEY:-}" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "error: no Kaggle credentials: set KAGGLE_USERNAME and KAGGLE_KEY, or create ~/.kaggle/kaggle.json" >&2
  exit 1
fi
command -v kaggle >/dev/null || pip install -q kaggle

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
DL_DIR="$(dirname "$DEST")/downloads"
mkdir -p "$DL_DIR"
echo "Free space at $DEST:"; df -h "$DEST" | tail -1

# Download to local disk; writing many small files to a network/Drive mount is slow.
kaggle datasets download "$SLUG" -p "$DL_DIR"

cd "$DL_DIR"
sha256sum ./*.zip | tee SHA256SUMS
echo "https://www.kaggle.com/datasets/$SLUG" > SOURCE

echo "Unzipping to $DEST ..."
unzip -q -o ./*.zip -d "$DEST"
echo "Top-level contents:"; ls "$DEST"

if [ -n "$DRIVE" ]; then
  mkdir -p "$DRIVE"
  cp ./*.zip SHA256SUMS SOURCE "$DRIVE/"
  echo "Copied to $DRIVE:"; ls -lh "$DRIVE"
fi

echo "DONE"
