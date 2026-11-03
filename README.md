# TrafficAnalysisSystem

[![CI](https://github.com/Mage-Knight/TrafficAnalysisSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/Mage-Knight/TrafficAnalysisSystem/actions/workflows/ci.yml)

## Quickstart

```bash
uv sync --all-extras   # dev tools + CPU torch (development, tests)
uv run ta env          # Python / torch / CUDA / GPU info
uv run pytest
```

### GPU training

uv pins torch to the CPU index, so install with pip on a CUDA machine (PyPI torch on Linux is the
CUDA build; on Colab the preinstalled CUDA torch is kept):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[yolo]'
ta env                 # must report "CUDA available: True"
```

On headless servers OpenCV needs `libgl1 libglib2.0-0` (apt).

Data and runs live under `$TA_ROOT` (default: the current directory, i.e. `./data/`, `./runs/`).

## Dataset

[UA-DETRAC](https://arxiv.org/abs/1511.04136): 100 traffic videos from Beijing and Tianjin,
~140k frames, classes car / bus / van / others.

- Source: Kaggle mirror [bratjay/ua-detrac-orig](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig)
  (the original Albany site is offline)
- SHA-256 of the archive: `ae4317b2a81ccf8bebc78fa2bdbbc5ec2720b745e0305746d9aa12d40127c32d`
- Download: `bash scripts/download_detrac.sh [--dest data/detrac]` (Kaggle credentials:
  `KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`)
- Licence: UA-DETRAC is for academic and research use only; the data is not redistributed here.

### Data preparation

```bash
ta data stats --root data/detrac --out data/stats      # per-sequence stats + contact sheet
ta data split                                          # configs/splits.yaml from stats + scenes
ta data prepare --root data/detrac --out data/prepared --stride 10
ta data preview --prepared data/prepared --split val   # QA grid
```

Workflow: `ta data stats` → copy `sequence_stats.csv` to `reports/data/` → group
same-viewpoint sequences in `configs/scenes.yaml` → `ta data split` locally → commit all three.

- Split by scene: test = the Test annotation folder. Val = whole train-folder scenes
  (`configs/scenes.yaml`: sequences sharing a camera viewpoint, default = 4-digit MVI prefix),
  so near-duplicates never land in both train and val. Hard constraints: val holds 13–20% of
  train boxes, ≥15% of each class, and every weather appears in train and val; among feasible
  candidates (seeded search) the one closest to a 1/6 share of boxes, classes, small and
  occluded boxes and unstable-camera sequences wins. Scenes are labels only: sequences keep
  their names, and a `scene` column is added to `targets.csv` / `metadata.csv`.
- `coco/{train,valid,test}/` (RF-DETR layout), `yolo/` (Ultralytics, `data.yaml`),
  `mot/{val,test}/` (MOTChallenge gt at full frame rate, class-agnostic), `targets.csv`
  (one row per box), `metadata.csv` (one row per sequence).
- Ignore regions: grey-filled in train images (GT boxes >50% inside are dropped); val/test images
  are raw, with ignore regions stored as COCO `iscrowd` annotations, an `ignore_regions` field,
  `yolo/ignore/` and `mot/*/<seq>/ignore.txt`.
- Re-runs skip existing images; changed settings need `--force`.

## Detection: YOLO baseline

```bash
ta train yolo --config configs/train/yolo_s.yaml [--device 0] [--no-wandb] [--resume]
```

- Reads `data/prepared/yolo` (`--prepared` to override) and writes `runs/yolo/<name>/`:
  `last.pt` and `best.pt` only, `run_meta.json` (git commit, full config) and a runtime
  `data.yaml`. Paths come from `${TA_ROOT}` in the config.
- `--resume` continues `weights/last.pt` after an interruption.
- W&B logging is on by default (`WANDB_API_KEY` or `wandb login`); `--no-wandb` turns it off.
- `device: null` auto-selects the GPU and warns loudly if training falls back to CPU.
- YOLO (Ultralytics) is AGPL-3.0; it is the comparison baseline, not the shipped model.

## Running on Colab (claude-colab, optional)

How this project was developed: code in a dev container, GPU jobs on Colab driven through
[claude-colab](https://github.com/Mage-Knight/claude-colab), data and runs on Google Drive under
`$TA_COLAB_ROOT` (default `/content/drive/MyDrive/traffic-analytics`, passed to Colab as `TA_ROOT`).

```bash
bash scripts/colab_bootstrap.sh                               # unzip data from Drive, clone/pull, pip install
bash scripts/colab_run.sh --bg bash scripts/download_detrac.sh --dest /content/detrac \
  --drive /content/drive/MyDrive/traffic-analytics/data         # first time only
bash scripts/colab_run.sh [--bg] ta data prepare --root /content/detrac --out /content/detrac_prepared
bash scripts/colab_train.sh start yolo [--resume] [config]    # --device 0, refuses if a run is active
bash scripts/colab_train.sh status yolo [--follow]            # alive?, GPU, progress line, last epochs
bash scripts/colab_train.sh log yolo [N]                      # last N log lines
bash scripts/colab_train.sh stop                              # SIGINT, SIGKILL after 30 s
```

`claude-colab exec` doesn't stream output and stops after 600 s, so long jobs run under `nohup`
(`--bg`) and log to Drive.

## License

Apache-2.0
