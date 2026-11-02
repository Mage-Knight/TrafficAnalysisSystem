# TrafficAnalysisSystem

[![CI](https://github.com/Mage-Knight/TrafficAnalysisSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/Mage-Knight/TrafficAnalysisSystem/actions/workflows/ci.yml)

## Quickstart

```bash
uv sync --all-extras   # dev tools + CPU torch
uv run ta env          # Python / torch / CUDA / GPU info
uv run pytest
```

On Colab (CUDA torch preinstalled): `pip install -e .`

## Dataset

[UA-DETRAC](https://arxiv.org/abs/1511.04136): 100 traffic videos from Beijing and Tianjin,
~140k frames, classes car / bus / van / others.

- Source: Kaggle mirror [bratjay/ua-detrac-orig](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig)
  (the original Albany site is offline)
- SHA-256 of the archive: `ae4317b2a81ccf8bebc78fa2bdbbc5ec2720b745e0305746d9aa12d40127c32d`
- Download: `bash scripts/download_detrac.sh` on Colab (needs Kaggle API credentials)
- Licence: UA-DETRAC is for academic and research use only; the data is not redistributed here.

### Data preparation

```bash
ta data stats --root /content/detrac --out /content/detrac_stats   # Colab: per-sequence stats + contact sheet
ta data split                                           # local: configs/splits.yaml from stats + scenes
ta data prepare --root /content/detrac --out /content/detrac_prepared --stride 10   # Colab
```

Workflow: `ta data stats` on Colab → copy `sequence_stats.csv` to `reports/data/` → group
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

## License

Apache-2.0
