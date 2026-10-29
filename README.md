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
ta data split --root /content/detrac                    # writes configs/splits.yaml (commit it)
ta data prepare --root /content/detrac --out /content/detrac_prepared --stride 10
```

- Split by sequence: test = the Test annotation folder; 10 train sequences held out as val,
  stratified by weather and camera state (seed 0).
- `coco/{train,valid,test}/` (RF-DETR layout), `yolo/` (Ultralytics, `data.yaml`),
  `mot/{val,test}/` (MOTChallenge gt at full frame rate, class-agnostic), `targets.csv`
  (one row per box), `metadata.csv` (one row per sequence).
- Ignore regions: grey-filled in train images (GT boxes >50% inside are dropped); val/test images
  are raw, with ignore regions stored as COCO `iscrowd` annotations, an `ignore_regions` field,
  `yolo/ignore/` and `mot/*/<seq>/ignore.txt`.
- Re-runs skip existing images; changed settings need `--force`.

## License

Apache-2.0
