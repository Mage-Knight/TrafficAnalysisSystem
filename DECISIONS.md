# Decisions

Why the project is built the way it is

## Dataset

### D1. UA-DETRAC from the Kaggle mirror (2026-10)
- **Choice:** [bratjay/ua-detrac-orig](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig),
  archive SHA-256 `ae4317b2a81ccf8bebc78fa2bdbbc5ec2720b745e0305746d9aa12d40127c32d`.
  Downloaded on Colab (`scripts/download_detrac.sh`), kept as a zip on Drive and unzipped to
  `/content/detrac` each session (`scripts/colab_bootstrap.sh`).
- **Why:** the original Albany site is offline. The mirror has the original per-sequence XML
  annotations: 60 sequences in the Train folder and 40 in the Test folder, classes car / bus /
  van / others.
- **Constraint:** academic and research use only, so no data, frames or labels are committed.
  Only per-sequence aggregates (`reports/data/sequence_stats.csv`) are in the repo.

### D2. XML was parsed once into typed dataclasses
- `traffic_analytics.data.detrac` reads the following:
  - sequence: weather (the dataset spells it `sence_weather`), `camera_state`, ignore regions
  - per box: track id, box (left/top/width/height), `vehicle_type`, `truncation_ratio`, occlusion regions
  - all other `<attribute>` fields are kept as a raw dict
- An unknown `vehicle_type` is an error rather than a silent drop.

## Splits

### D3. Test = the official Test folder
- **Choice:** test is exactly the 40 sequences in the Test annotation folder and is never used
  to choose anything.
- **Why:** this keeps results comparable with published UA-DETRAC numbers. The Train and Test
  folders share no 4-digit MVI prefix. Viewpoints that looked similar across the two folders were checked and are different scenes.

### D4. Val is held out by scene, not by frame or sequence
- **Problem:** frames of one video are near-duplicates, so a split by frame inflates mAP. Splitting
  by sequence is still not enough. The contact sheet (`ta data stats`) shows that sequences
  sharing a 4-digit MVI prefix (ex.: MVI_**4016**1, MVI_**4016**2) were recorded by one camera, from one
  viewpoint, in the same weather and at about the same time. Some viewpoints also span several
  prefixes.
- **Choice:** `configs/scenes.yaml` gives every sequence a scene label: by default its 4-digit MVI
  prefix. The following prefix groups were judged by eye to share a viewpoint:
  - 2001 + 2006
  - 2003 + 2005
  - 3950 + 3951 (test)
  - 3976 + 3977
  - 3980 + 3981 + 3982
  - 4016 + 4017 + 4018
  - 4073 + 4075
  - 4106 + 4107
  - 6355 + 6356

  Val is a set of **whole** train-folder scenes, so a scene is never in both train and val.
- **Scenes are labels only.** Sequences are never renamed, merged or re-saved. Every output
  keeps the original sequence names, and `targets.csv` / `metadata.csv` gain a `scene` column
  for sliced analysis.
- **Rejected:**
  - Sequence-level stratification by weather × camera_state (the first version): it puts
    near-duplicate sequences of one scene in both train and val.
  - The 3-digit MVI prefix: too coarse. For example, group 401 holds several different viewpoints.

### D5. How val scenes are chosen (`ta data split`)
The split runs locally from the committed `reports/data/sequence_stats.csv` and
`configs/scenes.yaml`. It is deterministic (seed 2, 20,000 seeded shuffles of the 25
train-folder scenes). Each shuffle adds whole scenes one at a time; every prefix that lands in
the box-share window is a candidate.

**Hard constraints:**
- val holds 13–20% of the train-folder boxes (the sequence count is free, because scene sizes
  vary widely)
- val holds ≥ 15% of each class's train-folder boxes, so per-class AP on val means something.
  `others` is scarce in train (3,726 boxes).
- every weather (cloudy, night, rainy, sunny) has at least one scene in val **and** in train

**Score among feasible candidates** (lower is better):
- mean of nine equally weighted balance terms:
  - distance of val's share from 1/6, one term each for: all boxes, each class, small boxes
    (√area < 50 px), occluded boxes and heavily occluded boxes (visibility < 0.5)
  - the weather term: total-variation distance between the box-weighted weather distributions
    of val and the train folder (0 = same mix, 1 = no overlap)
- plus the gap in the fraction of unstable-camera sequences
- minus 0.002 per val scene

The weather term is soft. The one-scene-per-weather rule above stays hard, so every weather is
always present in val. The scene bonus is a tie-breaker only: a bonus of 0.005 outweighed the
balance terms, pulled a whole 6-sequence sunny scene into val and left train with a single
sunny scene.

**Result** (written into the header of `configs/splits.yaml`, `--seed 2`): 44 train / 16 val /
40 test sequences. Val holds 9 scenes:
- s3980, s3985, s4099 (night)
- s3993, s4014, s4015, s4021 (cloudy)
- s4106 (sunny)
- s6352 (rainy)

Val shares of the train-folder totals:

| boxes | car | bus | van | others | small | occluded | heavy occl. |
|---|---|---|---|---|---|---|---|
| 0.170 | 0.170 | 0.160 | 0.167 | 0.232 | 0.128 | 0.163 | 0.160 |

Weather TVD (box-weighted): 0.061. Unstable-camera sequences: val 0.62 vs train folder 0.63.
Adding the weather term did not change this split: seed 2 picks the same val scenes with and
without it.

**The seed decides the result (search is not exhaustive):** the random search does not reach
the best-scoring split, so different seeds return different val sets.

| | seed 0 | seed 2 (used) |
|---|---|---|
| sequences train / val | 49 / 11 | 44 / 16 |
| val box share | 0.190 | 0.170 |
| van / others share | 0.249 / 0.166 | 0.167 / 0.232 |
| night share of val vs train sequences | 0.09 vs 0.31 | 0.38 vs 0.23 |
| weather TVD | 0.179 | 0.061 |
| score without / with weather term | 0.0186 / 0.0350 | 0.0062 / 0.0112 |

The seed-2 split scores better under both scores, so seed 0 keeps its split only because its
20,000 shuffles never reach the better one. Even with 200,000 shuffles, seeds 0–5 each return a
different val set. Seed 2 is the split in use.

**Known imbalance:** weather share by sequences:

| | cloudy | night | rainy | sunny |
|---|---|---|---|---|
| val | 0.38 | 0.38 | 0.12 | 0.12 |
| train | 0.30 | 0.23 | 0.18 | 0.30 |

Sunny is under-represented in val (a single scene, s4106) and night is over-represented. However, I think this is the optimal decision, because sunny is not that much different from cloudy or rainy (they are significantly brighter than night).

**Changing the split:** edit `configs/scenes.yaml` or the flags (e.g. `--seed`), re-run
`ta data split`, and commit both files. `tests/test_splits.py` fails if the committed stats +
scenes become infeasible.

## Exports (`ta data prepare`)

### D6. Detection data: sampled frames, two formats
- **Sampling:** every Nth annotated frame (`--stride`, default 10, which is 2.5 fps at 25 fps;
  per-split overrides are available). Consecutive frames add little and make Colab runs longer.
- **Formats:**
  - COCO for RF-DETR: `coco/{train,valid,test}/_annotations.coco.json`, 0-based category ids
  - YOLO for Ultralytics: `yolo/images|labels/{train,val,test}`, `data.yaml`

  Images are duplicated between the two trees so each stands alone and zips cleanly to Drive.
- **Boxes:** clipped to the image; zero-area boxes are dropped.

### D7. Ignore regions
DETRAC leaves vehicles inside "ignore regions" unlabelled. If this is skipped, correct detections
there count as false positives.
- **Train:**
  - regions are filled with flat grey (114, the Ultralytics letterbox value) and re-encoded at
    JPEG quality 95
  - GT boxes more than 50% inside a region are dropped
  - frames with no region are byte-copied
- **Val/test:**
  - images stay raw and all GT boxes are kept
  - each region is stored as an `iscrowd=1` annotation once **per class**, because pycocotools
    matches crowd regions per category
  - each region is also stored as an `ignore_regions` field on the image, in `yolo/ignore/` and
    in `mot/*/<seq>/ignore.txt`
- **Evaluation contract:** our own eval drops predictions and GT more than 50% inside ignore
  regions, the same way for every model.

### D8. Tracking ground truth
- **Format:** MOTChallenge `gt.txt` for val and test at **full** frame rate, because trackers need
  small motion between frames. `seqinfo.ini` and `seqmaps.txt` are written for TrackEval.
- **Class-agnostic** (class 1 for every vehicle), following the UA-DETRAC tracking protocol.
  The vehicle type stays in `targets.csv` and the detection files.
- **Boxes and visibility:** boxes are unclipped. Visibility = 1 − the fraction of the box
  covered by `<occlusion>` regions; truncation is left out of it.

### D9. Analysis tables
- **`targets.csv`:** one row per box, all splits, full frame rate. Columns:
  - sequence, scene, split, frame, track_id, box
  - class, occluded, truncation_ratio
  - speed, orientation, trajectory_length
  - weather, camera_state

  These fields are deliberately not added to the COCO, YOLO or MOT files.
- **`metadata.csv`:** one row per sequence, with counts per class.

### D10. Re-runs must be safe on Colab
- **Idempotent:** existing JPEGs are skipped, and small files are rewritten.
- **Atomic writes:** every write (JPEGs included) goes to a temp file and is then renamed, so an
  interrupted session never leaves a partial file that a re-run would skip.
- **Config guard:** `prepare_config.json` records the strides, grey value, JPEG quality and the
  hashes of `splits.yaml` and `scenes.yaml`. Re-using an output folder with different settings
  is an error unless `--force` is given.
- **Self-describing output:** copies of both YAML files are stored in the output folder.
