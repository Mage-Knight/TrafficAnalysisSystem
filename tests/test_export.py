import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import IMAGE_SIZE
from PIL import Image

from traffic_analytics.data.detrac import Box, DetracError
from traffic_analytics.data.export import GREY, mask_regions
from traffic_analytics.data.prepare import prepare

STRIDES = {"train": 1, "val": 1, "test": 1}
WIDTH, HEIGHT = IMAGE_SIZE


@pytest.fixture
def prepared(detrac_root: Path, splits_file: Path, tmp_path: Path) -> Path:
    out = tmp_path / "out"
    prepare(detrac_root, out, splits_file, STRIDES)
    return out


def _coco(out: Path, folder: str) -> dict:
    return json.loads((out / "coco" / folder / "_annotations.coco.json").read_text())


def test_mask_regions_on_synthetic_image() -> None:
    image = Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8))
    masked = np.asarray(mask_regions(image, [Box(2.5, 1, 4, 3), Box(15, 5, 10, 10)]))
    assert (masked[1:4, 2:7] == GREY).all()  # partial pixels at 2.5 and 6.5 are filled
    assert (masked[5:, 15:] == GREY).all()  # region clipped to the image
    assert masked[0].sum() == 0 and masked[:, :2].sum() == 0
    assert (masked == GREY).all(axis=2).sum() == 3 * 5 + 5 * 5


def test_train_images_masked_eval_images_copied(prepared: Path, detrac_root: Path) -> None:
    train = np.asarray(Image.open(prepared / "coco/train/MVI_00001_img00001.jpg")).astype(int)
    # JPEG re-encode: chroma bleeds a few pixels at the region edge, interior stays flat
    assert np.abs(train[0:14, 48:64] - GREY).max() <= 3
    assert np.abs(train[0:16, 40:64] - GREY).mean() <= 3
    assert np.abs(train[30:, :30] - GREY).mean() > 30  # outside the region untouched
    src = detrac_root / "Insight-MVT_Annotation"
    for folder, seq in (("valid", "MVI_00003"), ("test", "MVI_00002")):
        assert (prepared / f"coco/{folder}/{seq}_img00001.jpg").read_bytes() == (
            src / seq / "img00001.jpg"
        ).read_bytes()
    yolo = prepared / "yolo/images/train/MVI_00001_img00001.jpg"
    assert yolo.read_bytes() == (prepared / "coco/train/MVI_00001_img00001.jpg").read_bytes()


@pytest.mark.parametrize("folder", ["train", "valid", "test"])
def test_coco_json_valid(prepared: Path, folder: str) -> None:
    coco = _coco(prepared, folder)
    assert [c["id"] for c in coco["categories"]] == [0, 1, 2, 3]
    image_ids = [i["id"] for i in coco["images"]]
    ann_ids = [a["id"] for a in coco["annotations"]]
    assert len(set(image_ids)) == len(image_ids) == 3
    assert len(set(ann_ids)) == len(ann_ids)
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    for image in coco["images"]:
        assert image["ignore_regions"] == [[40, 0, 24, 16]]
        assert (prepared / "coco" / folder / image["file_name"]).is_file()
    for a in coco["annotations"]:
        w, h = sizes[a["image_id"]]
        left, top, bw, bh = a["bbox"]
        assert bw > 0 and bh > 0
        assert left >= 0 and left + bw <= w and top >= 0 and top + bh <= h


def test_coco_crowd_and_ignore_drop(prepared: Path) -> None:
    train = _coco(prepared, "train")["annotations"]
    assert not any(a["iscrowd"] for a in train)
    # 5 boxes minus the bus inside the ignore region on frame 1
    assert len(train) == 4
    for folder in ("valid", "test"):
        anns = _coco(prepared, folder)["annotations"]
        crowd = [a for a in anns if a["iscrowd"]]
        assert len([a for a in anns if not a["iscrowd"]]) == 5
        assert len(crowd) == 3 * 4  # 3 images x 1 region x 4 classes
        assert {a["category_id"] for a in crowd} == {0, 1, 2, 3}
    clipped = [a for a in train if a["bbox"][0] == 56]
    assert clipped[0]["bbox"] == [56, 30, 8, 8]


def test_yolo_labels_normalized(prepared: Path) -> None:
    data = yaml.safe_load((prepared / "yolo/data.yaml").read_text())
    assert data["names"] == {0: "car", 1: "bus", 2: "van", 3: "others"}
    assert {data["train"], data["val"], data["test"]} == {
        "images/train",
        "images/val",
        "images/test",
    }
    labels = sorted((prepared / "yolo/labels").rglob("*.txt"))
    assert len(labels) == 9
    for label in labels:
        for line in label.read_text().splitlines():
            cls, *coords = line.split()
            assert int(cls) in range(4)
            assert all(0 <= float(v) <= 1 for v in coords)
    ignore = (prepared / "yolo/ignore/val/MVI_00003_img00001.txt").read_text().split()
    assert [float(v) for v in ignore] == pytest.approx(
        [52 / 64, 8 / 48, 24 / 64, 16 / 48], abs=1e-6
    )
    assert not (prepared / "yolo/ignore/train").exists()


def test_mot_export(prepared: Path) -> None:
    assert not (prepared / "mot/train").exists()
    seq_dir = prepared / "mot/val/MVI_00003"
    rows = [line.split(",") for line in (seq_dir / "gt/gt.txt").read_text().splitlines()]
    assert len(rows) == 5
    assert all(r[6] == "1" and r[7] == "1" for r in rows)
    assert [float(r[8]) for r in rows if r[0] == "2" and r[1] == "1"] == [0.5]
    assert ["56.00", "30.00", "12.00", "8.00"] in [r[2:6] for r in rows]  # unclipped
    assert (seq_dir / "ignore.txt").read_text() == "40.00,0.00,24.00,16.00\n"
    assert "seqLength=3" in (seq_dir / "seqinfo.ini").read_text()
    assert (prepared / "mot/test/seqmaps.txt").read_text() == "name\nMVI_00002\n"


def test_csv_tables(prepared: Path) -> None:
    with (prepared / "targets.csv").open() as f:
        targets = list(csv.DictReader(f))
    assert len(targets) == 15  # 3 sequences x 5 boxes, no crowd regions
    first = targets[0]
    assert first["sequence"] == "MVI_00001" and first["split"] == "train"
    assert first["speed"] == "6.86" and first["weather"] == "sunny"
    with (prepared / "metadata.csv").open() as f:
        meta = {r["sequence"]: r for r in csv.DictReader(f)}
    assert meta["MVI_00002"]["split"] == "test"
    assert (meta["MVI_00002"]["boxes"], meta["MVI_00002"]["tracks"]) == ("5", "2")
    assert (meta["MVI_00002"]["n_car"], meta["MVI_00002"]["n_bus"]) == ("3", "2")


def test_stride_samples_frames(detrac_root: Path, splits_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = prepare(detrac_root, out, splits_file, {"train": 2, "val": 3, "test": 3})
    assert sorted(p.name for p in (out / "coco/train").glob("*.jpg")) == [
        "MVI_00001_img00001.jpg",
        "MVI_00001_img00003.jpg",
    ]
    assert summary.stats["val"].images_written == 1
    assert len((out / "mot/val/MVI_00003/gt/gt.txt").read_text().splitlines()) == 5


def test_rerun_idempotent_and_config_guard(
    prepared: Path, detrac_root: Path, splits_file: Path
) -> None:
    jpgs = sorted(prepared.rglob("*.jpg"))
    mtimes = [p.stat().st_mtime_ns for p in jpgs]
    summary = prepare(detrac_root, prepared, splits_file, STRIDES)
    assert summary.stats["train"].images_written == 0
    assert summary.stats["train"].images_skipped == 3
    assert [p.stat().st_mtime_ns for p in jpgs] == mtimes
    assert not list(prepared.rglob("*.tmp"))
    assert (prepared / "splits.yaml").read_bytes() == splits_file.read_bytes()
    config = json.loads((prepared / "prepare_config.json").read_text())
    assert config["settings"]["strides"] == STRIDES

    with pytest.raises(DetracError, match="different settings"):
        prepare(detrac_root, prepared, splits_file, {**STRIDES, "train": 2})
    prepare(detrac_root, prepared, splits_file, {**STRIDES, "train": 2}, force=True)
    assert len(list((prepared / "coco/train").glob("*.jpg"))) == 2


def test_missing_splits_file(detrac_root: Path, tmp_path: Path) -> None:
    with pytest.raises(DetracError, match="ta data split"):
        prepare(detrac_root, tmp_path / "out", tmp_path / "nope.yaml", STRIDES)


def test_missing_images_counted(detrac_root: Path, splits_file: Path, tmp_path: Path) -> None:
    (detrac_root / "Insight-MVT_Annotation/MVI_00002/img00002.jpg").unlink()
    summary = prepare(detrac_root, tmp_path / "out", splits_file, STRIDES)
    assert summary.stats["test"].missing_images == 1
    assert len(_coco(tmp_path / "out", "test")["images"]) == 2
