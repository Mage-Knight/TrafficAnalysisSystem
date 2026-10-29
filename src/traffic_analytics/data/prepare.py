"""`ta data prepare`: DETRAC download -> COCO (RF-DETR), YOLO, MOT gt and CSV tables.

Idempotent: existing JPEGs are skipped; small files (JSON, YAML, txt, CSV) are rewritten.
All writes are atomic, so an interrupted run leaves no partial file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from traffic_analytics import __version__
from traffic_analytics.data import export
from traffic_analytics.data.detrac import DetracError, Sequence, parse_sequence
from traffic_analytics.data.discover import Discovery, discover
from traffic_analytics.data.files import atomic_path, copy_file, write_text
from traffic_analytics.data.splits import SPLITS, load_splits, validate_splits

DEFAULT_IMAGE_SIZE = (960, 540)  # DETRAC frame size; used only if a sequence has no images
GENERATED_DIRS = ("coco", "yolo", "mot")
MOT_SPLITS = ("val", "test")


@dataclass
class SplitStats:
    sequences: int = 0
    images_written: int = 0
    images_skipped: int = 0
    boxes: int = 0  # non-crowd COCO/YOLO boxes
    dropped_in_ignore: int = 0
    missing_images: int = 0  # annotated frames (any frame, not only sampled) without a JPEG


@dataclass
class PrepareSummary:
    out: Path
    discovery: Discovery
    stats: dict[str, SplitStats] = field(default_factory=dict)


def _settings(strides: dict[str, int], splits_path: Path) -> dict:
    return {
        "strides": strides,
        "grey": export.GREY,
        "jpeg_quality": export.JPEG_QUALITY,
        "ignore_drop_fraction": export.IGNORE_DROP_FRACTION,
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
    }


def _check_config(out: Path, settings: dict, force: bool) -> None:
    config_path = out / "prepare_config.json"
    if force:
        for name in GENERATED_DIRS:
            shutil.rmtree(out / name, ignore_errors=True)
    elif config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8")).get("settings")
        if previous != settings:
            raise DetracError(
                f"{out} was prepared with different settings ({previous}); "
                "use a new --out or pass --force to rebuild"
            )
    write_text(config_path, json.dumps({"settings": settings, "version": __version__}, indent=2))


def _image_size(img_dir: Path | None, available: set[str]) -> tuple[int, int]:
    if img_dir is None or not available:
        return DEFAULT_IMAGE_SIZE
    with Image.open(img_dir / min(available)) as im:
        return im.size


def _export_detection(
    seq: Sequence,
    split: str,
    stride: int,
    img_dir: Path,
    available: set[str],
    size: tuple[int, int],
    out: Path,
    coco: export.CocoBuilder,
    stats: SplitStats,
) -> None:
    width, height = size
    regions = export.clipped_regions(seq.ignore_regions, width, height)
    is_train = split == "train"
    coco_dir = out / "coco" / export.COCO_DIRS[split]
    yolo_dir = out / "yolo"

    for frame in seq.frames:
        src_name = export.source_name(frame.num)
        if (frame.num - 1) % stride or src_name not in available:
            continue
        stem = export.image_stem(seq.name, frame.num)
        coco_img = coco_dir / f"{stem}.jpg"
        yolo_img = yolo_dir / "images" / split / f"{stem}.jpg"
        if coco_img.exists():
            stats.images_skipped += 1
        else:
            if is_train and regions:
                export.write_masked_jpeg(img_dir / src_name, coco_img, regions)
            else:
                copy_file(img_dir / src_name, coco_img)
            stats.images_written += 1
        if not yolo_img.exists():
            copy_file(coco_img, yolo_img)

        boxes, dropped = export.detection_boxes(frame, width, height, regions, is_train)
        stats.boxes += len(boxes)
        stats.dropped_in_ignore += dropped
        coco.add_image(coco_img.name, width, height, boxes, regions, crowd_regions=not is_train)
        write_text(
            yolo_dir / "labels" / split / f"{stem}.txt", export.yolo_lines(boxes, width, height)
        )
        if not is_train:
            write_text(
                yolo_dir / "ignore" / split / f"{stem}.txt",
                export.yolo_lines(((None, r) for r in regions), width, height),
            )


def _export_mot(
    seq: Sequence, split: str, available: set[str], size: tuple[int, int], out: Path
) -> None:
    seq_dir = out / "mot" / split / seq.name
    length = max([len(available), *(f.num for f in seq.frames)])
    write_text(seq_dir / "gt" / "gt.txt", export.mot_gt(seq))
    write_text(seq_dir / "ignore.txt", export.mot_ignore(seq))
    write_text(seq_dir / "seqinfo.ini", export.seqinfo_ini(seq, length, *size))


def prepare(
    root: Path,
    out: Path,
    splits_path: Path,
    strides: dict[str, int],
    force: bool = False,
    train_xml: Path | None = None,
    test_xml: Path | None = None,
    images: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> PrepareSummary:
    if not splits_path.is_file():
        raise DetracError(f"splits file {splits_path} not found; run `ta data split` first")
    if any(s < 1 for s in strides.values()):
        raise DetracError(f"strides must be >= 1: {strides}")
    splits = load_splits(splits_path)
    discovery = discover(root, train_xml, test_xml, images)
    validate_splits(splits, discovery.train.keys(), discovery.test.keys())

    out.mkdir(parents=True, exist_ok=True)
    _check_config(out, _settings(strides, splits_path), force)
    copy_file(splits_path, out / "splits.yaml")

    summary = PrepareSummary(out, discovery)
    metadata: list[tuple] = []
    with (
        atomic_path(out / "targets.csv") as targets_tmp,
        targets_tmp.open("w", newline="", encoding="utf-8") as targets_file,
    ):
        targets = csv.writer(targets_file)
        targets.writerow(export.TARGET_COLUMNS)

        for split in SPLITS:
            stats = summary.stats[split] = SplitStats()
            coco = export.CocoBuilder()
            for name in sorted(splits[split]):
                xml = discovery.test[name] if split == "test" else discovery.train[name]
                seq = parse_sequence(xml)
                if seq.name != name:
                    raise DetracError(f"{xml}: sequence name {seq.name!r} != file name {name!r}")
                img_dir = discovery.image_dirs.get(name)
                available = (
                    {f for f in os.listdir(img_dir) if f.endswith(".jpg")} if img_dir else set()
                )
                size = _image_size(img_dir, available)
                missing = sum(export.source_name(f.num) not in available for f in seq.frames)

                stats.sequences += 1
                stats.missing_images += missing
                targets.writerows(export.target_rows(seq, split))
                metadata.append(export.metadata_row(seq, split))
                if img_dir is not None:
                    _export_detection(
                        seq, split, strides[split], img_dir, available, size, out, coco, stats
                    )
                if split in MOT_SPLITS:
                    _export_mot(seq, split, available, size, out)
                log(
                    f"[{split}] {name}: {len(seq.frames)} frames, {seq.num_boxes} boxes"
                    + (f", {missing} missing images" if missing else "")
                )

            write_text(
                out / "coco" / export.COCO_DIRS[split] / "_annotations.coco.json",
                json.dumps(coco.to_dict()),
            )
            if split in MOT_SPLITS:
                write_text(
                    out / "mot" / split / "seqmaps.txt",
                    "name\n" + "".join(f"{n}\n" for n in sorted(splits[split])),
                )

    write_text(
        out / "yolo" / "data.yaml",
        yaml.safe_dump(export.yolo_data_yaml(out / "yolo"), sort_keys=False),
    )
    with (
        atomic_path(out / "metadata.csv") as meta_tmp,
        meta_tmp.open("w", newline="", encoding="utf-8") as f,
    ):
        writer = csv.writer(f)
        writer.writerow(export.METADATA_COLUMNS)
        writer.writerows(metadata)
    return summary


def format_summary(summary: PrepareSummary) -> list[str]:
    d = summary.discovery
    lines = [
        *(f"annotation folder: {p}" for p in d.xml_folders),
        *(f"image folder: {p}" for p in d.image_folders),
        *d.warnings,
        f"{'split':<6} {'seqs':>5} {'written':>8} {'skipped':>8} {'boxes':>8} "
        f"{'ignored':>8} {'missing':>8}",
    ]
    for split, s in summary.stats.items():
        row = asdict(s)
        lines.append(
            f"{split:<6} {row['sequences']:>5} {row['images_written']:>8} "
            f"{row['images_skipped']:>8} {row['boxes']:>8} {row['dropped_in_ignore']:>8} "
            f"{row['missing_images']:>8}"
        )
    if total_missing := sum(s.missing_images for s in summary.stats.values()):
        lines.append(f"WARNING: {total_missing} annotated frames have no image file")
    lines.append(f"output: {summary.out}")
    return lines
