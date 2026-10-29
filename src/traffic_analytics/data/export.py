"""Converters from parsed DETRAC sequences to COCO, YOLO, MOT and CSV rows."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from traffic_analytics.data.detrac import CLASS_IDS, CLASSES, Box, Frame, Sequence, covered_fraction
from traffic_analytics.data.files import atomic_path

GREY = 114  # Ultralytics letterbox grey
JPEG_QUALITY = 95
IGNORE_DROP_FRACTION = 0.5  # train: drop a GT box if more than this much of it is ignored
COCO_DIRS = {"train": "train", "val": "valid", "test": "test"}  # RF-DETR folder names

TARGET_COLUMNS = (
    "sequence",
    "split",
    "frame",
    "track_id",
    "left",
    "top",
    "width",
    "height",
    "class",
    "occluded",
    "truncation_ratio",
    "speed",
    "orientation",
    "trajectory_length",
    "weather",
    "camera_state",
)
METADATA_COLUMNS = (
    "sequence",
    "split",
    "weather",
    "camera_state",
    "frames",
    "boxes",
    "tracks",
    *(f"n_{c}" for c in CLASSES),
)


def image_stem(sequence: str, frame_num: int) -> str:
    return f"{sequence}_img{frame_num:05d}"


def source_name(frame_num: int) -> str:
    return f"img{frame_num:05d}.jpg"


def clipped_regions(regions: Iterable[Box], width: int, height: int) -> list[Box]:
    return [c for c in (r.clip(width, height) for r in regions) if c.area > 0]


def mask_regions(image: Image.Image, regions: Iterable[Box], grey: int = GREY) -> Image.Image:
    """Fill regions with flat grey; pixels partly covered by a region are filled too."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    for r in clipped_regions(regions, w, h):
        x0, y0 = math.floor(r.left), math.floor(r.top)
        x1, y1 = math.ceil(r.right), math.ceil(r.bottom)
        arr[y0:y1, x0:x1] = grey
    return Image.fromarray(arr)


def write_masked_jpeg(src: Path, dst: Path, regions: Iterable[Box]) -> None:
    with Image.open(src) as im:
        masked = mask_regions(im, regions)
    with atomic_path(dst) as tmp:
        masked.save(tmp, format="JPEG", quality=JPEG_QUALITY)


def detection_boxes(
    frame: Frame, width: int, height: int, regions: list[Box], drop_ignored: bool
) -> tuple[list[tuple[int, Box]], int]:
    """(class id, box clipped to the image) per target, and how many were dropped as ignored."""
    boxes: list[tuple[int, Box]] = []
    dropped = 0
    for t in frame.targets:
        box = t.box.clip(width, height)
        if box.area <= 0:
            continue
        if drop_ignored and covered_fraction(box, regions) > IGNORE_DROP_FRACTION:
            dropped += 1
            continue
        boxes.append((CLASS_IDS[t.vehicle_type], box))
    return boxes, dropped


class CocoBuilder:
    """Accumulates one split's COCO JSON. Category ids are 0-based (RF-DETR convention)."""

    def __init__(self) -> None:
        self.images: list[dict] = []
        self.annotations: list[dict] = []

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        boxes: list[tuple[int, Box]],
        ignore_regions: list[Box],
        crowd_regions: bool,
    ) -> None:
        """crowd_regions: also add each ignore region as iscrowd=1, once per class, so that
        pycocotools ignores detections of any class there."""
        image_id = len(self.images) + 1
        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "ignore_regions": [r.as_list() for r in ignore_regions],
            }
        )
        for category_id, box in boxes:
            self._add_annotation(image_id, category_id, box, iscrowd=0)
        if crowd_regions:
            for region in ignore_regions:
                for category_id in range(len(CLASSES)):
                    self._add_annotation(image_id, category_id, region, iscrowd=1)

    def _add_annotation(self, image_id: int, category_id: int, box: Box, iscrowd: int) -> None:
        self.annotations.append(
            {
                "id": len(self.annotations) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": box.as_list(),
                "area": round(box.area, 2),
                "iscrowd": iscrowd,
            }
        )

    def to_dict(self) -> dict:
        return {
            "info": {"description": "UA-DETRAC (traffic-analytics prepare)"},
            "licenses": [],
            "categories": [
                {"id": i, "name": name, "supercategory": "vehicle"}
                for i, name in enumerate(CLASSES)
            ],
            "images": self.images,
            "annotations": self.annotations,
        }


def yolo_lines(boxes: Iterable[tuple[int | None, Box]], width: int, height: int) -> str:
    """`class cx cy w h` lines, normalized; class None writes `cx cy w h` (ignore files)."""
    lines = []
    for category_id, box in boxes:
        coords = " ".join(f"{v:.6f}" for v in box.to_yolo(width, height))
        lines.append(coords if category_id is None else f"{category_id} {coords}")
    return "".join(f"{line}\n" for line in lines)


def yolo_data_yaml(yolo_dir: Path) -> dict:
    return {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }


def mot_gt(seq: Sequence) -> str:
    """MOTChallenge gt.txt: frame,id,left,top,w,h,conf,class,visibility.

    Class-agnostic (class 1 for every vehicle) as in the DETRAC tracking protocol; boxes unclipped.
    """
    rows = []
    for frame in seq.frames:
        for t in sorted(frame.targets, key=lambda t: t.track_id):
            b = t.box
            rows.append(
                f"{frame.num},{t.track_id},{b.left:.2f},{b.top:.2f},{b.width:.2f},{b.height:.2f},"
                f"1,1,{t.visibility:.4f}\n"
            )
    return "".join(rows)


def mot_ignore(seq: Sequence) -> str:
    return "".join(
        f"{r.left:.2f},{r.top:.2f},{r.width:.2f},{r.height:.2f}\n" for r in seq.ignore_regions
    )


def seqinfo_ini(seq: Sequence, length: int, width: int, height: int) -> str:
    return (
        "[Sequence]\n"
        f"name={seq.name}\n"
        "imDir=img1\n"
        "frameRate=25\n"
        f"seqLength={length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )


def target_rows(seq: Sequence, split: str) -> Iterator[tuple]:
    for frame in seq.frames:
        for t in frame.targets:
            b = t.box
            yield (
                seq.name,
                split,
                frame.num,
                t.track_id,
                b.left,
                b.top,
                b.width,
                b.height,
                t.vehicle_type,
                int(t.occluded),
                t.truncation_ratio,
                t.attributes.get("speed", ""),
                t.attributes.get("orientation", ""),
                t.attributes.get("trajectory_length", ""),
                seq.weather or "",
                seq.camera_state or "",
            )


def metadata_row(seq: Sequence, split: str) -> tuple:
    counts = seq.class_counts()
    return (
        seq.name,
        split,
        seq.weather or "",
        seq.camera_state or "",
        len(seq.frames),
        seq.num_boxes,
        len(seq.track_ids),
        *(counts[c] for c in CLASSES),
    )
