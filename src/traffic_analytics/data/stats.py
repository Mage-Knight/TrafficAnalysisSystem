"""Per-sequence statistics for designing the split (grouping, constraints, balance)."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw

from traffic_analytics.data.detrac import CLASSES, Box, Sequence, covered_fraction
from traffic_analytics.data.files import atomic_path

SMALL_MAX, MEDIUM_MAX = 50, 150  # sqrt(box area) thresholds in px, as in the DETRAC paper
HEAVY_OCCLUSION_VISIBILITY = 0.5
THUMB_SIZE = (160, 90)
SHEET_COLUMNS = 10

STATS_COLUMNS = (
    "sequence",
    "folder",
    "group",
    "weather",
    "camera_state",
    "frames",
    "boxes",
    "tracks",
    *(f"n_{c}" for c in CLASSES),
    "density_mean",
    "density_max",
    "small_frac",
    "medium_frac",
    "large_frac",
    "occluded_frac",
    "heavy_occl_frac",
    "trunc_mean",
    "ignore_area_frac",
)


def mvi_group(name: str) -> str:
    """Location proxy: first 3 digits of the MVI id (MVI_40171 -> 401)."""
    match = re.fullmatch(r"MVI_(\d{3})\d*", name)
    return match.group(1) if match else name


def first_frame(img_dir: Path | None) -> Path | None:
    return min(img_dir.glob("img*.jpg"), default=None) if img_dir else None


def image_size(img_dir: Path | None, default: tuple[int, int] = (960, 540)) -> tuple[int, int]:
    first = first_frame(img_dir)
    if first is None:
        return default
    with Image.open(first) as im:
        return im.size


def _frac(count: float, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def sequence_stats(seq: Sequence, folder: str, image_size: tuple[int, int]) -> dict:
    targets = [t for f in seq.frames for t in f.targets]
    n = len(targets)
    scales = [math.sqrt(t.box.area) for t in targets]
    densities = [f.density if f.density is not None else len(f.targets) for f in seq.frames]
    width, height = image_size
    return {
        "sequence": seq.name,
        "folder": folder,
        "group": mvi_group(seq.name),
        "weather": seq.weather or "",
        "camera_state": seq.camera_state or "",
        "frames": len(seq.frames),
        "boxes": n,
        "tracks": len(seq.track_ids),
        **{f"n_{c}": k for c, k in seq.class_counts().items()},
        "density_mean": round(sum(densities) / len(densities), 2) if densities else 0.0,
        "density_max": max(densities, default=0),
        "small_frac": _frac(sum(s < SMALL_MAX for s in scales), n),
        "medium_frac": _frac(sum(SMALL_MAX <= s < MEDIUM_MAX for s in scales), n),
        "large_frac": _frac(sum(s >= MEDIUM_MAX for s in scales), n),
        "occluded_frac": _frac(sum(t.occluded for t in targets), n),
        "heavy_occl_frac": _frac(
            sum(t.visibility < HEAVY_OCCLUSION_VISIBILITY for t in targets), n
        ),
        "trunc_mean": _frac(sum(t.truncation_ratio for t in targets), n),
        "ignore_area_frac": round(
            covered_fraction(Box(0, 0, width, height), seq.ignore_regions), 4
        ),
    }


def group_summary(rows: Iterable[dict]) -> list[str]:
    """One line per MVI group; groups with sequences in both folders are flagged."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    lines = []
    for group, members in sorted(groups.items()):
        n_train = sum(r["folder"] == "train" for r in members)
        n_test = len(members) - n_train
        weathers = ",".join(sorted({r["weather"] for r in members}))
        boxes = sum(r["boxes"] for r in members)
        flag = "  <- spans train and test" if n_train and n_test else ""
        lines.append(
            f"group {group}: {n_train} train, {n_test} test, {boxes} boxes, {weathers}{flag}"
        )
    return lines


def folder_totals(rows: Iterable[dict]) -> list[str]:
    """Class counts and box-weighted scale / occlusion shares per folder."""
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_folder[row["folder"]].append(row)
    lines = []
    for folder, members in sorted(by_folder.items()):
        boxes = sum(r["boxes"] for r in members)

        def share(key: str, members: list[dict] = members, boxes: int = boxes) -> float:
            return _frac(sum(r[key] * r["boxes"] for r in members), boxes)

        classes = " ".join(f"{c}={sum(r[f'n_{c}'] for r in members)}" for c in CLASSES)
        lines.append(
            f"{folder}: {len(members)} seqs, {boxes} boxes ({classes}); "
            f"small={share('small_frac')} medium={share('medium_frac')} "
            f"large={share('large_frac')} occluded={share('occluded_frac')} "
            f"heavy_occl={share('heavy_occl_frac')}"
        )
    return lines


def contact_sheet(rows: list[dict], image_dirs: dict[str, Path], out: Path) -> None:
    """First frame of every sequence, ordered by group, labelled name / group / weather."""
    ordered = sorted(rows, key=lambda r: (r["group"], r["sequence"]))
    tw, th = THUMB_SIZE
    label_h = 12
    n_rows = max(math.ceil(len(ordered) / SHEET_COLUMNS), 1)
    sheet = Image.new("RGB", (SHEET_COLUMNS * tw, n_rows * (th + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(ordered):
        x, y = (i % SHEET_COLUMNS) * tw, (i // SHEET_COLUMNS) * (th + label_h)
        first = first_frame(image_dirs.get(row["sequence"]))
        if first is not None:
            with Image.open(first) as im:
                sheet.paste(im.convert("RGB").resize(THUMB_SIZE), (x, y + label_h))
        label = f"{row['sequence'].removeprefix('MVI_')} / {row['group']} / {row['weather']}"
        draw.text((x + 2, y), label, fill="black")
    with atomic_path(out) as tmp:
        sheet.save(tmp, format="JPEG", quality=85)
