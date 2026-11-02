"""`ta data preview`: grid of prepared images with GT boxes and ignore regions, for visual QA.

Reads only the prepared outputs, so each format is checked against its own files: YOLO train
has no ignore files (its ignore regions are grey-filled in the pixels), so nothing is hatched.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from traffic_analytics.data.detrac import Box, DetracError
from traffic_analytics.data.export import COCO_DIRS
from traffic_analytics.data.files import atomic_path
from traffic_analytics.data.splits import SPLITS

FORMATS = ("coco", "yolo")
GRID_COLUMNS = 4
TILE_WIDTH = 480
LABEL_HEIGHT = 14
HATCH_STEP = 8
IGNORE_COLOR = (255, 255, 255)
CLASS_COLORS = {
    "car": (0, 200, 0),
    "bus": (255, 140, 0),
    "van": (0, 160, 255),
    "others": (230, 0, 230),
}
FALLBACK_COLOR = (255, 255, 0)


@dataclass(frozen=True)
class PreviewItem:
    image: Path
    boxes: list[tuple[str, Box]]  # (class name, pixel box)
    ignore: list[Box]


def _not_prepared(path: Path) -> DetracError:
    return DetracError(f"{path} not found; run `ta data prepare` first")


def load_coco(prepared: Path, split: str) -> list[PreviewItem]:
    folder = prepared / "coco" / COCO_DIRS[split]
    path = folder / "_annotations.coco.json"
    if not path.is_file():
        raise _not_prepared(path)
    coco = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    boxes: dict[int, list[tuple[str, Box]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if not ann["iscrowd"]:  # crowd anns repeat the ignore regions once per class
            boxes[ann["image_id"]].append((names[ann["category_id"]], Box(*ann["bbox"])))
    return [
        PreviewItem(
            folder / im["file_name"],
            boxes[im["id"]],
            [Box(*r) for r in im.get("ignore_regions", [])],
        )
        for im in coco["images"]
    ]


def _read_yolo(path: Path, width: int, height: int, with_class: bool) -> list[tuple[int, Box]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        category_id = int(values.pop(0)) if with_class else -1
        rows.append((category_id, Box.from_yolo(*map(float, values), width, height)))
    return rows


def load_yolo(prepared: Path, split: str) -> list[PreviewItem]:
    yolo_dir = prepared / "yolo"
    data_yaml = yolo_dir / "data.yaml"
    image_dir = yolo_dir / "images" / split
    for path in (data_yaml, image_dir):
        if not path.exists():
            raise _not_prepared(path)
    names = {int(k): v for k, v in yaml.safe_load(data_yaml.read_text())["names"].items()}
    items = []
    for image in sorted(image_dir.glob("*.jpg")):
        with Image.open(image) as im:
            width, height = im.size
        label = yolo_dir / "labels" / split / f"{image.stem}.txt"
        ignore = yolo_dir / "ignore" / split / f"{image.stem}.txt"
        items.append(
            PreviewItem(
                image,
                [(names[c], b) for c, b in _read_yolo(label, width, height, with_class=True)],
                [b for _, b in _read_yolo(ignore, width, height, with_class=False)],
            )
        )
    return items


def load_items(prepared: Path, split: str, fmt: str) -> list[PreviewItem]:
    if split not in SPLITS:
        raise DetracError(f"unknown split {split!r}; expected one of {SPLITS}")
    if fmt not in FORMATS:
        raise DetracError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    items = load_coco(prepared, split) if fmt == "coco" else load_yolo(prepared, split)
    if not items:
        raise DetracError(f"no {fmt} images for split {split!r} in {prepared}")
    return items


def sample(items: list[PreviewItem], n: int, seed: int = 0) -> list[PreviewItem]:
    """Seeded sample, round-robin over sequences so tiles come from different sequences."""
    rng = random.Random(seed)
    groups: dict[str, list[PreviewItem]] = defaultdict(list)
    for item in items:
        groups[item.image.stem.rsplit("_img", 1)[0]].append(item)
    queues = [groups[k] for k in sorted(groups)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    picked: list[PreviewItem] = []
    depth = 0
    while len(picked) < min(n, len(items)):
        picked.extend(q[depth] for q in queues if depth < len(q))
        depth += 1
    return picked[:n]


def _scaled(box: Box, scale: float) -> tuple[float, float, float, float]:
    return tuple(v * scale for v in box.to_xyxy())


def _hatch(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float) -> None:
    """Diagonal lines (bottom-left to top-right) clipped to the rectangle, plus an outline."""
    w, h = x1 - x0, y1 - y0
    k = 0.0
    while k < w + h:
        # line x - x0 + (y1 - y) = k, clipped to the box
        ax, ay = x0 + max(k - h, 0), y1 - min(k, h)
        bx, by = x0 + min(k, w), y1 - max(k - w, 0)
        draw.line([(ax, ay), (bx, by)], fill=IGNORE_COLOR, width=1)
        k += HATCH_STEP
    draw.rectangle([x0, y0, x1, y1], outline=IGNORE_COLOR, width=2)


def render_tile(item: PreviewItem, tile_width: int = TILE_WIDTH) -> Image.Image:
    with Image.open(item.image) as im:
        scale = tile_width / im.width
        tile = im.convert("RGB").resize((tile_width, max(round(im.height * scale), 1)))
    draw = ImageDraw.Draw(tile)
    for region in item.ignore:
        _hatch(draw, *_scaled(region, scale))
    for name, box in item.boxes:
        color = CLASS_COLORS.get(name, FALLBACK_COLOR)
        x0, y0, x1, y1 = _scaled(box, scale)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        left, top, right, bottom = draw.textbbox((x0, y0), name)
        ty = y0 - (bottom - top) - 2 if y0 >= bottom - top + 2 else y0
        draw.rectangle([x0, ty, x0 + right - left + 2, ty + bottom - top + 2], fill=color)
        draw.text((x0 + 1, ty + 1 - (top - y0)), name, fill="black")
    return tile


def render_grid(
    items: list[PreviewItem], out: Path, cols: int = GRID_COLUMNS, tile_width: int = TILE_WIDTH
) -> None:
    tiles = [render_tile(item, tile_width) for item in items]
    tile_height = max(t.height for t in tiles)
    n_rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * tile_width, n_rows * (tile_height + LABEL_HEIGHT)), "white")
    draw = ImageDraw.Draw(grid)
    for i, (item, tile) in enumerate(zip(items, tiles, strict=True)):
        x, y = (i % cols) * tile_width, (i // cols) * (tile_height + LABEL_HEIGHT)
        grid.paste(tile, (x, y + LABEL_HEIGHT))
        draw.text((x + 2, y + 1), f"{item.image.stem} ({len(item.boxes)})", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        grid.save(tmp, format="JPEG", quality=90)
