"""UA-DETRAC annotation parsing: one sequence XML -> dataclasses.

XML layout (per sequence):
    <sequence name="MVI_20011">
      <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
      <ignored_region><box left top width height/>...</ignored_region>
      <frame density="7" num="1"><target_list>
        <target id="1">
          <box left top width height/>
          <attribute orientation speed trajectory_length truncation_ratio vehicle_type/>
          <occlusion><region_overlap occlusion_id occlusion_status left top width height/>
          </occlusion>  (optional)
        </target>
      </target_list></frame>
    </sequence>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

CLASSES: tuple[str, ...] = ("car", "bus", "van", "others")
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


class DetracError(ValueError):
    """Bad or inconsistent dataset input."""


@dataclass(frozen=True)
class Box:
    """Pixel box in DETRAC convention: left, top, width, height."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def intersection(self, other: Box) -> Box:
        left, top = max(self.left, other.left), max(self.top, other.top)
        right, bottom = min(self.right, other.right), min(self.bottom, other.bottom)
        return Box(left, top, max(right - left, 0.0), max(bottom - top, 0.0))

    def clip(self, width: float, height: float) -> Box:
        return self.intersection(Box(0.0, 0.0, width, height))

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def to_yolo(self, width: float, height: float) -> tuple[float, float, float, float]:
        """Normalized center x, center y, width, height."""
        return (
            (self.left + self.width / 2) / width,
            (self.top + self.height / 2) / height,
            self.width / width,
            self.height / height,
        )

    def as_list(self, ndigits: int = 2) -> list[float]:
        return [round(v, ndigits) for v in (self.left, self.top, self.width, self.height)]


def covered_fraction(box: Box, regions: Iterable[Box]) -> float:
    """Fraction of `box` covered by the union of `regions` (exact; overlaps counted once)."""
    if box.area <= 0:
        return 0.0
    parts = [p for p in (box.intersection(r) for r in regions) if p.area > 0]
    if not parts:
        return 0.0
    xs = sorted({x for p in parts for x in (p.left, p.right)})
    ys = sorted({y for p in parts for y in (p.top, p.bottom)})
    covered = 0.0
    for x0, x1 in zip(xs, xs[1:], strict=False):
        for y0, y1 in zip(ys, ys[1:], strict=False):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(p.left <= cx < p.right and p.top <= cy < p.bottom for p in parts):
                covered += (x1 - x0) * (y1 - y0)
    return min(covered / box.area, 1.0)


@dataclass(frozen=True)
class Target:
    track_id: int
    box: Box
    vehicle_type: str
    truncation_ratio: float
    occluded: bool
    occlusion_boxes: tuple[Box, ...] = ()
    attributes: dict[str, str] = field(default_factory=dict)  # every <attribute> field, raw

    @property
    def visibility(self) -> float:
        """1 - fraction of the box hidden by <occlusion> regions (truncation not included)."""
        return 1.0 - covered_fraction(self.box, self.occlusion_boxes)


@dataclass(frozen=True)
class Frame:
    num: int
    density: int | None
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class Sequence:
    name: str
    weather: str | None
    camera_state: str | None
    ignore_regions: tuple[Box, ...]
    frames: tuple[Frame, ...]

    @property
    def num_boxes(self) -> int:
        return sum(len(f.targets) for f in self.frames)

    @property
    def track_ids(self) -> set[int]:
        return {t.track_id for f in self.frames for t in f.targets}

    def class_counts(self) -> dict[str, int]:
        counts = Counter(t.vehicle_type for f in self.frames for t in f.targets)
        return {c: counts.get(c, 0) for c in CLASSES}


@dataclass(frozen=True)
class SequenceInfo:
    """Sequence-level attributes only; cheap to read (see `read_sequence_info`)."""

    name: str
    weather: str | None
    camera_state: str | None


def read_box(el: ET.Element) -> Box:
    return Box(*(float(el.get(k, 0)) for k in ("left", "top", "width", "height")))


def _parse_target(el: ET.Element, frame_num: int, xml_path: Path) -> Target:
    box_el = el.find("box")
    attr_el = el.find("attribute")
    if box_el is None:
        raise DetracError(f"{xml_path}: frame {frame_num}, target {el.get('id')}: no <box>")
    attributes = dict(attr_el.attrib) if attr_el is not None else {}
    vehicle_type = attributes.get("vehicle_type", "")
    if vehicle_type not in CLASS_IDS:
        raise DetracError(
            f"{xml_path}: frame {frame_num}, target {el.get('id')}: "
            f"unknown vehicle_type {vehicle_type!r} (expected one of {CLASSES})"
        )
    occlusion = el.find("occlusion")
    occlusion_boxes = (
        tuple(read_box(r) for r in occlusion.iter("region_overlap"))
        if occlusion is not None
        else ()
    )
    return Target(
        track_id=int(el.get("id", -1)),
        box=read_box(box_el),
        vehicle_type=vehicle_type,
        truncation_ratio=float(attributes.get("truncation_ratio", 0) or 0),
        occluded=occlusion is not None,
        occlusion_boxes=occlusion_boxes,
        attributes=attributes,
    )


def parse_sequence(xml_path: Path) -> Sequence:
    root = ET.parse(xml_path).getroot()
    seq_attr = root.find("sequence_attribute")
    ignored = root.find("ignored_region")

    frames = []
    for frame_el in root.iter("frame"):
        num = int(frame_el.get("num", 0))
        density = frame_el.get("density")
        targets = tuple(_parse_target(t, num, xml_path) for t in frame_el.iter("target"))
        frames.append(Frame(num, int(density) if density is not None else None, targets))
    frames.sort(key=lambda f: f.num)

    return Sequence(
        name=root.get("name") or xml_path.stem,
        # "sence_weather" is the dataset's own spelling
        weather=seq_attr.get("sence_weather") if seq_attr is not None else None,
        camera_state=seq_attr.get("camera_state") if seq_attr is not None else None,
        ignore_regions=(
            tuple(read_box(b) for b in ignored.iter("box")) if ignored is not None else ()
        ),
        frames=tuple(frames),
    )


def read_sequence_info(xml_path: Path) -> SequenceInfo:
    """Read name, weather and camera state without parsing all frames."""
    name = xml_path.stem
    for event, el in ET.iterparse(xml_path, events=("start",)):
        if event == "start" and el.tag == "sequence":
            name = el.get("name") or name
        elif el.tag == "sequence_attribute":
            return SequenceInfo(name, el.get("sence_weather"), el.get("camera_state"))
        elif el.tag == "frame":
            break
    return SequenceInfo(name, None, None)
