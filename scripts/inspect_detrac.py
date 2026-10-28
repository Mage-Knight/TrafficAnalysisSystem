"""Quick look at a UA-DETRAC download: annotation summary and one drawn frame.

Usage (on Colab):
    python scripts/inspect_detrac.py --root /content/detrac --out /content/sample.png
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

Box = tuple[float, float, float, float]  # left, top, width, height


def read_box(el: ET.Element) -> Box:
    return tuple(float(el.get(k, 0)) for k in ("left", "top", "width", "height"))  # type: ignore[return-value]


def summarize(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    seq_attr = root.find("sequence_attribute")
    ignored = root.find("ignored_region")

    classes: Counter[str] = Counter()
    track_ids: set[str] = set()
    n_boxes = n_occluded = 0
    frames = root.findall("frame")

    for frame in frames:
        for target in frame.iter("target"):
            n_boxes += 1
            track_ids.add(target.get("id", ""))
            attr = target.find("attribute")
            if attr is not None:
                classes[attr.get("vehicle_type", "?")] += 1
            if target.find("occlusion") is not None:
                n_occluded += 1

    first = frames[0] if frames else None
    return {
        "sequence": root.get("name", xml_path.stem),
        # "sence_weather" is the dataset's own spelling
        "weather": seq_attr.get("sence_weather") if seq_attr is not None else None,
        "camera": seq_attr.get("camera_state") if seq_attr is not None else None,
        "frames": len(frames),
        "boxes": n_boxes,
        "tracks": len(track_ids),
        "occluded_boxes": n_occluded,
        "classes": dict(classes),
        "ignored_regions": [read_box(b) for b in ignored.findall("box")]
        if ignored is not None
        else [],
        "first_frame_num": int(first.get("num", 1)) if first is not None else None,
        "first_frame_boxes": [read_box(b) for b in first.iter("box")] if first is not None else [],
    }


def find_image(root: Path, sequence: str, frame_num: int) -> Path | None:
    for seq_dir in root.rglob(sequence):
        candidate = seq_dir / f"img{frame_num:05d}.jpg"
        if candidate.is_file():
            return candidate
    return None


def draw(image_path: Path, info: dict, out: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    pen = ImageDraw.Draw(img)
    for left, top, w, h in info["ignored_regions"]:
        pen.rectangle([left, top, left + w, top + h], outline="yellow", width=3)
    for left, top, w, h in info["first_frame_boxes"]:
        pen.rectangle([left, top, left + w, top + h], outline="red", width=2)
    img.save(out)
    print(f"image size: {img.size}  ->  drawn frame saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="unzipped dataset folder")
    parser.add_argument("--n-xml", type=int, default=3, help="how many sequences to summarize")
    parser.add_argument("--out", type=Path, default=Path("sample.png"))
    args = parser.parse_args()

    xmls = sorted(args.root.rglob("*.xml"))
    print(f"found {len(xmls)} annotation XML files")
    if not xmls:
        raise SystemExit("no XML files: this mirror may not include the original annotations")

    total: Counter[str] = Counter()
    first_info = None
    for xml_path in xmls[: args.n_xml]:
        info = summarize(xml_path)
        first_info = first_info or info
        total.update(info["classes"])
        print(
            f"\n{info['sequence']}: weather={info['weather']} camera={info['camera']} "
            f"frames={info['frames']} boxes={info['boxes']} tracks={info['tracks']} "
            f"occluded={info['occluded_boxes']} ignored_regions={len(info['ignored_regions'])}"
        )
        print(f"  classes: {info['classes']}")

    print(f"\nclasses in the first {args.n_xml} sequences: {dict(total)}")

    image = find_image(args.root, first_info["sequence"], first_info["first_frame_num"])
    if image is None:
        print(f"no image found for {first_info['sequence']}; check the image folder layout")
    else:
        draw(image, first_info, args.out)


if __name__ == "__main__":
    main()
