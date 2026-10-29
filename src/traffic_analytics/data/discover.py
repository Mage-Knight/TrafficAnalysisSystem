"""Find annotation XMLs (Train vs Test folders) and image folders in a DETRAC download."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from traffic_analytics.data.detrac import DetracError

EXPECTED_COUNTS = {"train": 60, "test": 40}


@dataclass
class Discovery:
    train: dict[str, Path]  # sequence name -> XML path, from the Train annotation folder
    test: dict[str, Path]  # from the Test annotation folder
    image_dirs: dict[str, Path]  # sequence name -> folder with img00001.jpg ...
    warnings: list[str] = field(default_factory=list)

    @property
    def xml_folders(self) -> list[Path]:
        return sorted({p.parent for p in (*self.train.values(), *self.test.values())})

    @property
    def image_folders(self) -> list[Path]:
        return sorted({p.parent for p in self.image_dirs.values()})


def _split_of(xml: Path, root: Path) -> str | None:
    """Nearest ancestor folder whose name mentions train or test decides the split."""
    for part in reversed(xml.relative_to(root).parent.parts):
        low = part.lower()
        if "train" in low:
            return "train"
        if "test" in low:
            return "test"
    return None


def _walk(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    """One pass over the tree: all XML files and all directories by name."""
    xmls: list[Path] = []
    dirs: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        xmls.extend(base / f for f in filenames if f.lower().endswith(".xml"))
        for d in dirnames:
            dirs.setdefault(d, []).append(base / d)
    return sorted(xmls), dirs


def _by_name(xmls: list[Path], split: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for xml in xmls:
        if xml.stem in out:
            raise DetracError(f"{split}: sequence {xml.stem} found twice: {out[xml.stem]}, {xml}")
        out[xml.stem] = xml
    return out


def discover(
    root: Path,
    train_xml: Path | None = None,
    test_xml: Path | None = None,
    images: Path | None = None,
) -> Discovery:
    if not root.is_dir():
        raise DetracError(f"--root {root} is not a directory")
    xmls, dirs = _walk(root)

    auto: dict[str, list[Path]] = {"train": [], "test": []}
    unassigned: list[Path] = []
    for xml in xmls:
        split = _split_of(xml, root)
        (auto[split] if split else unassigned).append(xml)

    if (train_xml is None or test_xml is None) and unassigned:
        listed = ", ".join(str(p) for p in unassigned[:5])
        raise DetracError(
            f"{len(unassigned)} XML files are not under a Train or Test folder ({listed} ...); "
            "pass --train-xml and --test-xml"
        )
    train = _by_name(sorted(train_xml.rglob("*.xml")) if train_xml else auto["train"], "train")
    test = _by_name(sorted(test_xml.rglob("*.xml")) if test_xml else auto["test"], "test")
    if both := sorted(train.keys() & test.keys()):
        raise DetracError(f"sequences in both Train and Test annotation folders: {both}")

    warnings: list[str] = []
    for split, found in (("train", train), ("test", test)):
        if len(found) != EXPECTED_COUNTS[split]:
            warnings.append(
                f"WARNING: expected {EXPECTED_COUNTS[split]} {split} sequences, found {len(found)}"
            )

    if images is not None:
        _, dirs = _walk(images)
    image_dirs: dict[str, Path] = {}
    for name in sorted(train.keys() | test.keys()):
        candidates = [d for d in sorted(dirs.get(name, [])) if any(d.glob("img*.jpg"))]
        if candidates:
            image_dirs[name] = candidates[0]
        else:
            warnings.append(f"WARNING: no image folder found for {name}")

    return Discovery(train, test, image_dirs, warnings)
