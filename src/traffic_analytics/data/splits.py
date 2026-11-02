"""Train / val / test split by scene.

Test is exactly the Test annotation folder. Val is a set of whole scenes from the Train folder
(a scene = sequences sharing one camera viewpoint, see configs/scenes.yaml), so near-duplicate
sequences never land in both train and val. Scenes are only grouping labels; sequences keep
their own names in every output.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Collection, Iterable
from pathlib import Path

import yaml

from traffic_analytics.data.detrac import CLASSES, DetracError
from traffic_analytics.data.files import write_text

SPLITS = ("train", "val", "test")

TARGET_SHARE = 1 / 6  # val's ideal share of every train-folder total
SCENE_BONUS = 0.002  # score reward per distinct val scene (diversity tie-breaker)
# Totals whose val share should be close to TARGET_SHARE; each weighs equally in the score.
BALANCE_KEYS = ("boxes", *(f"n_{c}" for c in CLASSES), "small", "occluded", "heavy_occl")

_INT_COLUMNS = {"frames", "boxes", "tracks", "density_max", *(f"n_{c}" for c in CLASSES)}
_STR_COLUMNS = {"sequence", "folder", "group", "weather", "camera_state"}


def load_stats(path: Path) -> list[dict]:
    """Read sequence_stats.csv (from `ta data stats`) with typed values."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    typed = []
    for row in rows:
        typed.append(
            {
                k: v if k in _STR_COLUMNS else int(v) if k in _INT_COLUMNS else float(v)
                for k, v in row.items()
            }
        )
    return typed


def load_scenes(path: Path) -> dict[str, str]:
    """configs/scenes.yaml (`scenes: {scene: [sequence, ...]}`) -> {sequence: scene}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), dict):
        raise DetracError(f"{path}: expected a top-level 'scenes' mapping")
    out: dict[str, str] = {}
    for scene, names in data["scenes"].items():
        if not isinstance(names, list) or not names:
            raise DetracError(f"{path}: scene {scene} must be a non-empty list of sequences")
        for name in names:
            if name in out:
                raise DetracError(f"{path}: {name} is in scenes {out[name]} and {scene}")
            out[name] = str(scene)
    return out


def validate_scenes(scenes: dict[str, str], stats: Iterable[dict]) -> None:
    """Every sequence has exactly one scene; no unknown names; no scene spans both folders."""
    folders = {r["sequence"]: r["folder"] for r in stats}
    if missing := sorted(folders.keys() - scenes.keys()):
        raise DetracError(f"sequences without a scene: {missing}")
    if unknown := sorted(scenes.keys() - folders.keys()):
        raise DetracError(f"scenes list unknown sequences: {unknown}")
    scene_folders: dict[str, set[str]] = defaultdict(set)
    for name, scene in scenes.items():
        scene_folders[scene].add(folders[name])
    if mixed := sorted(s for s, f in scene_folders.items() if len(f) > 1):
        raise DetracError(f"scenes mixing train and test folders: {mixed}")


def _totals(rows: Iterable[dict]) -> dict[str, float]:
    t = dict.fromkeys(BALANCE_KEYS, 0.0)
    for r in rows:
        t["boxes"] += r["boxes"]
        for c in CLASSES:
            t[f"n_{c}"] += r[f"n_{c}"]
        t["small"] += r["small_frac"] * r["boxes"]
        t["occluded"] += r["occluded_frac"] * r["boxes"]
        t["heavy_occl"] += r["heavy_occl_frac"] * r["boxes"]
    return t


def _shares(val: dict[str, float], folder: dict[str, float]) -> dict[str, float]:
    return {k: val[k] / folder[k] if folder[k] else 0.0 for k in BALANCE_KEYS}


def _weather_boxes(rows: Iterable[dict]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for r in rows:
        out[r["weather"]] += r["boxes"]
    return out


def weather_tvd(val: dict[str, float], folder: dict[str, float]) -> float:
    """Total-variation distance between box-weighted weather distributions (0 = identical)."""
    val_total, folder_total = sum(val.values()), sum(folder.values())
    if not val_total or not folder_total:
        return 0.0
    return 0.5 * sum(
        abs(val.get(w, 0.0) / val_total - folder.get(w, 0.0) / folder_total)
        for w in val.keys() | folder.keys()
    )


def _unstable_frac(rows: Iterable[dict]) -> float:
    rows = list(rows)
    return sum(r["camera_state"] == "unstable" for r in rows) / len(rows) if rows else 0.0


def make_grouped_split(
    stats: list[dict],
    scenes: dict[str, str],
    seed: int = 0,
    candidates: int = 20000,
    min_box_share: float = 0.13,
    max_box_share: float = 0.20,
    min_class_share: float = 0.15,
) -> dict[str, list[str]]:
    """Pick whole train-folder scenes as val.

    Candidates: `candidates` seeded shuffles of the scenes; each prefix of a shuffle whose val
    box share is in [min_box_share, max_box_share] is a candidate.
    Hard constraints: every train-folder weather has >= 1 scene in val and in train; val holds
    >= min_class_share of each class's train-folder boxes.
    Score (lower is better): mean of |val share - TARGET_SHARE| over BALANCE_KEYS and the
    box-weighted weather TVD between val and the train folder (all equally weighted)
    + |unstable-camera fraction of val sequences - that of the train folder|
    - SCENE_BONUS * number of val scenes. Ties keep the earliest candidate.
    """
    validate_scenes(scenes, stats)
    train_rows = [r for r in stats if r["folder"] == "train"]
    test_names = sorted(r["sequence"] for r in stats if r["folder"] == "test")
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_scene[scenes[r["sequence"]]].append(r)
    scene_ids = sorted(by_scene)
    scene_totals = {s: _totals(rows) for s, rows in by_scene.items()}
    scene_weathers = {s: {r["weather"] for r in rows} for s, rows in by_scene.items()}
    scene_weather_boxes = {s: _weather_boxes(rows) for s, rows in by_scene.items()}
    folder = _totals(train_rows)
    folder_weather_boxes = _weather_boxes(train_rows)
    weathers = {r["weather"] for r in train_rows}
    folder_unstable = _unstable_frac(train_rows)

    rng = random.Random(seed)
    seen: set[frozenset[str]] = set()
    best: tuple[float, frozenset[str]] | None = None
    best_min_class = 0.0
    n_in_range = n_weather_ok = 0
    for _ in range(candidates):
        order = scene_ids[:]
        rng.shuffle(order)
        acc = dict.fromkeys(BALANCE_KEYS, 0.0)
        chosen: list[str] = []
        for scene in order:
            chosen.append(scene)
            for k in BALANCE_KEYS:
                acc[k] += scene_totals[scene][k]
            box_share = acc["boxes"] / folder["boxes"]
            if box_share > max_box_share:
                break
            key = frozenset(chosen)
            if box_share < min_box_share or key in seen:
                continue
            seen.add(key)
            n_in_range += 1

            val_weathers = set().union(*(scene_weathers[s] for s in key))
            train_weathers = set().union(*(scene_weathers[s] for s in scene_ids if s not in key))
            if val_weathers != weathers or train_weathers != weathers:
                continue
            n_weather_ok += 1
            shares = _shares(acc, folder)
            min_class = min(shares[f"n_{c}"] for c in CLASSES if folder[f"n_{c}"])
            best_min_class = max(best_min_class, min_class)
            if min_class < min_class_share:
                continue

            val_rows = [r for s in key for r in by_scene[s]]
            val_weather_boxes: dict[str, float] = defaultdict(float)
            for s in key:
                for w, n in scene_weather_boxes[s].items():
                    val_weather_boxes[w] += n
            balance = [abs(shares[k] - TARGET_SHARE) for k in BALANCE_KEYS]
            balance.append(weather_tvd(val_weather_boxes, folder_weather_boxes))
            score = (
                sum(balance) / len(balance)
                + abs(_unstable_frac(val_rows) - folder_unstable)
                - SCENE_BONUS * len(key)
            )
            if best is None or score < best[0]:
                best = (score, key)

    if best is None:
        raise DetracError(
            f"no feasible split in {candidates} shuffles: {n_in_range} candidates with box share "
            f"in [{min_box_share}, {max_box_share}], {n_weather_ok} of them cover every weather "
            f"in train and val; best min class share {best_min_class:.3f} < {min_class_share}"
        )
    val = sorted(r["sequence"] for s in best[1] for r in by_scene[s])
    return {
        "train": sorted(r["sequence"] for r in train_rows if r["sequence"] not in val),
        "val": val,
        "test": test_names,
    }


def split_report(
    stats: list[dict], scenes: dict[str, str], splits: dict[str, list[str]]
) -> list[str]:
    """Val scenes and val's share of each train-folder total (for the splits.yaml header)."""
    train_rows = [r for r in stats if r["folder"] == "train"]
    val_names = set(splits["val"])
    val_rows = [r for r in train_rows if r["sequence"] in val_names]
    members: dict[str, list[str]] = defaultdict(list)
    for name in sorted(val_names):
        members[scenes[name]].append(name)
    scene_weathers: dict[str, set[str]] = defaultdict(set)
    for r in val_rows:
        scene_weathers[scenes[r["sequence"]]].add(r["weather"])
    shares = _shares(_totals(val_rows), _totals(train_rows))
    fit_rows = [r for r in train_rows if r["sequence"] not in val_names]
    weathers = sorted({r["weather"] for r in train_rows})

    def weather_frac(rows: list[dict], weather: str) -> float:
        return sum(r["weather"] == weather for r in rows) / len(rows) if rows else 0.0

    return [
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
        "sequences",
        "val scenes:",
        *(
            f"  {s} ({', '.join(sorted(scene_weathers[s]))}): {', '.join(names)}"
            for s, names in sorted(members.items())
        ),
        "val share of train-folder totals: "
        + " ".join(f"{k.removeprefix('n_')}={v:.3f}" for k, v in shares.items()),
        "weather share of sequences, val vs train: "
        + ", ".join(
            f"{w} {weather_frac(val_rows, w):.2f} vs {weather_frac(fit_rows, w):.2f}"
            for w in weathers
        ),
        "weather TVD, box-weighted, val vs train folder: "
        f"{weather_tvd(_weather_boxes(val_rows), _weather_boxes(train_rows)):.3f}",
        f"unstable-camera sequences: val {_unstable_frac(val_rows):.2f} "
        f"vs train folder {_unstable_frac(train_rows):.2f}",
    ]


def save_splits(path: Path, splits: dict[str, list[str]], header: Iterable[str] = ()) -> None:
    comments = "".join(f"# {line}\n" for line in header)
    body = yaml.safe_dump(
        {k: sorted(splits[k]) for k in SPLITS}, sort_keys=False, default_flow_style=False
    )
    write_text(path, comments + body)


def load_splits(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != set(SPLITS):
        raise DetracError(f"{path}: expected keys {SPLITS}")
    for k in SPLITS:
        if not isinstance(data[k], list) or not all(isinstance(n, str) for n in data[k]):
            raise DetracError(f"{path}: '{k}' must be a list of sequence names")
    return {k: list(data[k]) for k in SPLITS}


def validate_splits(
    splits: dict[str, list[str]], train_names: Collection[str], test_names: Collection[str]
) -> None:
    """Splits must be disjoint, test must equal the Test folder, train+val the Train folder."""
    test_folder, train_folder = set(test_names), set(train_names)
    if leaked := sorted((set(splits["train"]) | set(splits["val"])) & test_folder):
        raise DetracError(f"test-folder sequences in train/val (leakage): {leaked}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if overlap := sorted(set(splits[a]) & set(splits[b])):
            raise DetracError(f"sequences in both {a} and {b}: {overlap}")
    if misplaced := sorted(set(splits["test"]) & train_folder):
        raise DetracError(f"train-folder sequences in test: {misplaced}")
    for label, listed, found in (
        ("train+val", set(splits["train"]) | set(splits["val"]), train_folder),
        ("test", set(splits["test"]), test_folder),
    ):
        if missing := sorted(found - listed):
            raise DetracError(f"{label}: discovered sequences missing from splits: {missing}")
        if unknown := sorted(listed - found):
            raise DetracError(f"{label}: split sequences not found in the data: {unknown}")
