"""Train / val / test split by sequence.

Test is exactly the Test annotation folder. Val is held out from the Train folder,
stratified by (weather, camera_state), with a fixed seed.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Collection, Iterable
from pathlib import Path

import yaml

from traffic_analytics.data.detrac import DetracError, SequenceInfo
from traffic_analytics.data.files import write_text

SPLITS = ("train", "val", "test")
Stratum = tuple[str, str]


def stratum(info: SequenceInfo) -> Stratum:
    return info.weather or "unknown", info.camera_state or "unknown"


def allocate(counts: dict[Stratum, int], n: int) -> dict[Stratum, int]:
    """Split n across strata proportionally to counts (largest-remainder rounding)."""
    total = sum(counts.values())
    quotas = {k: n * c / total for k, c in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    by_remainder = sorted(counts, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in by_remainder[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def make_splits(
    train_infos: Iterable[SequenceInfo], test_names: Iterable[str], n_val: int = 10, seed: int = 0
) -> dict[str, list[str]]:
    infos = sorted(train_infos, key=lambda i: i.name)
    if not 0 < n_val < len(infos):
        raise DetracError(f"n_val={n_val} must be between 1 and {len(infos) - 1}")

    strata: dict[Stratum, list[str]] = {}
    for info in infos:
        strata.setdefault(stratum(info), []).append(info.name)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n_val)

    rng = random.Random(seed)
    val: set[str] = set()
    for key in sorted(strata):
        val.update(rng.sample(strata[key], alloc[key]))

    return {
        "train": sorted(i.name for i in infos if i.name not in val),
        "val": sorted(val),
        "test": sorted(test_names),
    }


def describe(infos: Iterable[SequenceInfo], splits: dict[str, list[str]]) -> list[str]:
    """One line per (weather, camera_state) stratum: train and val counts."""
    val = set(splits["val"])
    totals: Counter[Stratum] = Counter()
    in_val: Counter[Stratum] = Counter()
    for info in infos:
        totals[stratum(info)] += 1
        in_val[stratum(info)] += info.name in val
    return [
        f"{w}/{c}: {totals[(w, c)] - in_val[(w, c)]} train, {in_val[(w, c)]} val"
        for w, c in sorted(totals)
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
