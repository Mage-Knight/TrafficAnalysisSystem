import math
from collections import Counter

import pytest

from traffic_analytics.data.detrac import DetracError, SequenceInfo
from traffic_analytics.data.splits import (
    load_splits,
    make_splits,
    save_splits,
    stratum,
    validate_splits,
)

# 60 train sequences with uneven strata, like DETRAC
STRATA = {
    ("sunny", "stable"): 14,
    ("sunny", "unstable"): 6,
    ("cloudy", "stable"): 18,
    ("rainy", "stable"): 8,
    ("night", "stable"): 10,
    ("night", "unstable"): 4,
}
TRAIN = [SequenceInfo(f"MVI_{w}_{c}_{i}", w, c) for (w, c), n in STRATA.items() for i in range(n)]
TEST = [f"MVI_test_{i}" for i in range(40)]


def test_split_disjoint_and_complete() -> None:
    splits = make_splits(TRAIN, TEST, n_val=10, seed=0)
    assert len(splits["val"]) == 10
    assert len(splits["train"]) == 50
    assert splits["test"] == sorted(TEST)
    assert not set(splits["train"]) & set(splits["val"])
    assert not (set(splits["train"]) | set(splits["val"])) & set(TEST)
    validate_splits(splits, [i.name for i in TRAIN], TEST)


def test_split_deterministic_and_seed_dependent() -> None:
    assert make_splits(TRAIN, TEST, seed=0) == make_splits(TRAIN, TEST, seed=0)
    assert make_splits(TRAIN, TEST, seed=0)["val"] != make_splits(TRAIN, TEST, seed=1)["val"]


def test_split_stratified_proportionally() -> None:
    by_name = {i.name: stratum(i) for i in TRAIN}
    val_counts = Counter(by_name[n] for n in make_splits(TRAIN, TEST, n_val=10)["val"])
    for key, n in STRATA.items():
        quota = 10 * n / len(TRAIN)
        assert math.floor(quota) <= val_counts[key] <= math.ceil(quota)


def test_validate_rejects_test_leakage() -> None:
    splits = make_splits(TRAIN, TEST)
    splits["val"].append(TEST[0])
    with pytest.raises(DetracError, match="leakage"):
        validate_splits(splits, [i.name for i in TRAIN], TEST)


def test_validate_rejects_overlap_and_unknown() -> None:
    names = [i.name for i in TRAIN]
    splits = make_splits(TRAIN, TEST)
    splits["val"].append(splits["train"][0])
    with pytest.raises(DetracError, match="both train and val"):
        validate_splits(splits, names, TEST)
    splits = make_splits(TRAIN, TEST)
    splits["train"].pop()
    with pytest.raises(DetracError, match="missing"):
        validate_splits(splits, names, TEST)


def test_save_load_roundtrip(tmp_path) -> None:
    splits = make_splits(TRAIN, TEST)
    path = tmp_path / "splits.yaml"
    save_splits(path, splits, ["seed=0"])
    assert path.read_text().startswith("# seed=0\n")
    assert load_splits(path) == splits
