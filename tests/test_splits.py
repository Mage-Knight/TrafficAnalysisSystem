import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from traffic_analytics.cli import app
from traffic_analytics.data.detrac import CLASSES, DetracError
from traffic_analytics.data.splits import (
    load_scenes,
    load_splits,
    load_stats,
    make_grouped_split,
    save_splits,
    validate_scenes,
    validate_splits,
    weather_tvd,
)

REPO = Path(__file__).parents[1]
REAL_STATS = REPO / "reports" / "data" / "sequence_stats.csv"
REAL_SCENES = REPO / "configs" / "scenes.yaml"
WEATHERS = ("sunny", "cloudy", "rainy", "night")


def _row(name: str, folder: str, weather: str, boxes: int = 1000, camera="stable") -> dict:
    return {
        "sequence": name,
        "folder": folder,
        "weather": weather,
        "camera_state": camera,
        "boxes": boxes,
        **{f"n_{c}": boxes // 4 for c in CLASSES},
        "small_frac": 0.3,
        "occluded_frac": 0.2,
        "heavy_occl_frac": 0.05,
    }


def _synthetic() -> tuple[list[dict], dict[str, str]]:
    """24 train scenes (6 per weather, 2 sequences each) + 4 test scenes."""
    stats, scenes = [], {}
    for w_i, weather in enumerate(WEATHERS):
        for s_i in range(6):
            scene = f"s{w_i}{s_i}"
            for q in range(2):
                name = f"MVI_{w_i}{s_i}{q}"
                camera = "unstable" if (s_i + q) % 2 else "stable"
                stats.append(_row(name, "train", weather, 1000 + 100 * s_i, camera))
                scenes[name] = scene
        stats.append(_row(f"MVI_T{w_i}", "test", weather))
        scenes[f"MVI_T{w_i}"] = f"t{w_i}"
    return stats, scenes


def _val_share(stats: list[dict], val: set[str], key: str) -> float:
    train = [r for r in stats if r["folder"] == "train"]
    return sum(r[key] for r in train if r["sequence"] in val) / sum(r[key] for r in train)


def _check(stats, scenes, splits, min_box=0.13, max_box=0.20, min_class=0.15) -> None:
    val = set(splits["val"])
    train_seqs = [r for r in stats if r["folder"] == "train"]
    validate_splits(
        splits,
        [r["sequence"] for r in train_seqs],
        [r["sequence"] for r in stats if r["folder"] == "test"],
    )
    val_scenes = {scenes[n] for n in val}
    train_scenes = {scenes[n] for n in splits["train"]}
    assert not val_scenes & train_scenes
    assert min_box <= _val_share(stats, val, "boxes") <= max_box
    for c in CLASSES:
        assert _val_share(stats, val, f"n_{c}") >= min_class
    weathers = {r["weather"] for r in train_seqs}
    assert {r["weather"] for r in train_seqs if r["sequence"] in val} == weathers
    assert {r["weather"] for r in train_seqs if r["sequence"] not in val} == weathers


def test_grouped_split_synthetic() -> None:
    stats, scenes = _synthetic()
    splits = make_grouped_split(stats, scenes, candidates=2000)
    _check(stats, scenes, splits)
    assert splits["test"] == ["MVI_T0", "MVI_T1", "MVI_T2", "MVI_T3"]


def test_grouped_split_deterministic() -> None:
    stats, scenes = _synthetic()
    first = make_grouped_split(stats, scenes, seed=3, candidates=500)
    assert first == make_grouped_split(stats, scenes, seed=3, candidates=500)


def test_grouped_split_infeasible() -> None:
    stats, scenes = _synthetic()
    # one scene per weather can't be both in train and val
    one_scene = {n: (f"w{n[4]}" if not n.startswith("MVI_T") else s) for n, s in scenes.items()}
    with pytest.raises(DetracError, match="no feasible split"):
        make_grouped_split(stats, one_scene, candidates=200)


def test_validate_scenes() -> None:
    stats, scenes = _synthetic()
    validate_scenes(scenes, stats)
    with pytest.raises(DetracError, match="without a scene"):
        validate_scenes({k: v for k, v in scenes.items() if k != "MVI_000"}, stats)
    with pytest.raises(DetracError, match="unknown"):
        validate_scenes({**scenes, "MVI_99999": "x"}, stats)
    with pytest.raises(DetracError, match="mixing"):
        validate_scenes({**scenes, "MVI_T0": "s00"}, stats)


def test_load_scenes_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "scenes.yaml"
    path.write_text(yaml.safe_dump({"scenes": {"a": ["MVI_1"], "b": ["MVI_1"]}}))
    with pytest.raises(DetracError, match="MVI_1"):
        load_scenes(path)


def test_validate_rejects_test_leakage_and_overlap() -> None:
    stats, scenes = _synthetic()
    train = [r["sequence"] for r in stats if r["folder"] == "train"]
    test = [r["sequence"] for r in stats if r["folder"] == "test"]
    splits = make_grouped_split(stats, scenes, candidates=500)
    with pytest.raises(DetracError, match="leakage"):
        validate_splits({**splits, "val": [*splits["val"], test[0]]}, train, test)
    with pytest.raises(DetracError, match="both train and val"):
        validate_splits({**splits, "val": [*splits["val"], splits["train"][0]]}, train, test)


def test_real_stats_and_scenes_feasible() -> None:
    stats, scenes = load_stats(REAL_STATS), load_scenes(REAL_SCENES)
    validate_scenes(scenes, stats)
    splits = make_grouped_split(stats, scenes)
    _check(stats, scenes, splits)
    assert len(splits["train"]) + len(splits["val"]) == 60 and len(splits["test"]) == 40


def test_split_cli_writes_report(tmp_path) -> None:
    out = tmp_path / "splits.yaml"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "split",
            "--stats",
            str(REAL_STATS),
            "--scenes",
            str(REAL_SCENES),
            "--out",
            str(out),
            "--candidates",
            "2000",
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert text.startswith("# UA-DETRAC split by scene")
    assert "# val scenes:" in text
    assert "# weather TVD, box-weighted, val vs train folder: " in text
    assert re.search(r"^# weather share of sequences, val vs train: cloudy \d\.\d\d vs", text, re.M)
    assert re.search(r"^#   s\d{4} \((sunny|cloudy|rainy|night)\): MVI_\d{5}", text, re.M)
    splits = load_splits(out)
    assert len(splits["test"]) == 40
    save_splits(tmp_path / "copy.yaml", splits)
    assert load_splits(tmp_path / "copy.yaml") == splits


def test_weather_tvd() -> None:
    assert weather_tvd({"sunny": 10, "night": 10}, {"sunny": 50, "night": 50}) == 0
    assert weather_tvd({"sunny": 10}, {"night": 7}) == 1
    assert weather_tvd({"sunny": 3, "night": 1}, {"sunny": 1, "night": 1}) == pytest.approx(0.25)
