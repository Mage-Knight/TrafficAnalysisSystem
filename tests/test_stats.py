import csv

import pytest
from conftest import FIXTURE_XML, IMAGE_SIZE
from PIL import Image
from typer.testing import CliRunner

from traffic_analytics.cli import app
from traffic_analytics.data.detrac import parse_sequence
from traffic_analytics.data.stats import group_summary, mvi_group, sequence_stats


def test_mvi_group() -> None:
    assert mvi_group("MVI_40171") == "401"
    assert mvi_group("MVI_20011") == "200"
    assert mvi_group("odd_name") == "odd_name"


def test_sequence_stats_on_fixture() -> None:
    row = sequence_stats(parse_sequence(FIXTURE_XML), "train", IMAGE_SIZE)
    assert (row["group"], row["weather"], row["camera_state"]) == ("000", "sunny", "unstable")
    assert (row["frames"], row["boxes"], row["tracks"]) == (3, 5, 2)
    assert (row["n_car"], row["n_bus"], row["n_van"], row["n_others"]) == (3, 2, 0, 0)
    assert row["density_mean"] == pytest.approx(5 / 3, abs=0.01)
    assert row["density_max"] == 2
    assert row["small_frac"] + row["medium_frac"] + row["large_frac"] == pytest.approx(1)
    assert row["occluded_frac"] == pytest.approx(1 / 5)
    assert row["heavy_occl_frac"] == 0  # visibility 0.5 is not heavy
    assert row["trunc_mean"] == pytest.approx((0.33 + 0.25) / 5, abs=1e-4)
    assert row["ignore_area_frac"] == pytest.approx(384 / 3072)


def test_group_summary_flags_cross_folder_groups() -> None:
    rows = [
        {"group": "401", "folder": "train", "weather": "sunny", "boxes": 10},
        {"group": "401", "folder": "test", "weather": "night", "boxes": 5},
        {"group": "200", "folder": "train", "weather": "cloudy", "boxes": 7},
    ]
    lines = group_summary(rows)
    assert lines[0] == "group 200: 1 train, 0 test, 7 boxes, cloudy"
    assert lines[1].endswith("<- spans train and test")


def test_stats_cli(detrac_root, tmp_path) -> None:
    out = tmp_path / "stats"
    result = CliRunner().invoke(
        app, ["data", "stats", "--root", str(detrac_root), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "group 000: 2 train, 1 test, 15 boxes, sunny  <- spans train and test" in result.output
    with (out / "sequence_stats.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["sequence"] for r in rows] == ["MVI_00001", "MVI_00003", "MVI_00002"]
    assert Image.open(out / "contact_sheet.jpg").size == (1600, 102)
