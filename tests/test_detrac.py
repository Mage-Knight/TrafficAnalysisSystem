import pytest
from conftest import FIXTURE_XML

from traffic_analytics.data.detrac import (
    Box,
    DetracError,
    covered_fraction,
    parse_sequence,
    read_sequence_info,
)


def test_parse_sequence_fields() -> None:
    seq = parse_sequence(FIXTURE_XML)
    assert seq.name == "MVI_00001"
    assert seq.weather == "sunny"  # from the misspelled sence_weather
    assert seq.camera_state == "unstable"
    assert seq.ignore_regions == (Box(40, 0, 24, 16),)
    assert [f.num for f in seq.frames] == [1, 2, 3]
    assert seq.frames[0].density == 2
    assert seq.num_boxes == 5
    assert seq.track_ids == {1, 2}
    assert seq.class_counts() == {"car": 3, "bus": 2, "van": 0, "others": 0}


def test_parse_target_fields() -> None:
    seq = parse_sequence(FIXTURE_XML)
    t = seq.frames[0].targets[0]
    assert t.track_id == 1
    assert t.box == Box(4.5, 20.25, 16, 12)
    assert t.vehicle_type == "car"
    assert not t.occluded
    assert t.visibility == 1.0
    assert t.attributes == {
        "orientation": "18.5",
        "speed": "6.86",
        "trajectory_length": "3",
        "truncation_ratio": "0",
        "vehicle_type": "car",
    }
    occluded = seq.frames[1].targets[0]
    assert occluded.occluded
    assert occluded.visibility == pytest.approx(0.5)
    assert seq.frames[2].targets[0].truncation_ratio == 0.25


def test_read_sequence_info() -> None:
    info = read_sequence_info(FIXTURE_XML)
    assert (info.name, info.weather, info.camera_state) == ("MVI_00001", "sunny", "unstable")


def test_unknown_vehicle_type_raises(tmp_path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text(FIXTURE_XML.read_text().replace('vehicle_type="bus"', 'vehicle_type="tank"', 1))
    with pytest.raises(DetracError, match="tank"):
        parse_sequence(bad)


def test_box_conversions() -> None:
    box = Box(56, 30, 12, 8)
    assert box.to_xyxy() == (56, 30, 68, 38)
    assert box.clip(64, 48) == Box(56, 30, 8, 8)
    assert Box(-5, -5, 10, 10).clip(64, 48) == Box(0, 0, 5, 5)
    assert Box(70, 0, 5, 5).clip(64, 48).area == 0
    assert Box(16, 12, 32, 24).to_yolo(64, 48) == (0.5, 0.5, 0.5, 0.5)


def test_covered_fraction_counts_overlap_once() -> None:
    box = Box(0, 0, 10, 10)
    assert covered_fraction(box, []) == 0
    assert covered_fraction(box, [Box(0, 0, 5, 10)]) == pytest.approx(0.5)
    # two overlapping halves cover 75%, not 100%
    assert covered_fraction(box, [Box(0, 0, 5, 10), Box(2.5, 0, 5, 10)]) == pytest.approx(0.75)
    assert covered_fraction(box, [Box(-5, -5, 50, 50)]) == 1.0
