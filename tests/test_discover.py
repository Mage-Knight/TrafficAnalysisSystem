import pytest

from traffic_analytics.data.detrac import DetracError
from traffic_analytics.data.discover import discover


def test_discover_by_folder_name(detrac_root) -> None:
    found = discover(detrac_root)
    assert sorted(found.train) == ["MVI_00001", "MVI_00003"]
    assert sorted(found.test) == ["MVI_00002"]
    assert found.image_dirs["MVI_00002"] == detrac_root / "Insight-MVT_Annotation" / "MVI_00002"
    assert any("expected 40 test sequences, found 1" in w for w in found.warnings)


def test_discover_rejects_sequence_in_both(detrac_root) -> None:
    src = detrac_root / "DETRAC-Train-Annotations-XML" / "MVI_00001.xml"
    (detrac_root / "DETRAC-Test-Annotations-XML" / "MVI_00001.xml").write_text(src.read_text())
    with pytest.raises(DetracError, match="both Train and Test"):
        discover(detrac_root)


def test_discover_rejects_unassigned_xml(detrac_root) -> None:
    (detrac_root / "stray.xml").write_text("<sequence/>")
    with pytest.raises(DetracError, match="not under a Train or Test folder"):
        discover(detrac_root)
