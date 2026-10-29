from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURE_XML = Path(__file__).parent / "fixtures" / "detrac_sequence.xml"
IMAGE_SIZE = (64, 48)
# sequence -> annotation folder, mirroring the Kaggle layout
SEQUENCES = {
    "MVI_00001": "DETRAC-Train-Annotations-XML",
    "MVI_00003": "DETRAC-Train-Annotations-XML",
    "MVI_00002": "DETRAC-Test-Annotations-XML",
}
SPLITS = {"train": ["MVI_00001"], "val": ["MVI_00003"], "test": ["MVI_00002"]}


@pytest.fixture
def detrac_root(tmp_path: Path) -> Path:
    """Tiny fake download: the fixture XML copied under 3 names + synthetic 64x48 JPEGs."""
    root = tmp_path / "detrac"
    xml_text = FIXTURE_XML.read_text()
    rng = np.random.default_rng(0)
    for name, folder in SEQUENCES.items():
        xml_dir = root / folder
        xml_dir.mkdir(parents=True, exist_ok=True)
        (xml_dir / f"{name}.xml").write_text(xml_text.replace("MVI_00001", name))
        img_dir = root / "Insight-MVT_Annotation" / name
        img_dir.mkdir(parents=True)
        for num in (1, 2, 3):
            pixels = rng.integers(0, 256, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
            Image.fromarray(pixels).save(img_dir / f"img{num:05d}.jpg", quality=90)
    return root


@pytest.fixture
def splits_file(tmp_path: Path) -> Path:
    from traffic_analytics.data.splits import save_splits

    path = tmp_path / "splits.yaml"
    save_splits(path, SPLITS)
    return path
