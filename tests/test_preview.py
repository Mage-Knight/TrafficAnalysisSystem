from pathlib import Path

import pytest
from conftest import IMAGE_SIZE
from PIL import Image

from traffic_analytics.data.detrac import Box, DetracError
from traffic_analytics.data.preview import (
    PreviewItem,
    load_coco,
    load_items,
    load_yolo,
    render_grid,
    sample,
)


def _by_stem(items: list[PreviewItem]) -> dict[str, PreviewItem]:
    return {i.image.stem: i for i in items}


def test_box_from_yolo_inverts_to_yolo() -> None:
    box = Box(3.5, 7, 20, 11.25)
    back = Box.from_yolo(*box.to_yolo(64, 48), 64, 48)
    assert back.as_list(6) == box.as_list(6)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_coco_and_yolo_agree(prepared: Path, split: str) -> None:
    coco, yolo = _by_stem(load_coco(prepared, split)), _by_stem(load_yolo(prepared, split))
    assert coco.keys() == yolo.keys() and coco
    for stem, c in coco.items():
        y = yolo[stem]
        assert c.image.is_file() and y.image.is_file()
        assert [n for n, _ in c.boxes] == [n for n, _ in y.boxes]
        for (_, a), (_, b) in zip(c.boxes, y.boxes, strict=True):
            assert a.as_list() == pytest.approx(b.as_list(), abs=0.01)
        if split != "train":
            assert len(c.ignore) == len(y.ignore)
            for a, b in zip(c.ignore, y.ignore, strict=True):
                assert a.as_list() == pytest.approx(b.as_list(), abs=0.01)


def test_ignore_sources(prepared: Path) -> None:
    val = load_coco(prepared, "val")
    assert all(i.ignore for i in val)
    assert all(len(i.boxes) <= 2 for i in val)  # crowd anns (4 per region) not drawn
    assert all(i.ignore for i in load_coco(prepared, "train"))
    assert not any(i.ignore for i in load_yolo(prepared, "train"))  # no yolo/ignore/train


def test_load_items_errors(prepared: Path, tmp_path: Path) -> None:
    with pytest.raises(DetracError, match="unknown split"):
        load_items(prepared, "dev", "coco")
    with pytest.raises(DetracError, match="unknown format"):
        load_items(prepared, "val", "voc")
    with pytest.raises(DetracError, match="ta data prepare"):
        load_items(tmp_path / "empty", "val", "yolo")


def _items(counts: dict[str, int]) -> list[PreviewItem]:
    return [
        PreviewItem(Path(f"{seq}_img{num:05d}.jpg"), [], [])
        for seq, n in counts.items()
        for num in range(1, n + 1)
    ]


def test_sample_spreads_over_sequences() -> None:
    items = _items({"MVI_1": 10, "MVI_2": 10, "MVI_3": 1})
    picked = sample(items, 3, seed=0)
    assert {i.image.stem.split("_img")[0] for i in picked} == {"MVI_1", "MVI_2", "MVI_3"}
    assert sample(items, 3, seed=0) == picked
    assert sample(items, 3, seed=1) != picked or len(items) == 3
    assert len(sample(items, 12, seed=0)) == 12
    assert sorted(i.image.name for i in sample(items, 100, seed=0)) == sorted(
        i.image.name for i in items
    )


def test_render_grid_size(prepared: Path, tmp_path: Path) -> None:
    items = load_coco(prepared, "val")
    out = tmp_path / "grid.jpg"
    render_grid(items, out, cols=2, tile_width=128)
    tile_h = round(IMAGE_SIZE[1] * 128 / IMAGE_SIZE[0])
    with Image.open(out) as im:
        assert im.size == (2 * 128, 2 * (tile_h + 14))
