"""Atomic file writes: an interrupted run never leaves a partial file behind."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temp path next to `path`; move it into place only if the block succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    with atomic_path(dst) as tmp:
        shutil.copyfile(src, tmp)
