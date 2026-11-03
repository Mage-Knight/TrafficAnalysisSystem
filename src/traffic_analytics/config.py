"""YAML run configs with `${TA_ROOT}` expanded to the storage root (default: the cwd)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from traffic_analytics.data.detrac import DetracError

ROOT_VAR = "${TA_ROOT}"
_UNEXPANDED = re.compile(r"\$\{[^}]*\}")


def storage_root() -> str:
    """Where data/ and runs/ live: $TA_ROOT, else the current directory (the repo root)."""
    return os.environ.get("TA_ROOT") or str(Path.cwd())


def _expand(value: object, root: str) -> object:
    if isinstance(value, str):
        value = value.replace(ROOT_VAR, root)
        if m := _UNEXPANDED.search(value):
            raise DetracError(f"unknown variable {m.group()} (only {ROOT_VAR} is supported)")
        return value
    if isinstance(value, dict):
        return {k: _expand(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, root) for v in value]
    return value


def load_config(path: Path, required: tuple[str, ...] = ()) -> dict:
    """Load a YAML mapping, expand `${TA_ROOT}` and check that `required` keys exist."""
    if not path.is_file():
        raise DetracError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DetracError(f"config must be a mapping: {path}")
    cfg = _expand(raw, storage_root())
    if missing := [k for k in required if k not in cfg]:
        raise DetracError(f"config {path} is missing keys: {', '.join(missing)}")
    return cfg
