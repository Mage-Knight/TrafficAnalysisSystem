import json
import sys
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from traffic_analytics.cli import app
from traffic_analytics.config import load_config
from traffic_analytics.data import export
from traffic_analytics.data.detrac import DetracError
from traffic_analytics.detection import yolo
from traffic_analytics.detection.yolo import (
    REQUIRED_KEYS,
    check_config,
    package_version,
    runtime_data_yaml,
    start_run,
    train_kwargs,
    write_runtime_data_yaml,
)

REPO = Path(__file__).parent.parent


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    """Prepared dir whose data.yaml still points where `ta data prepare` ran."""
    prep = tmp_path / "prep"
    (prep / "yolo").mkdir(parents=True)
    data = export.yolo_data_yaml(Path("/elsewhere/detrac_prepared/yolo"))
    (prep / "yolo" / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return prep


def test_runtime_data_yaml(prepared, tmp_path) -> None:
    original = prepared / "yolo" / "data.yaml"
    before = original.read_bytes()
    out = write_runtime_data_yaml(prepared, tmp_path / "run" / "data.yaml")

    data = yaml.safe_load(out.read_text())
    expected = export.yolo_data_yaml(prepared / "yolo")
    assert data == expected
    assert data["path"] == str((prepared / "yolo").resolve())
    assert data["names"] == {0: "car", 1: "bus", 2: "van", 3: "others"}
    assert original.read_bytes() == before


def test_runtime_data_yaml_missing(tmp_path) -> None:
    with pytest.raises(DetracError, match="ta data prepare"):
        runtime_data_yaml(tmp_path)


def _write_cfg(tmp_path: Path, **values) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_load_config_expands_root(tmp_path, monkeypatch) -> None:
    path = _write_cfg(tmp_path, project="${TA_ROOT}/runs/yolo", epochs=3)
    monkeypatch.delenv("TA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_config(path)["project"] == f"{tmp_path}/runs/yolo"
    monkeypatch.setenv("TA_ROOT", "/mnt/drive")
    assert load_config(path) == {"project": "/mnt/drive/runs/yolo", "epochs": 3}


def test_load_config_rejects_unknown_variable(tmp_path) -> None:
    with pytest.raises(DetracError, match=r"\$\{HOME\}"):
        load_config(_write_cfg(tmp_path, project="${HOME}/runs"))


def test_load_config_missing_keys(tmp_path) -> None:
    with pytest.raises(DetracError, match="missing keys: name"):
        load_config(_write_cfg(tmp_path, project="x"), ("project", "name"))


def test_committed_yolo_config(monkeypatch) -> None:
    monkeypatch.delenv("TA_ROOT", raising=False)
    monkeypatch.chdir(REPO)
    cfg = load_config(REPO / "configs/train/yolo_s.yaml", REQUIRED_KEYS)
    assert cfg["project"] == f"{REPO}/runs/yolo"
    assert cfg["prepared"] == f"{REPO}/data/prepared"
    assert cfg["imgsz"] == 640
    assert cfg["device"] is None
    assert cfg["wandb"] is True
    check_config(cfg)


def test_check_config_wandb_project() -> None:
    check_config({"wandb": False})
    with pytest.raises(DetracError, match="wandb_project"):
        check_config({"wandb": True})
    with pytest.raises(DetracError, match="wandb_project"):
        check_config({})


def test_check_config_train_args() -> None:
    check_config({"wandb": False, "train_args": None})
    check_config({"wandb": False, "train_args": {"mosaic": 0.5}})
    with pytest.raises(DetracError, match="must be a mapping"):
        check_config({"wandb": False, "train_args": ["mosaic"]})
    with pytest.raises(DetracError, match="cannot set: epochs"):
        check_config({"wandb": False, "train_args": {"epochs": 3, "mosaic": 0.5}})


TRAIN_CFG = {"imgsz": 640, "epochs": 3, "batch": 16, "seed": 0, "workers": 8}


def test_train_kwargs_without_train_args() -> None:
    out = Path("/runs/yolo/yolo_s")
    assert train_kwargs(TRAIN_CFG, "/run/data.yaml", out) == {
        **TRAIN_CFG,
        "data": "/run/data.yaml",
        "project": "/runs/yolo",
        "name": "yolo_s",
        "exist_ok": True,
        "save_period": -1,
    }


def test_train_kwargs_merges_train_args() -> None:
    out = Path("/runs/yolo/yolo_s")
    base = train_kwargs(TRAIN_CFG, "/run/data.yaml", out)
    cfg = {**TRAIN_CFG, "train_args": {"mosaic": 0.5, "hsv_v": 0.6}}
    assert train_kwargs(cfg, "/run/data.yaml", out) == {**base, "mosaic": 0.5, "hsv_v": 0.6}


def test_train_rejects_reserved_train_arg(prepared, tmp_path) -> None:
    cfg = {k: 1 for k in REQUIRED_KEYS} | {"train_args": {"epochs": 3}}
    path = _write_cfg(tmp_path, **cfg)
    args = ["train", "yolo", "--config", str(path), "--prepared", str(prepared), "--no-wandb"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "error: `train_args` cannot set: epochs" in result.output


def _run_cfg(tmp_path: Path) -> dict:
    return {"project": str(tmp_path / "runs"), "name": "yolo_s"}


def test_start_run_records_prepare_config(prepared, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(yolo, "package_version", lambda name: "9.9.9")
    settings = {"settings": {"strides": {"train": 10}}}
    (prepared / "prepare_config.json").write_text(json.dumps(settings))
    logged: list[str] = []
    meta = start_run(_run_cfg(tmp_path), prepared, Path("cfg.yaml"), logged.append)
    assert meta["prepare_config"] == settings
    assert meta["albumentations"] == "9.9.9"
    assert not [line for line in logged if "prepare_config" in line]


def test_start_run_without_prepare_config(prepared, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(yolo, "package_version", lambda name: None)
    logged: list[str] = []
    meta = start_run(_run_cfg(tmp_path), prepared, Path("cfg.yaml"), logged.append)
    assert meta["prepare_config"] is None
    assert meta["albumentations"] is None
    assert [line for line in logged if "prepare_config.json" in line]


def test_package_version() -> None:
    assert package_version("pyyaml")
    assert package_version("no-such-package-xyz") is None


def test_train_yolo_options() -> None:
    # Inspect the Click command, not rendered help: Rich output differs under GITHUB_ACTIONS.
    command = typer.main.get_command(app).commands["train"].commands["yolo"]
    flags = {flag for p in command.params for flag in (*p.opts, *p.secondary_opts)}
    assert {"--config", "--prepared", "--resume", "--device", "--wandb", "--no-wandb"} <= flags


def test_train_refuses_existing_run_dir(prepared, tmp_path) -> None:
    project = tmp_path / "runs"
    (project / "yolo_s").mkdir(parents=True)
    cfg = {k: 1 for k in REQUIRED_KEYS} | {
        "prepared": str(prepared),
        "project": str(project),
        "name": "yolo_s",
    }
    path = _write_cfg(tmp_path, **cfg)
    result = CliRunner().invoke(app, ["train", "yolo", "--config", str(path), "--no-wandb"])
    assert result.exit_code == 1
    assert "--resume" in result.output
    assert "ultralytics" not in sys.modules


def test_train_resume_without_checkpoint(prepared, tmp_path) -> None:
    cfg = {k: 1 for k in REQUIRED_KEYS} | {"project": str(tmp_path / "runs"), "name": "yolo_s"}
    path = _write_cfg(tmp_path, **cfg)
    args = ["train", "yolo", "--config", str(path), "--prepared", str(prepared), "--resume"]
    args.append("--no-wandb")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "nothing to resume" in result.output
