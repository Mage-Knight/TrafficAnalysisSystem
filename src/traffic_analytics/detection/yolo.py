"""`ta train yolo`: Ultralytics YOLO baseline on the prepared YOLO export (optional W&B)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml

from traffic_analytics.data.detrac import DetracError
from traffic_analytics.data.files import write_text

REQUIRED_KEYS = (
    "model",
    "prepared",
    "project",
    "name",
    "imgsz",
    "epochs",
    "batch",
    "seed",
    "workers",
)
META_FILE = "run_meta.json"
CPU_WARNING = """
WARNING: no CUDA GPU found, training on CPU (very slow).
  torch is probably a CPU build (uv installs one); for GPU training install with pip,
  see README "GPU training". Pass --device cpu to train on CPU on purpose.
"""


def check_config(cfg: dict) -> None:
    """`wandb_project` is required only when W&B logging is on."""
    if cfg.get("wandb", True) and not cfg.get("wandb_project"):
        raise DetracError("config needs `wandb_project` when `wandb` is true (or pass --no-wandb)")


def runtime_data_yaml(prepared: Path) -> dict:
    """`<prepared>/yolo/data.yaml` with `path` pointing at this runtime's copy of the export."""
    src = prepared / "yolo" / "data.yaml"
    if not src.is_file():
        raise DetracError(f"no YOLO data.yaml in {prepared} (run `ta data prepare` and unzip it)")
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    return {**data, "path": str((prepared / "yolo").resolve())}


def write_runtime_data_yaml(prepared: Path, out: Path) -> Path:
    write_text(out, yaml.safe_dump(runtime_data_yaml(prepared), sort_keys=False))
    return out


def git_info(repo: Path | None = None) -> dict:
    """Commit hash and dirty flag of the checkout the code runs from; None outside a repo."""
    cwd = repo or Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def run_dir(cfg: dict) -> Path:
    return Path(cfg["project"]) / cfg["name"]


def start_run(cfg: dict, prepared: Path, config_path: Path, log: Callable[[str], None]) -> dict:
    """Create the run dir with its data.yaml and run_meta.json; refuse to reuse an existing one."""
    out = run_dir(cfg)
    if out.exists():
        raise DetracError(f"run dir already exists: {out} (use --resume, or change `name`)")
    git = git_info()
    if git["dirty"]:
        log("warning: git tree is dirty; the recorded commit does not match the running code")
    data_yaml = write_runtime_data_yaml(prepared, out / "data.yaml")
    meta = {
        "git": git,
        "config_file": str(config_path),
        "config": cfg,
        "prepared": str(prepared),
        "data_yaml": str(data_yaml),
        "wandb_id": None,
    }
    save_meta(out, meta)
    return meta


def load_meta(out: Path) -> dict:
    path = out / META_FILE
    if not path.is_file():
        raise DetracError(f"no {META_FILE} in {out}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_meta(out: Path, meta: dict) -> None:
    write_text(out / META_FILE, json.dumps(meta, indent=2) + "\n")


def train(
    cfg: dict,
    config_path: Path,
    prepared: Path,
    resume: bool,
    log: Callable[[str], None] = print,
) -> Path:
    """Train (or resume) and return the run dir; only last.pt and best.pt are kept."""
    # Background (&) jobs inherit SIGINT as ignored; restore it so `colab_train.sh stop` works.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    out = run_dir(cfg).resolve()
    log(f"TA_ROOT={os.environ.get('TA_ROOT') or 'unset -> cwd'}; run dir: {out}")
    last = out / "weights" / "last.pt"
    if resume:
        if not last.is_file():
            raise DetracError(f"nothing to resume: {last} not found")
        meta = load_meta(out)
        # The checkpoint points at this file; refresh it in case the export moved.
        write_runtime_data_yaml(prepared, Path(meta["data_yaml"]))
    else:
        meta = start_run(cfg, prepared, config_path, log)

    use_wandb = cfg.get("wandb", True)
    device = cfg.get("device")
    if use_wandb:
        import wandb

        if not wandb.login(prompt=False):
            raise DetracError(
                "no W&B API key: set WANDB_API_KEY or run `wandb login`, or pass --no-wandb"
            )

    import torch
    from ultralytics import YOLO, settings

    if device is None and not torch.cuda.is_available():
        log(CPU_WARNING)
    settings.update({"wandb": use_wandb})  # persists on disk, so set it both ways
    if use_wandb:
        # Start the W&B run ourselves: Ultralytics' callback reuses it instead of naming the W&B
        # project after the path in `project`. The stored id keeps resumes in the same run.
        run = wandb.init(
            project=cfg["wandb_project"],
            name=cfg["name"],
            id=meta["wandb_id"],
            resume="allow",
            config={"ta": meta},
        )
        if meta["wandb_id"] != run.id:
            meta["wandb_id"] = run.id
            save_meta(out, meta)
        log(f"W&B: {run.url}")
    log(f"commit: {meta['git']['commit']}")

    overrides = {} if device is None else {"device": device}
    try:
        if resume:
            YOLO(str(last)).train(resume=True, **overrides)
        else:
            YOLO(cfg["model"]).train(
                data=meta["data_yaml"],
                project=str(out.parent),
                name=out.name,
                exist_ok=True,  # start_run created the dir; never let Ultralytics pick name2
                save_period=-1,  # only last.pt and best.pt (disk space)
                imgsz=cfg["imgsz"],
                epochs=cfg["epochs"],
                batch=cfg["batch"],
                seed=cfg["seed"],
                workers=cfg["workers"],
                **overrides,
            )
    finally:
        if use_wandb:
            wandb.finish()
    return out
