"""Command-line interface (`ta`)."""

import platform
import sys

import typer

from traffic_analytics import __version__

app = typer.Typer(help="Traffic analytics on UA-DETRAC.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Traffic analytics CLI."""


@app.command()
def env() -> None:
    """Print Python, torch, CUDA and GPU info."""
    typer.echo(f"traffic-analytics: {__version__}")
    typer.echo(f"Python: {platform.python_version()} ({sys.executable})")
    typer.echo(f"Platform: {platform.platform()}")

    try:
        import torch
    except ImportError:
        typer.echo("torch: not installed (pip install -e '.[ml]')")
        return

    typer.echo(f"torch: {torch.__version__}")
    typer.echo(f"CUDA (build): {torch.version.cuda or 'none'}")
    cuda = torch.cuda.is_available()
    typer.echo(f"CUDA available: {cuda}")
    if not cuda:
        return
    typer.echo(f"cuDNN: {torch.backends.cudnn.version()}")
    count = torch.cuda.device_count()
    typer.echo(f"GPUs: {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        typer.echo(f"  [{i}] {props.name}, {mem_gb:.1f} GB, compute {props.major}.{props.minor}")
