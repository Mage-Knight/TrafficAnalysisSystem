from typer.testing import CliRunner

from traffic_analytics.cli import app


def test_env_command_runs() -> None:
    result = CliRunner().invoke(app, ["env"])
    assert result.exit_code == 0, result.output
    assert "Python" in result.output


def test_data_prepare(detrac_root, splits_file, scenes_file, tmp_path) -> None:
    out = tmp_path / "out"
    args = ["data", "prepare", "--root", str(detrac_root), "--out", str(out)]
    result = CliRunner().invoke(
        app, [*args, "--splits", str(splits_file), "--scenes", str(scenes_file), "--stride", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "annotation folder:" in result.output
    assert (out / "coco/valid/_annotations.coco.json").is_file()


def test_data_prepare_without_splits_fails(detrac_root, tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "data",
            "prepare",
            "--root",
            str(detrac_root),
            "--out",
            str(tmp_path / "out"),
            "--splits",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "ta data split" in result.output
