from typer.testing import CliRunner

from traffic_analytics.cli import app


def test_env_command_runs() -> None:
    result = CliRunner().invoke(app, ["env"])
    assert result.exit_code == 0, result.output
    assert "Python" in result.output
