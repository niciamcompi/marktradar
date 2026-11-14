"""The self-update cycle: detect new data, measure live accuracy, retrain, record."""

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from marktradar import data, monitoring
from marktradar.cli import app as cli
from marktradar.config import Settings

from .conftest import LAST_PREMIUM_YEAR, config_file

runner = CliRunner()


def test_update_cycle(settings: Settings, remote: Callable[[int], None], tmp_path: Path) -> None:
    config = config_file(settings)

    # 1) first run: the 'BAG' publishes data up to premium year LAST-1
    remote(LAST_PREMIUM_YEAR - 1)
    first = runner.invoke(cli, ["update", "--config", config])
    assert first.exit_code == 0, first.output
    manifest = data.Manifest.load(settings.data.manifest_path)
    assert f"archiv_{LAST_PREMIUM_YEAR - 2}.zip" in manifest.files

    # 2) nothing new -> nothing happens
    again = runner.invoke(cli, ["update", "--config", config])
    assert "No new data" in again.output

    # 3) new release: one more year of premiums and insured data
    remote(LAST_PREMIUM_YEAR)
    report = tmp_path / "report.md"
    third = runner.invoke(cli, ["update", "--config", config, "--report", str(report)])
    assert third.exit_code in (0, 2), third.output
    live = monitoring.load(settings.registry.path)
    # the old model is measured on the base year that just became evaluable, per horizon
    assert {(r["horizont"], r["prognosejahr"]) for r in live} == {
        (1, LAST_PREMIUM_YEAR - 2),
        (2, LAST_PREMIUM_YEAR - 2),
    }
    assert "Live accuracy" in report.read_text()
    if third.exit_code == 0:
        assert (
            f"archiv_{LAST_PREMIUM_YEAR - 1}.zip"
            in data.Manifest.load(settings.data.manifest_path).files
        )


def test_monitoring_append_is_idempotent(tmp_path: Path) -> None:
    rec = {"modell_version": "v1", "horizont": 1, "basisjahr": 2014, "mae_modell": 0.1}
    monitoring.append(tmp_path, [rec])
    monitoring.append(tmp_path, [rec, {**rec, "basisjahr": 2015}])
    assert [r["basisjahr"] for r in monitoring.load(tmp_path)] == [2014, 2015]
