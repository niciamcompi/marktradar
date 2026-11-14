"""Selbstständige Aktualisierung: Datenerkennung, Live-Monitoring, Config-Nachführung."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from marktradar import data, monitoring, registry
from marktradar.api import create_app
from marktradar.cli import app as cli
from marktradar.config import ModelConfig, Settings, load_settings, update_data_source
from marktradar.data import find_latest_data_url, parse_workbook
from marktradar.features import build_features, training_frame
from marktradar.model import fit

runner = CliRunner()


def test_find_latest_data_url_picks_newest_year() -> None:
    html = (
        '<a href="/dam/de/sd-web/AAA/aufsichtsdaten-okp-1996-2024.xlsx">alt</a>'
        '<a href="https://www.bag.admin.ch/dam/de/sd-web/BBB/aufsichtsdaten-okp-1996-2025.xlsx">neu</a>'
        '<a href="/dam/de/sd-web/CCC/bericht.pdf">pdf</a>'
    )
    assert find_latest_data_url(html) == (
        "https://www.bag.admin.ch/dam/de/sd-web/BBB/aufsichtsdaten-okp-1996-2025.xlsx"
    )
    assert find_latest_data_url("<html>nichts</html>") is None


def test_update_data_source_keeps_comments(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        '# Kommentar\ndata:\n  # Quelle\n  source_url: "https://alt"\n  sha256: "abc"\n'
        '  raw_path: "x"\n  processed_path: "y"\n'
    )
    update_data_source(path, "https://neu", "def")
    text = path.read_text()
    assert '"https://neu"' in text and '"def"' in text and "# Quelle" in text
    cfg = load_settings(path)
    assert cfg.data.source_url == "https://neu" and cfg.data.sha256 == "def"


def test_live_evaluation_only_uses_unseen_years(workbook: Path) -> None:
    frame = training_frame(build_features(parse_workbook(workbook)), 100, 0.5)
    model = fit(frame[frame["jahr"] <= 2011], ModelConfig(max_iter=30))
    meta = {"version": "v1", "trainiert_auf_basisjahre": [2006, 2011]}
    records = monitoring.live_evaluation(model, meta, frame)
    assert [r["basisjahr"] for r in records] == sorted(frame["jahr"][frame["jahr"] > 2011].unique())
    assert all(r["prognosejahr"] == r["basisjahr"] + 1 for r in records)


def test_monitoring_append_is_idempotent(tmp_path: Path) -> None:
    rec = {"modell_version": "v1", "basisjahr": 2014, "mae_modell": 0.1}
    monitoring.append(tmp_path, [rec])
    monitoring.append(tmp_path, [rec, {**rec, "basisjahr": 2015}])
    assert [r["basisjahr"] for r in monitoring.load(tmp_path)] == [2014, 2015]


def _config_file(settings: Settings) -> str:
    return str(settings.registry.path.parent / "config" / "config.yaml")


@pytest.fixture
def fake_bag(monkeypatch: pytest.MonkeyPatch, workbook: Path) -> dict[str, Path]:
    """Ersetzt das Netzwerk: "download" kopiert die gerade aktuelle synthetische Datei."""
    current = {"file": workbook}

    def fake_download(url: str, dest: Path, expected_sha256: str | None = None) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if current["file"] != dest:
            shutil.copy(current["file"], dest)
        return data.sha256_of(dest)

    monkeypatch.setattr(data, "download", fake_download)
    monkeypatch.setattr(data, "discover_latest_url", lambda page: "https://bag.example/neu.xlsx")
    return current


def test_update_cycle(settings: Settings, fake_bag: dict[str, Path], tmp_path: Path) -> None:
    config = _config_file(settings)
    cfg_dict = yaml.safe_load(Path(config).read_text())
    cfg_dict["data"].update(
        discovery_page="https://bag.example", sha256="", raw_path=str(tmp_path / "raw.xlsx")
    )
    Path(config).write_text(yaml.safe_dump(cfg_dict))

    # 1) Altes Modell: nur Daten bis 2013 bekannt
    old = tmp_path / "alt.xlsx"
    with pd.ExcelWriter(old) as w:
        for y in range(2006, 2014):
            pd.read_excel(fake_bag["file"], sheet_name=str(y), header=None).to_excel(
                w, sheet_name=str(y), header=False, index=False
            )
    fake_bag["file"] = old
    first = runner.invoke(cli, ["update", "--config", config])
    assert first.exit_code == 0, first.output
    cfg = load_settings(config)
    assert cfg.data.source_url == "https://bag.example/neu.xlsx" and cfg.data.sha256

    # 2) Gleiche Daten -> nichts zu tun
    again = runner.invoke(cli, ["update", "--config", config])
    assert "Keine neuen Daten" in again.output

    # 3) BAG publiziert neue Jahre -> Live-Güte des alten Modells, dann Neutraining
    fake_bag["file"] = settings.data.raw_path
    report = tmp_path / "report.md"
    third = runner.invoke(cli, ["update", "--config", config, "--report", str(report)])
    assert third.exit_code in (0, 2), third.output
    live = monitoring.load(cfg.registry.path)
    assert live and min(r["basisjahr"] for r in live) == 2013  # alt: Basisjahre bis 2012
    assert "Live-Güte" in report.read_text()


def test_web_ui_and_monitoring_in_api(settings: Settings) -> None:
    config = _config_file(settings)
    for cmd in (["build"], ["train"]):
        assert runner.invoke(cli, [*cmd, "--config", config]).exit_code == 0
    monitoring.append(settings.registry.path, [{"modell_version": "x", "basisjahr": 2014}])
    with TestClient(create_app(settings)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Marktradar" in page.text
        assert client.get("/static/app.js").status_code == 200
        meta = client.get("/modell").json()
        assert meta["live_monitoring"][0]["basisjahr"] == 2014
        assert registry.production_version(settings.registry.path) == meta["version"]
        assert json.dumps(meta)
