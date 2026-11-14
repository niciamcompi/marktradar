"""End to end: CLI pipeline (download -> build -> train) and the API on top of it."""

import json
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from marktradar import registry
from marktradar.api import create_app
from marktradar.cli import app as cli
from marktradar.config import Settings

from .conftest import LAST_PREMIUM_YEAR, config_file, install_remote, make_settings

runner = CliRunner()


def _train(settings: Settings) -> Settings:
    for cmd in ("download", "build", "train"):
        result = runner.invoke(cli, [cmd, "--config", config_file(settings)])
        assert result.exit_code == 0, result.output
    return settings


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    """Trained once for the whole module (training is the slow part)."""
    base = tmp_path_factory.mktemp("api")
    with pytest.MonkeyPatch.context() as mp:
        install_remote(base, mp)
        yield _train(make_settings(base))


@pytest.fixture(scope="module")
def client(trained: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(trained)) as c:
        yield c


def test_quality_gate_blocks_worse_model(settings: Settings, remote: Callable[[int], None]) -> None:
    trained = _train(settings)
    root = trained.registry.path
    prod = registry.production_version(root)
    assert prod is not None
    meta_path = root / prod / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["horizonte"]["2"]["backtest"]["mae_modell"] = 1e-9  # pretend production is perfect
    meta_path.write_text(json.dumps(meta))
    result = runner.invoke(cli, ["train", "--config", config_file(trained)])
    assert result.exit_code == 2, result.output
    assert registry.production_version(root) == prod


def test_cli_predict(trained: Settings) -> None:
    result = runner.invoke(cli, ["predict", "8", "--config", config_file(trained)])
    assert result.exit_code == 0, result.output
    assert f'"prognosejahr": {LAST_PREMIUM_YEAR}' in result.output


def test_health_model_and_ui(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/modell").json()
    assert set(meta["horizonte"]) == {"1", "2"} and meta["horizonte_verfuegbar"] == [1, 2]
    assert meta["live_monitoring"] == []
    page = client.get("/")
    assert page.status_code == 200 and "Marktradar" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_lists(client: TestClient) -> None:
    insurers = client.get("/versicherer").json()
    assert len(insurers) == 14 and insurers[0]["versicherte"] >= insurers[-1]["versicherte"]
    assert {k["kanton"] for k in client.get("/kantone").json()} == {
        "ZH",
        "BE",
        "VD",
        "AG",
        "GE",
        "TI",
    }


def test_prognose_with_cantons(client: TestClient) -> None:
    body = client.post("/prognose", json={"bag_nr": 8}).json()
    assert body["prognosejahr"] == LAST_PREMIUM_YEAR and body["horizont"] == 2
    assert len(body["kantone"]) == 6
    assert (
        body["anteilswachstum_untergrenze_pct"]
        <= body["anteilswachstum_pct"]
        <= body["anteilswachstum_obergrenze_pct"]
    )
    one_year = client.post("/prognose", json={"bag_nr": 8, "horizont": 1}).json()
    assert one_year["prognosejahr"] == LAST_PREMIUM_YEAR - 1


def test_elasticity_is_monotone(client: TestClient) -> None:
    curve = client.post("/elastizitaet", json={"bag_nr": 8}).json()
    growth = [p["anteilswachstum_pct"] for p in curve]
    assert len(growth) == 11 and growth == sorted(growth, reverse=True)


def test_market_national_and_canton(client: TestClient) -> None:
    rows = client.get("/markt").json()
    growth = [r["anteilswachstum_pct"] for r in rows]
    assert growth == sorted(growth, reverse=True)
    zh = client.get("/markt", params={"kanton": "zh"}).json()
    assert sum(r["marktanteil_basis_pct"] for r in zh) == pytest.approx(100, abs=0.1)


def test_errors(client: TestClient) -> None:
    assert client.post("/prognose", json={"bag_nr": 999999}).status_code == 404
    assert (
        client.post("/prognose", json={"bag_nr": 8, "praemienanpassung_rel_pp": 50}).status_code
        == 422
    )
    assert client.post("/prognose", json={"bag_nr": 8, "horizont": 5}).status_code == 422
    bad_range = {"bag_nr": 8, "von_pp": 3, "bis_pp": 1}
    assert client.post("/elastizitaet", json=bad_range).status_code == 422
