"""Shared fixtures: a simulated BAG open-data server.

The synthetic files mimic the real ones closely: yearly zip archives, the current-year
CSVs, the old (2011-2014) and new formats, UTF-8 and cp1252 encodings, comma and
semicolon separators, a "CH" total row in old insured files, and a names workbook.

A known effect is built in: an insurer that raises its premium more than its canton
market loses share there (elasticity -2). The tests check that the pipeline finds it.
"""

from __future__ import annotations

import base64
import io
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from marktradar import data
from marktradar.config import Settings, load_settings

CANTONS = ["ZH", "BE", "VD", "AG", "GE", "TI"]
N_INSURERS = 14
FIRST_PREMIUM_YEAR, LAST_PREMIUM_YEAR = 2011, 2021  # insured data: 2009..2019


def _simulate(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    nrs = [8, 32, 62, 134, 290, 312, 343, 376, 455, 1384, 1509, 1535, 1542, 1562][:N_INSURERS]
    size = {(i, k): rng.lognormal(11.3, 0.9) for i in nrs for k in CANTONS}
    rel = {(i, k): rng.normal(0, 0.06) for i in nrs for k in CANTONS}
    base = dict(zip(CANTONS, rng.uniform(300, 450, len(CANTONS)), strict=True))
    premiums, insured = [], []
    for year in range(2009, LAST_PREMIUM_YEAR + 1):
        market = {k: base[k] * 1.04 ** (year - 2009) for k in CANTONS}
        for i in nrs:
            for k in CANTONS:
                premiums.append((i, k, year, round(market[k] * np.exp(rel[i, k]), 1)))
                insured.append((i, k, year, size[i, k]))
        # next year's premium shock; insured react to it within the same year
        for i in nrs:
            for k in CANTONS:
                s = rng.normal(0, 0.03)
                rel[i, k] += s
                size[i, k] *= np.exp(-2.0 * s + rng.normal(0, 0.004))
    p = pd.DataFrame(premiums, columns=["bag_nr", "kanton", "jahr", "praemie"])
    b = pd.DataFrame(insured, columns=["bag_nr", "kanton", "jahr", "versicherte"])
    return p, b


PREMIUMS, INSURED = _simulate()


def _premium_csv(year: int, old_format: bool, cp1252: bool) -> bytes:
    p = PREMIUMS[PREMIUMS["jahr"] == year]
    rows = []
    for r in p.itertuples():
        # reference premium plus a non-reference row (other model/franchise) that must be ignored
        for is_base, factor in ((1, 1.0), (0, 0.85)):
            if old_format:
                rows.append(
                    [
                        f"{r.bag_nr:04d}",
                        r.kanton,
                        "CH",
                        year - 1,
                        year,
                        0,
                        26,
                        "05",
                        "BASE",
                        "Base",
                        1,
                        "E1",
                        "E",
                        is_base,
                        is_base,
                        1,
                        300,
                        round(r.praemie * factor, 1),
                        1,
                        "Grundversicherung",
                        1,
                        1,
                    ]
                )
            else:
                rows.append(
                    [
                        r.bag_nr,
                        r.kanton,
                        "CH",
                        year,
                        year - 1,
                        "PR-REG CH0",
                        "AKL-ERW",
                        "MIT-UNF",
                        "BASE",
                        "TAR-BASE",
                        "E1",
                        "FRAST1",
                        "FRA-300",
                        round(r.praemie * factor, 1),
                        is_base,
                        1,
                        "Base",
                    ]
                )
            # a child premium that must be ignored as well
        if not old_format:
            rows.append(
                [
                    r.bag_nr,
                    r.kanton,
                    "CH",
                    year,
                    year - 1,
                    "PR-REG CH0",
                    "AKL-KIN",
                    "MIT-UNF",
                    "BASE",
                    "TAR-BASE",
                    "K1",
                    "FRAST1",
                    "FRA-0",
                    99.0,
                    1,
                    1,
                    "Base",
                ]
            )
    if old_format:
        cols = [
            "G_ID",
            "C_ID",
            "C_GRP",
            "EJAHR",
            "JAHR",
            "R_ID",
            "M_ID",
            "VAR_ID",
            "V_ID",
            "V_TYP",
            "isBASE_V2",
            "V2_ID",
            "V2_TYP",
            "isBASE_P",
            "isBASE_F",
            "F_STUFE",
            "F",
            "P",
            "V_SORT_NR",
            "V_KBEZ",
            "isBASE_V",
            "IstTaetig",
        ]
    else:
        cols = [
            "Versicherer",
            "Kanton",
            "Hoheitsgebiet",
            "Geschäftsjahr",
            "Erhebungsjahr",
            "Region",
            "Altersklasse",
            "Unfalleinschluss",
            "Tarif",
            "Tariftyp",
            "Altersuntergruppe",
            "Franchisestufe",
            "Franchise",
            "Prämie",
            "isBaseP",
            "isBaseF",
            "Tarifbezeichnung",
        ]
    sep = ";" if (old_format or cp1252) else ","
    text = pd.DataFrame(rows, columns=cols).to_csv(sep=sep, index=False)
    return text.encode("cp1252") if cp1252 else text.encode("utf-8-sig")


def _insured_csv(year: int, old_format: bool) -> bytes:
    b = INSURED[INSURED["jahr"] == year]
    if old_format:
        df = pd.DataFrame(
            {
                "C_ID": b["kanton"],
                "G_ID": b["bag_nr"].map("{:04d}".format),
                "JAHR": year,
                "B": b["versicherte"],
            }
        )
        total = df.groupby("G_ID", as_index=False)["B"].sum().assign(C_ID="CH", JAHR=year)
        df = pd.concat([df, total[df.columns]])  # old files contain a "CH" total row
        return df.to_csv(sep=";", index=False).encode("utf-8-sig")
    df = pd.DataFrame(
        {
            "Versicherer": b["bag_nr"],
            "Kanton": b["kanton"],
            "Geschäftsjahr": year,
            "Durchschnittsbestand": b["versicherte"],
        }
    )
    return df.to_csv(index=False).encode("utf-8-sig")


def _names_xlsx(path: Path) -> None:
    nrs = sorted(PREMIUMS["bag_nr"].unique())
    with pd.ExcelWriter(path) as w:
        grid: list[list[Any]] = [[None, None] for _ in range(3)]
        grid.append(["0", None])
        grid += [[n, f"Krankenkasse {n}"] for n in nrs]
        pd.DataFrame(grid).to_excel(w, sheet_name="2019", header=False, index=False)


def _b64(name: str) -> str:
    return base64.b64encode(f"/Praemien/{name}".encode()).decode()


def make_remote(root: Path, last_premium_year: int) -> dict[str, Any]:
    """Write the files the BAG would publish when ``last_premium_year`` is current.

    Returns a CKAN-like catalogue whose URLs point at those files.
    """
    root.mkdir(parents=True, exist_ok=True)
    resources = []

    def publish(name: str, content: bytes) -> None:
        (root / _b64(name).replace("/", "_")).write_bytes(content)
        resources.append({"url": f"https://bag.invalid/?r=/download&path={_b64(name)}"})

    for year in range(FIRST_PREMIUM_YEAR, last_premium_year):
        old = year <= 2012
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            name = "Praemien_CH.csv" if old else "Prämien_CH.csv"
            zf.writestr(name, _premium_csv(year, old, cp1252=(year == 2016)))
            zf.writestr(
                "Versicherte_CH.csv" if old else "Versichertenbestand_CH.csv",
                _insured_csv(year - 2, old),
            )
        publish(f"Archiv_Praemien_{year}.zip", buf.getvalue())
    publish("Prämien_CH.csv", _premium_csv(last_premium_year, False, False))
    publish("Versichertenbestand_CH.csv", _insured_csv(last_premium_year - 2, False))
    publish("Tarife.csv", b"irrelevant")
    return {"result": {"resources": resources}}


def install_remote(base: Path, mp: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Replace the network: ``publish(year)`` switches what the 'BAG' currently publishes."""
    state: dict[str, Any] = {}
    names = base / "remote_names.xlsx"
    _names_xlsx(names)

    def publish(year: int) -> None:
        state["catalog"] = make_remote(base / f"remote_{year}", year)
        state["dir"] = base / f"remote_{year}"

    def fake_download(url: str, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = (
            names
            if url.endswith(".xlsx")
            else state["dir"] / url.split("path=")[1].replace("/", "_")
        )
        dest.write_bytes(src.read_bytes())
        state.setdefault("downloads", []).append(dest.name)
        return data.sha256_of(dest)

    mp.setattr(data, "fetch_catalog", lambda url: state["catalog"])
    mp.setattr(data, "download", fake_download)
    publish.state = state  # type: ignore[attr-defined]
    publish(LAST_PREMIUM_YEAR)
    return publish


def make_settings(base: Path) -> Settings:
    cfg = {
        "data": {
            "catalog_url": "https://catalog.invalid",
            "names_url": "https://names.invalid/names.xlsx",
            "raw_dir": str(base / "raw"),
            "processed_dir": str(base / "processed"),
            "manifest_path": str(base / "manifest.json"),
            "first_year": 2011,
        },
        "features": {"min_versicherte": 100, "max_abs_target_per_year": 0.5},
        "model": {"max_iter": 30, "min_samples_leaf": 10},
        "horizons": [1, 2],
        "backtest": {"first_test_year": 2015},
        "registry": {"path": str(base / "models")},
    }
    path = base / "config" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return load_settings(path)


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    return install_remote(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


def config_file(settings: Settings) -> str:
    return str(settings.registry.path.parent / "config" / "config.yaml")


@pytest.fixture(autouse=True, scope="session")
def _single_thread() -> Iterator[None]:
    """The test data is tiny; OpenMP thread overhead would dominate (4x slower)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        yield
