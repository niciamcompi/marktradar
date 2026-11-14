from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from marktradar import data, pipeline
from marktradar.config import Settings
from marktradar.data import DataValidationError, Manifest

from .conftest import INSURED, LAST_PREMIUM_YEAR, make_remote


def test_catalog_files_decodes_names_and_filters(tmp_path: Path) -> None:
    catalog = make_remote(tmp_path, 2015)
    files = data.catalog_files(catalog, first_year=2013)
    assert data.CURRENT_PREMIUMS in files and data.CURRENT_INSURED in files
    assert sorted(k for k in files if k.startswith("archiv_")) == [
        "archiv_2013.zip",
        "archiv_2014.zip",
    ]
    assert "Tarife.csv" not in files


def test_catalog_without_current_files_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="lacks"):
        data.catalog_files({"result": {"resources": []}}, 2011)


@pytest.mark.parametrize(
    ("raw", "sep"),
    [("a;b\n1;ä\n".encode("cp1252"), ";"), ("a,b\n1,ä\n".encode("utf-8-sig"), ",")],
)
def test_read_csv_any(raw: bytes, sep: str) -> None:
    df = data.read_csv_any(raw)
    assert list(df.columns) == ["a", "b"] and df.iloc[0, 1] == "ä"


def test_sync_downloads_archives_only_once(
    settings: Settings, remote: Callable[[int], None]
) -> None:
    manifest, changed = pipeline.download(settings)
    assert changed and "archiv_2011.zip" in manifest.files
    manifest.save(settings.data.manifest_path)
    remote.state["downloads"] = []  # type: ignore[attr-defined]
    again, changed = pipeline.download(settings)
    assert not changed
    downloaded = remote.state["downloads"]  # type: ignore[attr-defined]
    assert not any(n.startswith("archiv_") for n in downloaded)
    assert data.CURRENT_PREMIUMS in downloaded


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = Manifest(files={"a": {"url": "u", "sha256": "1"}})
    m.save(tmp_path / "m.json")
    loaded = Manifest.load(tmp_path / "m.json")
    assert loaded.files == m.files and loaded.fingerprint() == m.fingerprint()
    assert Manifest.load(tmp_path / "missing.json").files == {}


def test_build_tables_handles_all_formats(
    settings: Settings, remote: Callable[[int], None]
) -> None:
    pipeline.download(settings)
    p, b = data.build_tables(settings.data.raw_dir, 2011)
    assert p["jahr"].min() == 2011 and p["jahr"].max() == LAST_PREMIUM_YEAR
    assert b["jahr"].min() == 2009 and b["jahr"].max() == LAST_PREMIUM_YEAR - 2
    assert "CH" not in set(b["kanton"])  # total rows of the old format are dropped
    # only the reference premium survives (no child or non-base rows)
    assert not p.duplicated(["bag_nr", "kanton", "jahr"]).any()
    expected = INSURED[INSURED["jahr"] <= LAST_PREMIUM_YEAR - 2]["versicherte"].sum()
    assert b["versicherte"].sum() == pytest.approx(expected)
    data.validate(p, b)


def test_validate_reports_problems(settings: Settings, remote: Callable[[int], None]) -> None:
    pipeline.download(settings)
    p, b = data.build_tables(settings.data.raw_dir, 2011)
    with pytest.raises(DataValidationError, match="duplicate"):
        data.validate(pd.concat([p, p.head(1)]), b)
    with pytest.raises(DataValidationError, match="contiguous"):
        data.validate(p[p["jahr"] != 2015], b)
    with pytest.raises(DataValidationError, match="implausible premium"):
        data.validate(p.assign(praemie=5.0), b)


def test_parse_names(settings: Settings, remote: Callable[[int], None]) -> None:
    pipeline.download(settings)
    names = data.parse_names(settings.data.raw_dir / data.NAMES_FILE)
    assert names.set_index("bag_nr").loc[8, "name"] == "Krankenkasse 8"
