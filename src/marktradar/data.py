"""Fetch, parse and validate the BAG open data.

Main source: the BAG premium open data (opendata.swiss, "health-insurance-premiums").
It contains, for every insurer:

* the approved premium tariffs per canton and region (one file per premium year,
  current year plus yearly archives back to 2011), and
* the average number of insured persons per canton (two years behind the premiums).

The files changed format twice (2011-2014, 2015-2026, 2027+), with different column
names, separators and encodings. Everything here normalises them into two tidy tables:

    praemien:  bag_nr, kanton, jahr, praemie        (reference premium, CHF/month)
    bestand:   bag_nr, kanton, jahr, versicherte    (average insured persons)

Insurer names come from the BAG supervisory data (only used as a lookup table).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from marktradar import __version__

log = logging.getLogger(__name__)

CANTONS = tuple(
    "AG AI AR BE BL BS FR GE GL GR JU LU NE NW OW SG SH SO SZ TG TI UR VD VS ZG ZH".split()  # noqa: SIM905
)
CURRENT_PREMIUMS = "Praemien_CH.csv"
CURRENT_INSURED = "Versichertenbestand_CH.csv"
NAMES_FILE = "aufsichtsdaten.xlsx"


class DataValidationError(ValueError):
    """The data violates an assumption the model relies on."""


# --------------------------------------------------------------------------- download


def _headers() -> dict[str, str]:
    return {"User-Agent": f"marktradar/{__version__} (open-data research client)"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> str:
    """Download ``url`` atomically to ``dest`` and return its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with httpx.stream("GET", url, headers=_headers(), follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                h.update(chunk)
    tmp.replace(dest)
    return h.hexdigest()


def _resource_filename(url: str) -> str:
    """opendata.bagnet.ch encodes the file path as base64 in the ``path`` parameter."""
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "path" in query:
        try:
            return Path(base64.b64decode(query["path"][0]).decode("utf-8")).name
        except ValueError:
            pass
    return Path(urllib.parse.urlparse(url).path).name


def catalog_files(catalog: dict[str, Any], first_year: int) -> dict[str, str]:
    """Map local file names to download URLs for the files the model needs."""
    files: dict[str, str] = {}
    for res in catalog["result"]["resources"]:
        url = res.get("url") or res.get("download_url") or ""
        name = _resource_filename(url)
        archive = re.fullmatch(r"Archiv_Praemien_(\d{4})\.zip", name)
        if archive and int(archive.group(1)) >= first_year:
            files[f"archiv_{archive.group(1)}.zip"] = url
        elif re.fullmatch(r"Pr\w{1,3}mien_CH\.csv", name):
            files[CURRENT_PREMIUMS] = url
        elif name == "Versichertenbestand_CH.csv":
            files[CURRENT_INSURED] = url
    missing = {CURRENT_PREMIUMS, CURRENT_INSURED} - set(files)
    if missing:
        raise DataValidationError(f"Catalogue lacks {sorted(missing)}")
    return files


def fetch_catalog(url: str) -> dict[str, Any]:
    r = httpx.get(url, headers=_headers(), follow_redirects=True, timeout=60)
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    return data


@dataclass
class Manifest:
    """Which file versions the data (and thus the model) was built from."""

    files: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls()
        return cls(files=json.loads(path.read_text(encoding="utf-8"))["files"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"files": dict(sorted(self.files.items()))}
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fingerprint(self) -> str:
        joined = "|".join(f"{k}:{v['sha256']}" for k, v in sorted(self.files.items()))
        return hashlib.sha256(joined.encode()).hexdigest()


def sync(
    catalog_url: str,
    names_url: str | None,
    raw_dir: Path,
    previous: Manifest,
    first_year: int,
) -> Manifest:
    """Bring ``raw_dir`` up to date and return the new manifest.

    Yearly archives never change, so they are only downloaded when missing.
    The current-year files are always re-downloaded and compared by hash.
    """
    files = catalog_files(fetch_catalog(catalog_url), first_year)
    if names_url:
        files[NAMES_FILE] = names_url
    manifest = Manifest()
    for name, url in sorted(files.items()):
        dest = raw_dir / name
        known = previous.files.get(name)
        immutable = name.startswith("archiv_")
        if immutable and known and dest.exists() and sha256_of(dest) == known["sha256"]:
            manifest.files[name] = known
            continue
        log.info("Downloading %s", name)
        manifest.files[name] = {"url": url, "sha256": download(url, dest)}
    return manifest


# --------------------------------------------------------------------------- parsing


def read_csv_any(raw: bytes) -> pd.DataFrame:
    """Read a BAG CSV regardless of encoding (UTF-8/cp1252) and separator (,/;)."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 decodes any byte string
        raise DataValidationError("Unknown encoding")
    header = text.split("\n", 1)[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


def parse_premiums(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the reference premium: adult, standard model, CHF 300 deductible, with accident.

    That is the premium every insurer must offer, so it is comparable across insurers
    and years. Regions within a canton are averaged.
    """
    if "G_ID" in df.columns:  # 2011-2014 format
        mask = (df["M_ID"] == "26") & (df["isBASE_P"] == "1") & (df["isBASE_F"] == "1")
        out = pd.DataFrame(
            {"bag_nr": df["G_ID"], "kanton": df["C_ID"], "jahr": df["JAHR"], "praemie": df["P"]}
        )
    else:  # 2015 onwards
        accident = df["Unfalleinschluss"].str.replace("-", "_", regex=False)
        mask = (
            df["Altersklasse"].str.contains("ERW", na=False)
            & accident.str.startswith("MIT", na=False)
            & (df["isBaseP"] == "1")
            & (df["isBaseF"] == "1")
        )
        out = pd.DataFrame(
            {
                "bag_nr": df["Versicherer"],
                "kanton": df["Kanton"],
                "jahr": df["Geschäftsjahr"],
                "praemie": df["Prämie"],
            }
        )
    out = out[mask].copy()
    out["bag_nr"] = _num(out["bag_nr"]).astype(int)
    out["jahr"] = _num(out["jahr"]).astype(int)
    out["praemie"] = _num(out["praemie"])
    out = out[out["kanton"].isin(CANTONS) & (out["praemie"] > 0)]
    result: pd.DataFrame = out.groupby(["bag_nr", "kanton", "jahr"], as_index=False)[
        ["praemie"]
    ].mean()
    return result


def parse_insured(df: pd.DataFrame) -> pd.DataFrame:
    if "G_ID" in df.columns:
        cols = {"G_ID": "bag_nr", "C_ID": "kanton", "JAHR": "jahr", "B": "versicherte"}
    else:
        cols = {
            "Versicherer": "bag_nr",
            "Kanton": "kanton",
            "Geschäftsjahr": "jahr",
            "Durchschnittsbestand": "versicherte",
        }
    out = df.rename(columns=cols)[list(cols.values())].copy()
    out["bag_nr"] = _num(out["bag_nr"]).astype(int)
    out["jahr"] = _num(out["jahr"]).astype(int)
    out["versicherte"] = _num(out["versicherte"])
    # Older files also contain a "CH" total row and non-canton codes -> drop them
    result: pd.DataFrame = out[out["kanton"].isin(CANTONS) & (out["versicherte"] > 0)]
    return result.reset_index(drop=True)


def _archive_member(zf: zipfile.ZipFile, pattern: str) -> bytes:
    for name in zf.namelist():
        if re.search(pattern, name):
            return zf.read(name)
    raise DataValidationError(f"{zf.filename}: no file matching {pattern}")


def build_tables(raw_dir: Path, first_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all archives and current files into the premium and insured tables."""
    premiums, insured = [], []
    for path in sorted(raw_dir.glob("archiv_*.zip")):
        with zipfile.ZipFile(path) as zf:
            premiums.append(
                parse_premiums(read_csv_any(_archive_member(zf, r"Pr.{0,3}mien_CH_?\.csv$")))
            )
            insured.append(
                parse_insured(
                    read_csv_any(_archive_member(zf, r"Versichert(e|enbestand)_CH\.csv$"))
                )
            )
    premiums.append(parse_premiums(read_csv_any((raw_dir / CURRENT_PREMIUMS).read_bytes())))
    insured.append(parse_insured(read_csv_any((raw_dir / CURRENT_INSURED).read_bytes())))

    keys = ["bag_nr", "kanton", "jahr"]
    # later files win if the same year appears twice (e.g. corrections)
    p = pd.concat(premiums).drop_duplicates(keys, keep="last")
    b = pd.concat(insured).drop_duplicates(keys, keep="last")
    p = p[p["jahr"] >= first_year].sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    return p, b


def validate(praemien: pd.DataFrame, bestand: pd.DataFrame) -> None:
    """Plausibility checks. Raises ``DataValidationError`` with all problems found."""
    errors: list[str] = []
    for name, df in (("praemien", praemien), ("bestand", bestand)):
        if df.duplicated(["bag_nr", "kanton", "jahr"]).any():
            errors.append(f"{name}: duplicate (bag_nr, kanton, jahr)")
        years = sorted(df["jahr"].unique())
        if years != list(range(years[0], years[-1] + 1)):
            errors.append(f"{name}: years not contiguous: {years}")
    # Adult reference premium has been between ~200 and ~1000 CHF/month since 2011
    if not praemien["praemie"].between(100, 2000).all():
        errors.append("praemien: implausible premium (outside 100-2000 CHF)")
    # Switzerland has ~8-9.5 million insured persons
    totals = bestand.groupby("jahr")["versicherte"].sum()
    if not totals.between(6e6, 12e6).all():
        errors.append(f"bestand: implausible national total {totals.to_dict()}")
    if bestand["jahr"].max() >= praemien["jahr"].max():
        errors.append("bestand: insured data must lag behind premium data")
    if errors:
        raise DataValidationError("; ".join(errors))


# --------------------------------------------------------------------------- names


def _find_code_row(sheet: pd.DataFrame, max_rows: int = 20) -> int:
    for i in range(min(max_rows, len(sheet))):
        if str(sheet.iloc[i, 0]).strip().removesuffix(".0") == "0":
            return i
    raise DataValidationError("No column-code row found")


def parse_names(path: Path) -> pd.DataFrame:
    """Latest insurer name per BAG number from the supervisory data workbook."""
    book = pd.ExcelFile(path)
    rows = []
    for sheet_name in book.sheet_names:
        if not str(sheet_name).isdigit():
            continue
        sheet = pd.read_excel(book, sheet_name=sheet_name, header=None)
        try:
            start = _find_code_row(sheet) + 1
        except DataValidationError:
            continue
        body = sheet.iloc[start:, :2].copy()
        body.columns = ["bag_nr", "name"]
        body["bag_nr"] = pd.to_numeric(body["bag_nr"], errors="coerce")
        body = body.dropna()
        body["jahr"] = int(sheet_name)
        rows.append(body)
    if not rows:
        return pd.DataFrame({"bag_nr": pd.Series(dtype=int), "name": pd.Series(dtype=str)})
    names = pd.concat(rows)
    names["bag_nr"] = names["bag_nr"].astype(int)
    names["name"] = names["name"].astype(str).str.strip()
    latest = names.sort_values("jahr").drop_duplicates("bag_nr", keep="last")
    return latest[["bag_nr", "name"]].reset_index(drop=True)
