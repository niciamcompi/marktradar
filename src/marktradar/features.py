"""Feature engineering.

One row = one insurer in one canton in base year *t* (the latest year with insured data).

The core idea: people switch because of the premium *compared with the competition in
their canton*, not because of the absolute premium. So prices are expressed relative to
the canton market average, and the target is the change of the insurer's market share
*within the canton*.

Timing (the most important thing to get right): every feature must be known at the time
a forecast is made. Approved premiums for year t+1 are published by the BAG in late
September of year t, before the switching deadline (30 November). Insured numbers lag
by two years. So with insured data up to 2025 and premiums up to 2027 we can forecast
2026 (horizon 1) and 2027 (horizon 2) without peeking into the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["bag_nr", "kanton", "jahr"]

BASE_FEATURES = [
    "rel_praemie",  # premium level vs. canton market in year t (log ratio)
    "delta_rel_praemie",  # last premium change vs. market (t-1 -> t)
    "anteilswachstum_vorjahr",  # momentum: share change t-1 -> t
    "log_anteil",  # size within the canton
]


def lever(h: int) -> str:
    """Premium change vs. canton market from year t+h-1 to t+h (the scenario lever)."""
    return f"hebel_{h}"


def shock(h: int) -> str:
    """Average premium increase of the canton market in year t+h (drives switching volume)."""
    return f"schock_{h}"


def target(h: int) -> str:
    return f"ziel_{h}"


def features_for(h: int) -> list[str]:
    extra = [c for k in range(1, h + 1) for c in (shock(k), lever(k))]
    return BASE_FEATURES + extra


def monotonic_constraints(h: int) -> dict[str, int]:
    """Domain knowledge as a hard constraint: a relatively higher premium may never
    *increase* the predicted growth. Makes the model more robust and easier to trust."""
    return {"rel_praemie": -1, **{lever(k): -1 for k in range(1, h + 1)}}


def market_premiums(praemien: pd.DataFrame, bestand: pd.DataFrame) -> pd.DataFrame:
    """Insured-weighted (geometric) mean premium per canton and year.

    Weights come from the insured numbers of the same year, or of the latest year
    available for premium years without insured data yet.
    """
    last = int(bestand["jahr"].max())
    p = praemien.assign(gewichtsjahr=praemien["jahr"].clip(upper=last))
    w = bestand.rename(columns={"jahr": "gewichtsjahr", "versicherte": "gewicht"})
    p = p.merge(w, on=["bag_nr", "kanton", "gewichtsjahr"], how="left")
    # insurers without insured persons in that canton get a tiny weight instead of none
    p["gewicht"] = p["gewicht"].fillna(0) + 1e-6
    p["logp_w"] = np.log(p["praemie"]) * p["gewicht"]
    agg = p.groupby(["kanton", "jahr"])[["logp_w", "gewicht"]].sum()
    markt = pd.Series(np.exp(agg["logp_w"] / agg["gewicht"]), name="markt_praemie")
    return markt.reset_index()


def _shifted(values: pd.Series, frame: pd.DataFrame, offset: int, keys: list[str]) -> np.ndarray:
    """Look up ``values`` (indexed by ``keys``) at year ``jahr + offset`` for each row."""
    idx = pd.MultiIndex.from_arrays(
        [frame[k] + offset if k == "jahr" else frame[k] for k in keys], names=keys
    )
    return np.asarray(values.reindex(idx))


def build_features(
    praemien: pd.DataFrame, bestand: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """Premium and insured tables -> one feature row per (insurer, canton, base year).

    Targets are NaN where the future is not yet known; levers are NaN where the
    premium of that year is not yet published.
    """
    markt = market_premiums(praemien, bestand)
    p = praemien.merge(markt, on=["kanton", "jahr"])
    rel = pd.Series(
        np.log(p["praemie"].to_numpy() / p["markt_praemie"].to_numpy()),
        index=pd.MultiIndex.from_frame(p[KEYS]),
    )
    mkt = markt.set_index(["kanton", "jahr"])["markt_praemie"]

    df = bestand.copy()
    df["kanton_versicherte"] = df.groupby(["kanton", "jahr"])["versicherte"].transform("sum")
    df["log_anteil"] = np.log(df["versicherte"] / df["kanton_versicherte"])
    share = df.set_index(KEYS)["log_anteil"]

    df["rel_praemie"] = _shifted(rel, df, 0, KEYS)
    df["delta_rel_praemie"] = df["rel_praemie"] - _shifted(rel, df, -1, KEYS)
    df["anteilswachstum_vorjahr"] = df["log_anteil"] - _shifted(share, df, -1, KEYS)
    df["praemie"] = _shifted(p.set_index(KEYS)["praemie"], df, 0, KEYS)
    for h in range(1, max(horizons) + 1):
        df[lever(h)] = _shifted(rel, df, h, KEYS) - _shifted(rel, df, h - 1, KEYS)
        df[shock(h)] = np.log(
            _shifted(mkt, df, h, ["kanton", "jahr"]) / _shifted(mkt, df, h - 1, ["kanton", "jahr"])
        )
    for h in horizons:
        df[target(h)] = _shifted(share, df, h, KEYS) - df["log_anteil"]
    return df.reset_index(drop=True)


def training_frame(
    features: pd.DataFrame, h: int, min_versicherte: float, max_abs_target_per_year: float
) -> pd.DataFrame:
    """Rows usable for training/evaluating horizon ``h``.

    Excluded: unknown future, tiny insurer-canton combinations, structural breaks
    (mergers/portfolio transfers) and rows without the required premium information.
    """
    y = features[target(h)]
    mask = (
        y.notna()
        & (features["versicherte"] >= min_versicherte)
        & (y.abs() <= max_abs_target_per_year * h)
        & features["rel_praemie"].notna()
        & features[[lever(k) for k in range(1, h + 1)]].notna().all(axis=1)
    )
    return features.loc[mask].reset_index(drop=True)


def sample_weight(frame: pd.DataFrame) -> pd.Series:
    """Large portfolios count more, but not linearly (otherwise a few cantons dominate)."""
    weights: pd.Series = np.sqrt(frame["versicherte"])
    return weights


def pp_to_lever(prozentpunkte: float) -> float:
    """Premium change vs. market in percentage points -> log ratio (feature scale)."""
    return float(np.log1p(prozentpunkte / 100.0))


def lever_to_pp(value: float) -> float:
    return float(np.expm1(value) * 100.0)
