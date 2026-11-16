"""Model comparison: from simple rules to transformer foundation models.

Every candidate is evaluated in exactly the same rolling-origin backtest as the
production model (same test years, same rows, same metrics), so the numbers are
directly comparable. Run it with ``uv run marktradar compare`` (needs
``uv sync --extra benchmark`` for the heavy models).

Candidates, roughly from oldest to newest idea:

==================  ==========================================================
Baselines           "share stays the same", "last year's trend continues"
Linear regression   Ridge regression on the same features (classic statistics)
Random forest       many decision trees, averaged (2001)
Gradient boosting   the production model (scikit-learn, with monotonic rules)
LightGBM            a popular, faster gradient boosting library (2017)
TabPFN v2           transformer pre-trained on millions of synthetic tables;
                    learns "in context" from our training rows (2025)
Chronos-2           Amazon's transformer for time series, zero-shot (2025);
                    optionally fine-tuned on our series (slow on CPU)
==================  ==========================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marktradar.config import Settings
from marktradar.features import KEYS, features_for, market_premiums, monotonic_constraints, target
from marktradar.model import evaluate, fit

Prediction = pd.DataFrame  # columns: prognose, untergrenze, obergrenze (bounds may be NaN)


@dataclass
class Context:
    """Data some candidates need beyond the feature rows (e.g. full time series)."""

    praemien: pd.DataFrame
    bestand: pd.DataFrame
    settings: Settings
    cache: dict[str, Any] = field(default_factory=dict)


FitPredict = Callable[[Context, pd.DataFrame, pd.DataFrame, int], Prediction]


@dataclass
class Candidate:
    key: str
    name: str
    family: str
    year: int  # when the method became popular, for the "old to new" ordering
    fit_predict: FitPredict
    heavy: bool = False  # needs the `benchmark` extra
    slow: bool = False  # only run with --slow


def _frame(
    point: np.ndarray, index: pd.Index, lo: np.ndarray | None = None, hi: np.ndarray | None = None
) -> Prediction:
    nan = np.full(len(index), np.nan)
    return pd.DataFrame(
        {
            "prognose": point,
            "untergrenze": nan if lo is None else lo,
            "obergrenze": nan if hi is None else hi,
        },
        index=index,
    )


# --------------------------------------------------------------------------- baselines


def _constant(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    return _frame(np.zeros(len(test)), test.index)


def _trend(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    return _frame(test["anteilswachstum_vorjahr"].fillna(0).to_numpy() * h, test.index)


# --------------------------------------------------------------------------- classic ML


def _xy(
    train: pd.DataFrame, test: pd.DataFrame, h: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    cols = features_for(h)
    return train[cols], train[target(h)], test[cols], np.sqrt(train["versicherte"].to_numpy())


def _ridge(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, Xt, w = _xy(train, test, h)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y, ridge__sample_weight=w)
    return _frame(model.predict(Xt), test.index)


def _random_forest(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    from sklearn.ensemble import RandomForestRegressor

    X, y, Xt, w = _xy(train, test, h)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=0, n_jobs=1)
    model.fit(X.fillna(0), y, sample_weight=w)
    return _frame(model.predict(Xt.fillna(0)), test.index)


def _production(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    return fit(train, ctx.settings.model, h).predict(test)


def _lightgbm(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    import lightgbm as lgb

    X, y, Xt, w = _xy(train, test, h)
    mono = monotonic_constraints(h)
    params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 40,
        "verbose": -1,
        "n_jobs": 1,
    }
    # LightGBM supports monotonic constraints only for squared error, not for the
    # absolute-error or quantile losses, so the interval models run without them.
    constraints = {"monotone_constraints": [mono.get(c, 0) for c in X.columns]}
    out: dict[str, np.ndarray] = {}
    for name, kwargs in (
        ("prognose", {"objective": "regression", **constraints}),
        ("untergrenze", {"objective": "quantile", "alpha": 0.1}),
        ("obergrenze", {"objective": "quantile", "alpha": 0.9}),
    ):
        model = lgb.LGBMRegressor(**params, **kwargs).fit(X, y, sample_weight=w)
        out[name] = np.asarray(model.predict(Xt))
    return _frame(out["prognose"], test.index, out["untergrenze"], out["obergrenze"])


# --------------------------------------------------------------------------- transformers


def _tabpfn(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    """TabPFN v2 (Prior Labs License, Apache-2.0 based, commercial use allowed).

    Newer TabPFN versions (2.5+) are licensed for non-commercial use only, so v2 is
    pinned deliberately.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    X, y, Xt, _ = _xy(train, test, h)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=0, ignore_pretraining_limits=True
    )
    model.fit(X, y)
    q = model.predict(Xt, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    return _frame(np.asarray(q[1]), test.index, np.asarray(q[0]), np.asarray(q[2]))


def series_tables(
    praemien: pd.DataFrame, bestand: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long time-series tables for sequence models.

    Returns (history, covariates):
    - history: bag_nr, kanton, jahr, ls (log share in canton) for all insured years
    - covariates: bag_nr, kanton, jahr, rel (premium vs. canton market), schock
      (canton market premium change) for all premium years
    """
    markt = market_premiums(praemien, bestand).sort_values(["kanton", "jahr"])
    markt["schock"] = np.log(markt["markt_praemie"]).groupby(markt["kanton"]).diff()
    cov = praemien.merge(markt, on=["kanton", "jahr"])
    cov["rel"] = np.log(cov["praemie"] / cov["markt_praemie"])
    hist = bestand.copy()
    hist["ls"] = np.log(
        hist["versicherte"] / hist.groupby(["kanton", "jahr"])["versicherte"].transform("sum")
    )
    return hist[[*KEYS, "ls"]], cov[[*KEYS, "rel", "schock"]]


def chronos_inputs(
    hist: pd.DataFrame, cov: pd.DataFrame, items: pd.DataFrame, base_year: int, h: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past and future frames in Chronos' long format for the given (bag_nr, kanton) items.

    Only information available in ``base_year`` is used: shares up to ``base_year``,
    premiums up to ``base_year + h`` (approved premiums are published ahead).
    Each series is cut to its most recent gap-free stretch.
    """
    wanted = items[["bag_nr", "kanton"]].drop_duplicates()
    past = hist[hist["jahr"] <= base_year].merge(wanted, on=["bag_nr", "kanton"])
    past = past.merge(cov, on=KEYS, how="left").sort_values(KEYS)
    past["item_id"] = past["bag_nr"].astype(str) + "_" + past["kanton"]
    gap = past.groupby("item_id")["jahr"].diff().fillna(1).ne(1).astype(int)
    # number of gaps *after* each row (the row right after a gap starts the kept stretch)
    later_gaps = gap.iloc[::-1].groupby(past["item_id"].iloc[::-1]).cumsum().iloc[::-1] - gap
    past = past[later_gaps.eq(0)]
    past[["rel", "schock"]] = (
        past.groupby("item_id")[["rel", "schock"]].transform(lambda s: s.ffill().bfill()).fillna(0)
    )
    future = cov[(cov["jahr"] > base_year) & (cov["jahr"] <= base_year + h)].copy()
    future["item_id"] = future["bag_nr"].astype(str) + "_" + future["kanton"]
    complete = future.groupby("item_id")["jahr"].count().eq(h)
    keep = set(past["item_id"]) & set(complete[complete].index)
    past, future = past[past["item_id"].isin(keep)], future[future["item_id"].isin(keep)]
    for df in (past, future):
        df["timestamp"] = pd.to_datetime(df["jahr"].astype(str) + "-01-01")
    future["schock"] = future["schock"].fillna(0)
    return past.reset_index(drop=True), future.sort_values(["item_id", "jahr"]).reset_index(
        drop=True
    )


def _chronos_pipeline(ctx: Context) -> Any:
    if "chronos" not in ctx.cache:
        from chronos import Chronos2Pipeline

        ctx.cache["chronos"] = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map="cpu"
        )
        ctx.cache["series"] = series_tables(ctx.praemien, ctx.bestand)
    return ctx.cache["chronos"]


def _chronos_predict(
    ctx: Context, pipe: Any, test: pd.DataFrame, h: int, covariates: bool
) -> Prediction:
    hist, cov = ctx.cache["series"]
    base_year = int(test["jahr"].iloc[0])
    past, future = chronos_inputs(hist, cov, test, base_year, h)
    cols = ["item_id", "timestamp", "ls", *(["rel", "schock"] if covariates else [])]
    out = pipe.predict_df(
        past[cols],
        future_df=future[["item_id", "timestamp", "rel", "schock"]] if covariates else None,
        target="ls",
        prediction_length=h,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=100,
    )
    last = out.sort_values("timestamp").groupby("item_id").tail(1).set_index("item_id")
    base = past.groupby("item_id")["ls"].last()
    key = test["bag_nr"].astype(str) + "_" + test["kanton"]
    # series too short for Chronos fall back to "share stays the same"
    cols_q = {"prognose": "0.5", "untergrenze": "0.1", "obergrenze": "0.9"}
    pred = {c: (last[q] - base).reindex(key).fillna(0).to_numpy() for c, q in cols_q.items()}
    return _frame(pred["prognose"], test.index, pred["untergrenze"], pred["obergrenze"])


def _chronos_zero_shot(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    return _chronos_predict(ctx, _chronos_pipeline(ctx), test, h, covariates=False)


def _chronos_zero_shot_cov(
    ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int
) -> Prediction:
    return _chronos_predict(ctx, _chronos_pipeline(ctx), test, h, covariates=True)


def _chronos_finetuned(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, h: int) -> Prediction:
    """Fine-tunes Chronos-2 on all series up to the base year (≈3-4 min per year on CPU)."""
    import tempfile

    from chronos.chronos2 import preprocess

    pipe = _chronos_pipeline(ctx)
    hist, cov = ctx.cache["series"]
    base_year = int(test["jahr"].iloc[0])
    everything = hist[hist["jahr"] <= base_year][["bag_nr", "kanton"]]
    past, _ = chronos_inputs(hist, cov, everything, base_year, 0)
    past = past[past.groupby("item_id")["jahr"].transform("count") >= h + 2]
    inputs = preprocess.from_data_frame(
        past[["item_id", "timestamp", "ls", "rel", "schock"]],
        target_columns=["ls"],
        prediction_length=h,
        known_covariates_names=["rel", "schock"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        tuned = pipe.fit(
            inputs,
            prediction_length=h,
            finetune_mode="full",
            learning_rate=1e-5,
            num_steps=300,
            batch_size=64,
            output_dir=tmp,
            remove_printer_callback=True,
            min_past=2,
        )
        return _chronos_predict(ctx, tuned, test, h, covariates=True)


CANDIDATES: list[Candidate] = [
    Candidate("konstant", "Baseline: share stays the same", "Baseline", 0, _constant),
    Candidate("trend", "Baseline: last year's trend continues", "Baseline", 0, _trend),
    Candidate("ridge", "Linear regression (Ridge)", "Classic statistics", 1970, _ridge),
    Candidate("random_forest", "Random forest", "Classic ML", 2001, _random_forest),
    Candidate("gbm", "Gradient boosting (production model)", "Classic ML", 2017, _production),
    Candidate("lightgbm", "LightGBM", "Classic ML", 2017, _lightgbm, heavy=True),
    Candidate(
        "tabpfn_v2", "TabPFN v2 (tabular transformer)", "Transformer", 2025, _tabpfn, heavy=True
    ),
    Candidate(
        "chronos2",
        "Chronos-2 zero-shot, history only",
        "Transformer",
        2025,
        _chronos_zero_shot,
        heavy=True,
    ),
    Candidate(
        "chronos2_cov",
        "Chronos-2 zero-shot, with premiums",
        "Transformer",
        2025,
        _chronos_zero_shot_cov,
        heavy=True,
    ),
    Candidate(
        "chronos2_ft",
        "Chronos-2 fine-tuned, with premiums",
        "Transformer",
        2025,
        _chronos_finetuned,
        heavy=True,
        slow=True,
    ),
]


def select(keys: list[str] | None, include_slow: bool) -> list[Candidate]:
    if keys:
        unknown = set(keys) - {c.key for c in CANDIDATES}
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        return [c for c in CANDIDATES if c.key in keys]
    return [c for c in CANDIDATES if include_slow or not c.slow]


def run(
    ctx: Context,
    frames: dict[int, pd.DataFrame],
    candidates: list[Candidate],
    first_test_year: int,
    progress: Callable[[str], None] = lambda msg: None,
) -> pd.DataFrame:
    """Rolling-origin backtest for every candidate. One row per (model, horizon, test year)."""
    rows = []
    for h, frame in frames.items():
        years = sorted(y for y in frame["jahr"].unique() if y >= first_test_year)
        for cand in candidates:
            for year in years:
                train, test = frame[frame["jahr"] <= year - h], frame[frame["jahr"] == year]
                if len(train) < 200 or len(test) < 20:
                    continue
                start = time.perf_counter()
                pred = cand.fit_predict(ctx, train, test, h)
                seconds = time.perf_counter() - start
                metrics = evaluate(
                    pred.fillna({"untergrenze": -np.inf, "obergrenze": np.inf}), test, h
                )
                has_interval = bool(pred["untergrenze"].notna().all())
                rows.append(
                    {
                        "modell": cand.key,
                        "name": cand.name,
                        "familie": cand.family,
                        "jahr_methode": cand.year,
                        "horizont": h,
                        "basisjahr": int(year),
                        "prognosejahr": int(year) + h,
                        "mae_modell": metrics["mae_modell"],
                        "mae_konstant": metrics["mae_konstant"],
                        "rangkorrelation": metrics["rangkorrelation"],
                        "intervall_abdeckung": metrics["intervall_abdeckung"]
                        if has_interval
                        else np.nan,
                        "sekunden": seconds,
                    }
                )
            progress(f"horizon {h}: {cand.name} done")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model and horizon, sorted by error."""
    g = results.groupby(
        ["horizont", "modell", "name", "familie", "jahr_methode"], as_index=False
    ).agg(
        mae_modell=("mae_modell", "mean"),
        mae_konstant=("mae_konstant", "mean"),
        rangkorrelation=("rangkorrelation", "mean"),
        intervall_abdeckung=("intervall_abdeckung", "mean"),
        sekunden_pro_jahr=("sekunden", "mean"),
        testjahre=("basisjahr", "count"),
    )
    g["verbesserung_vs_konstant"] = 1 - g["mae_modell"] / g["mae_konstant"]
    return g.sort_values(["horizont", "mae_modell"]).reset_index(drop=True)


HEADER = (
    "| Model | Family | Error | Better than baseline | Rank corr. "
    "| 80 % interval coverage | Seconds per year |"
)


def markdown_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Model comparison",
        "",
        "Generated by `marktradar compare`. Every model runs in the same rolling backtest,",
        "with the same test years and metrics. Error = average error of the national",
        "market-share forecast per insurer (lower is better).",
        "",
    ]
    for h in sorted(summary["horizont"].unique()):
        g = summary[summary["horizont"] == h]
        lines += [f"## Forecast {h} year{'s' if h > 1 else ''} ahead", "", HEADER, "|---" * 7 + "|"]
        for r in g.to_dict(orient="records"):
            coverage = r["intervall_abdeckung"]
            cov = "–" if pd.isna(coverage) else f"{coverage:.0%}"
            rank = "–" if pd.isna(r["rangkorrelation"]) else f"{r['rangkorrelation']:.2f}"
            lines.append(
                f"| {r['name']} | {r['familie']} | {r['mae_modell']:.4f} | "
                f"{r['verbesserung_vs_konstant']:+.0%} | {rank} | {cov} | "
                f"{r['sekunden_pro_jahr']:.1f} |"
            )
        lines.append("")
    lines += [
        "Interval coverage is shown before conformal calibration (raw model output), so it is",
        "comparable across models. The production model is recalibrated afterwards.",
        "",
    ]
    return "\n".join(lines)


def save(
    results: pd.DataFrame, summary: pd.DataFrame, json_path: Path, md_path: Path | None
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "zusammenfassung": summary.replace({np.nan: None}).round(4).to_dict(orient="records"),
        "pro_jahr": results.replace({np.nan: None}).round(4).to_dict(orient="records"),
    }
    import json

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown_report(summary), encoding="utf-8")
