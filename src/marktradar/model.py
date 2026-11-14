"""Model, training and time-correct backtest.

Per forecast horizon h (1 or 2 years) there is one bundle of:
- a point model (median / absolute error, robust against outliers),
- two quantile models for an 80 % prediction interval, re-calibrated with
  conformal prediction (see ``conformal_margin``),
- monotonic constraints on the premium features (see ``features.py``).

All models are scikit-learn ``HistGradientBoostingRegressor``: stable, well documented,
installs everywhere. For maintainability that matters more than the last 2 % accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from marktradar.config import ModelConfig
from marktradar.features import (
    features_for,
    monotonic_constraints,
    sample_weight,
    target,
)


def _regressor(
    cfg: ModelConfig, h: int, loss: str, quantile: float | None = None
) -> HistGradientBoostingRegressor:
    kwargs: dict[str, object] = {"quantile": quantile} if quantile is not None else {}
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        monotonic_cst=monotonic_constraints(h),
        random_state=cfg.random_state,
        **kwargs,
    )


@dataclass
class HorizonModel:
    """Point + interval models for one horizon behind a simple ``predict``."""

    horizon: int
    point: HistGradientBoostingRegressor
    lower: HistGradientBoostingRegressor
    upper: HistGradientBoostingRegressor
    conformal_margin: float = 0.0

    @property
    def features(self) -> list[str]:
        return features_for(self.horizon)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features]
        point = self.point.predict(X)
        lo = np.minimum(self.lower.predict(X), point) - self.conformal_margin
        hi = np.maximum(self.upper.predict(X), point) + self.conformal_margin
        return pd.DataFrame({"prognose": point, "untergrenze": lo, "obergrenze": hi}, index=X.index)


def fit(
    train: pd.DataFrame, cfg: ModelConfig, h: int, conformal_margin: float = 0.0
) -> HorizonModel:
    X, y, w = train[features_for(h)], train[target(h)], sample_weight(train)
    q_lo, q_hi = cfg.quantiles
    return HorizonModel(
        horizon=h,
        point=_regressor(cfg, h, "absolute_error").fit(X, y, sample_weight=w),
        lower=_regressor(cfg, h, "quantile", q_lo).fit(X, y, sample_weight=w),
        upper=_regressor(cfg, h, "quantile", q_hi).fit(X, y, sample_weight=w),
        conformal_margin=conformal_margin,
    )


# --------------------------------------------------------------------------- metrics


def _wmae(err: np.ndarray | pd.Series, w: np.ndarray | pd.Series) -> float:
    return float(np.average(np.abs(err), weights=w))


def national(pred: pd.DataFrame, rows: pd.DataFrame, h: int | None = None) -> pd.DataFrame:
    """Aggregate canton forecasts to a national share change per insurer.

    Assumes each canton market grows at the same rate, so the national share change is
    the insured-weighted average of the canton share changes (in level terms). Interval
    bounds are aggregated the same way, which assumes cantons move together (conservative).
    """
    df = rows[["bag_nr", "versicherte"]].join(pred)
    if h is not None:
        df["ist"] = rows[target(h)]
    cols = [c for c in ("prognose", "untergrenze", "obergrenze", "ist") if c in df]
    for c in cols:
        df[c] = df["versicherte"] * np.exp(df[c])
    g = df.groupby("bag_nr")[["versicherte", *cols]].sum()
    for c in cols:
        g[c] = np.log(g[c] / g["versicherte"])
    return g


def evaluate(pred: pd.DataFrame, rows: pd.DataFrame, h: int) -> dict[str, float]:
    """Metrics for one test year, canton level and aggregated per insurer (national)."""
    y, w = rows[target(h)], sample_weight(rows)
    inside = (y >= pred["untergrenze"]) & (y <= pred["obergrenze"])
    n = national(pred, rows, h)
    n = n[n["versicherte"] >= 1000]
    wn = np.sqrt(n["versicherte"])
    return {
        # national view (what sales cares about)
        "mae_modell": _wmae(n["prognose"] - n["ist"], wn),
        "mae_konstant": _wmae(n["ist"], wn),  # baseline: market share stays the same
        "rangkorrelation": float(n["prognose"].corr(n["ist"], method="spearman")),
        # canton view
        "kanton_mae_modell": _wmae(pred["prognose"] - y, w),
        "kanton_mae_konstant": _wmae(y, w),
        "kanton_mae_vorjahr": _wmae(rows["anteilswachstum_vorjahr"].fillna(0) * h - y, w),
        "intervall_abdeckung": float(np.average(inside, weights=w)),
        "n_versicherer": float(len(n)),
        "n_kantonszeilen": float(len(rows)),
    }


def conformity_scores(pred: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """How far the truth lies outside the interval (negative = inside)."""
    lower = pred["untergrenze"].to_numpy() - y.to_numpy()
    upper = y.to_numpy() - pred["obergrenze"].to_numpy()
    scores: np.ndarray = np.maximum(lower, upper)
    return scores


def conformal_margin(scores: np.ndarray, coverage: float) -> float:
    """Conformalized quantile regression (Romano et al., 2019).

    Quantile models tend to be over-confident. From errors on *unseen* data (the
    backtest) we learn how much to widen the interval so it really reaches the
    target coverage (e.g. 80 %).
    """
    if len(scores) == 0:
        return 0.0
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return max(0.0, float(np.quantile(scores, level, method="higher")))


@dataclass
class BacktestResult:
    table: pd.DataFrame  # metrics per test year
    scores: np.ndarray  # conformity scores of all test years (for the final model)


def backtest(frame: pd.DataFrame, cfg: ModelConfig, h: int, first_test_year: int) -> BacktestResult:
    """Rolling-origin backtest: for base year T, train only on base years <= T - h.

    Base year T - h is the last one whose outcome (year T) is already known at T.
    This mirrors real use exactly. A random train/test split would leak information
    from the future and badly overstate accuracy.

    The conformal calibration is time-honest as well: for test year T the interval
    width is estimated only from errors of earlier test years.
    """
    coverage = cfg.quantiles[1] - cfg.quantiles[0]
    rows: list[dict[str, float]] = []
    past: list[np.ndarray] = []
    for year in sorted(y for y in frame["jahr"].unique() if y >= first_test_year):
        train = frame[frame["jahr"] <= year - h]
        test = frame[frame["jahr"] == year]
        if len(train) < 200 or len(test) < 20:
            continue
        model = fit(train, cfg, h)
        raw = model.predict(test)
        model.conformal_margin = conformal_margin(
            np.concatenate(past) if past else np.array([]), coverage
        )
        metrics = evaluate(model.predict(test), test, h)
        metrics["intervall_abdeckung_roh"] = evaluate(raw, test, h)["intervall_abdeckung"]
        rows.append({"basisjahr": int(year), "prognosejahr": int(year) + h, **metrics})
        past.append(conformity_scores(raw, test[target(h)]))
    if not rows:
        raise ValueError("Not enough data for a backtest")
    return BacktestResult(table=pd.DataFrame(rows), scores=np.concatenate(past))


def summarize(bt: pd.DataFrame) -> dict[str, float]:
    s = bt.drop(columns=["basisjahr", "prognosejahr", "n_versicherer", "n_kantonszeilen"]).mean()
    # the first test year has no calibration data yet -> leave it out for coverage
    s["intervall_abdeckung"] = bt["intervall_abdeckung"].iloc[1:].mean()
    summary = {str(k): round(float(v), 4) for k, v in s.items()}
    summary["verbesserung_vs_konstant"] = round(
        1 - summary["mae_modell"] / summary["mae_konstant"], 4
    )
    summary["testjahre"] = int(len(bt))
    return summary
