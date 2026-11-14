import numpy as np
import pandas as pd
import pytest

from marktradar.config import ModelConfig
from marktradar.features import build_features, lever, target, training_frame
from marktradar.model import backtest, conformal_margin, fit, national, summarize

from .conftest import INSURED, LAST_PREMIUM_YEAR, PREMIUMS

CFG = ModelConfig(max_iter=30, min_samples_leaf=10)


@pytest.fixture(scope="module")
def frames() -> dict[int, pd.DataFrame]:
    f = build_features(PREMIUMS, INSURED[INSURED["jahr"] <= LAST_PREMIUM_YEAR - 2], [1, 2])
    return {h: training_frame(f, h, 100, 0.5) for h in (1, 2)}


@pytest.mark.parametrize("h", [1, 2])
def test_model_learns_premium_effect(frames: dict[int, pd.DataFrame], h: int) -> None:
    model = fit(frames[h], CFG, h)
    X = frames[h].head(30)
    low = model.predict(X.assign(**{lever(h): -0.04}))["prognose"]
    high = model.predict(X.assign(**{lever(h): 0.04}))["prognose"]
    assert (high <= low).all()  # monotonic constraint
    assert (low - high).mean() > 0.05  # simulated elasticity is -2


def test_interval_contains_point(frames: dict[int, pd.DataFrame]) -> None:
    pred = fit(frames[1], CFG, 1, conformal_margin=0.01).predict(frames[1])
    assert (pred["untergrenze"] <= pred["prognose"]).all()
    assert (pred["prognose"] <= pred["obergrenze"]).all()


@pytest.mark.parametrize("h", [1, 2])
def test_backtest_is_out_of_time(
    frames: dict[int, pd.DataFrame], h: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marktradar.model as m

    seen: list[int] = []
    real_fit = m.fit

    def spy(
        train: pd.DataFrame, cfg: ModelConfig, hh: int, conformal_margin: float = 0.0
    ) -> m.HorizonModel:
        seen.append(int(train["jahr"].max()))
        return real_fit(train, cfg, hh, conformal_margin)

    monkeypatch.setattr(m, "fit", spy)
    table = m.backtest(frames[h], CFG, h, first_test_year=2015).table
    for max_train_year, test_year in zip(seen, table["basisjahr"], strict=True):
        assert max_train_year <= test_year - h  # outcome of training rows known at test time
    s = summarize(table)
    assert s["mae_modell"] < s["mae_konstant"]


def test_backtest_scores_cover_all_test_rows(frames: dict[int, pd.DataFrame]) -> None:
    result = backtest(frames[1], CFG, 1, first_test_year=2015)
    assert len(result.scores) == int(result.table["n_kantonszeilen"].sum())


def test_national_aggregation() -> None:
    rows = pd.DataFrame(
        {"bag_nr": [1, 1, 2], "versicherte": [100.0, 300.0, 50.0], target(1): [0.0, 0.0, 0.1]}
    )
    pred = pd.DataFrame(
        {"prognose": [np.log(1.2), np.log(1.0), 0.0], "untergrenze": 0.0, "obergrenze": 0.0}
    )
    n = national(pred, rows, 1)
    # insurer 1: (100*1.2 + 300*1.0) / 400 = 1.05
    assert np.exp(n.loc[1, "prognose"]) == pytest.approx(1.05)
    assert n.loc[2, "ist"] == pytest.approx(0.1)


def test_conformal_margin() -> None:
    assert 0.6 <= conformal_margin(np.linspace(-1, 1, 101), 0.8) <= 0.66
    assert conformal_margin(np.array([-1.0, -0.5]), 0.5) == 0.0
    assert conformal_margin(np.array([]), 0.8) == 0.0
