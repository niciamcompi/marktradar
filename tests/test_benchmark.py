"""Model comparison: shared backtest, Chronos input building (no PyTorch needed)."""

import numpy as np
import pandas as pd
import pytest

from marktradar import benchmark
from marktradar.config import Settings
from marktradar.features import build_features, training_frame

from .conftest import INSURED, LAST_PREMIUM_YEAR, PREMIUMS

BESTAND = INSURED[INSURED["jahr"] <= LAST_PREMIUM_YEAR - 2]


@pytest.fixture(scope="module")
def frames() -> dict[int, pd.DataFrame]:
    f = build_features(PREMIUMS, BESTAND, [1, 2])
    return {h: training_frame(f, h, 100, 0.5) for h in (1, 2)}


def test_light_models_in_same_backtest(frames: dict[int, pd.DataFrame], settings: Settings) -> None:
    settings.model.max_iter = 30
    ctx = benchmark.Context(PREMIUMS, BESTAND, settings)
    cands = benchmark.select(
        ["konstant", "trend", "ridge", "random_forest", "gbm"], include_slow=False
    )
    results = benchmark.run(ctx, frames, cands, first_test_year=2015)
    # every model is evaluated on exactly the same test years
    years = results.groupby(["horizont", "modell"])["basisjahr"].apply(tuple)
    assert years.groupby(level=0).nunique().eq(1).all()
    summary = benchmark.summarize(results)
    by_model = summary[summary["horizont"] == 1].set_index("modell")
    assert by_model.loc["konstant", "verbesserung_vs_konstant"] == pytest.approx(0)
    # the simulated premium effect must be found by the learning models
    assert by_model.loc["gbm", "mae_modell"] < by_model.loc["konstant", "mae_modell"]
    assert by_model.loc["ridge", "mae_modell"] < by_model.loc["konstant", "mae_modell"]
    report = benchmark.markdown_report(summary)
    assert "Gradient boosting (production model)" in report and "nan" not in report


def test_select() -> None:
    keys = {c.key for c in benchmark.select(None, include_slow=False)}
    assert "chronos2_ft" not in keys and "tabpfn_v2" in keys
    assert "chronos2_ft" in {c.key for c in benchmark.select(None, include_slow=True)}
    with pytest.raises(ValueError, match="Unknown"):
        benchmark.select(["gpt"], include_slow=False)


def test_chronos_inputs_use_only_known_information() -> None:
    hist, cov = benchmark.series_tables(PREMIUMS, BESTAND)
    items = pd.DataFrame({"bag_nr": [8, 32], "kanton": ["ZH", "BE"]})
    past, future = benchmark.chronos_inputs(hist, cov, items, base_year=2015, h=2)
    assert past["jahr"].max() == 2015  # no shares from the future
    assert sorted(future["jahr"].unique()) == [2016, 2017]  # only published premiums
    assert set(past["item_id"]) == {"8_ZH", "32_BE"} == set(future["item_id"])
    assert future.groupby("item_id").size().eq(2).all()
    assert past[["rel", "schock"]].notna().all().all()


def test_chronos_inputs_cut_gaps() -> None:
    hist, cov = benchmark.series_tables(PREMIUMS, BESTAND)
    hist = hist[~((hist["bag_nr"] == 8) & (hist["kanton"] == "ZH") & (hist["jahr"] == 2012))]
    past, _ = benchmark.chronos_inputs(
        hist, cov, pd.DataFrame({"bag_nr": [8], "kanton": ["ZH"]}), 2015, 1
    )
    assert past["jahr"].min() == 2013  # only the gap-free stretch after the hole
    assert np.all(np.diff(past["jahr"]) == 1)
