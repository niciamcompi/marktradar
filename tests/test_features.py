import numpy as np
import pandas as pd
import pytest

from marktradar.features import (
    build_features,
    features_for,
    lever,
    lever_to_pp,
    market_premiums,
    pp_to_lever,
    target,
    training_frame,
)

from .conftest import INSURED, LAST_PREMIUM_YEAR, PREMIUMS

BESTAND = INSURED[INSURED["jahr"] <= LAST_PREMIUM_YEAR - 2]
PRAEMIEN = PREMIUMS[PREMIUMS["jahr"] >= 2011]


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    return build_features(PRAEMIEN, BESTAND, [1, 2])


def _row(feats: pd.DataFrame, bag_nr: int, kanton: str, jahr: int) -> pd.Series:
    return feats.set_index(["bag_nr", "kanton", "jahr"]).loc[(bag_nr, kanton, jahr)]


def test_target_is_future_share_change(feats: pd.DataFrame) -> None:
    """Guards against leakage and off-by-one errors."""
    now, later = _row(feats, 8, "ZH", 2014), _row(feats, 8, "ZH", 2016)
    assert now[target(2)] == pytest.approx(later["log_anteil"] - now["log_anteil"])
    assert _row(feats, 8, "ZH", 2015)["anteilswachstum_vorjahr"] == pytest.approx(
        _row(feats, 8, "ZH", 2014)[target(1)]
    )


def test_levers_use_published_premiums_only(feats: pd.DataFrame) -> None:
    last = feats[feats["jahr"] == feats["jahr"].max()]
    assert last[target(1)].isna().all()
    # premiums are two years ahead of insured data -> both levers known for the last base year
    assert last[lever(1)].notna().all() and last[lever(2)].notna().all()


def test_relative_premium_is_centred_per_canton(feats: pd.DataFrame) -> None:
    for _, g in feats[feats["rel_praemie"].notna()].groupby(["kanton", "jahr"]):
        assert np.average(g["rel_praemie"], weights=g["versicherte"]) == pytest.approx(0, abs=1e-9)


def test_market_premium_for_years_without_insured_data() -> None:
    m = market_premiums(PRAEMIEN, BESTAND)
    assert m["jahr"].max() == LAST_PREMIUM_YEAR and m["markt_praemie"].notna().all()


def test_training_frame_filters(feats: pd.DataFrame) -> None:
    tf = training_frame(feats, 2, min_versicherte=0, max_abs_target_per_year=0.01)
    assert tf[target(2)].abs().le(0.02).all() and tf[lever(2)].notna().all()


def test_features_per_horizon() -> None:
    assert lever(2) in features_for(2) and lever(2) not in features_for(1)


@pytest.mark.parametrize("pp", [-5.0, 0.0, 2.5])
def test_pp_roundtrip(pp: float) -> None:
    assert lever_to_pp(pp_to_lever(pp)) == pytest.approx(pp)
