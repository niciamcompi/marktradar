from pathlib import Path

import pandas as pd
import pytest

from marktradar import registry
from marktradar.config import ModelConfig
from marktradar.features import build_features, training_frame
from marktradar.model import fit

from .conftest import INSURED, PREMIUMS


def test_save_promote_load(tmp_path: Path) -> None:
    frame = training_frame(
        build_features(PREMIUMS, INSURED[INSURED["jahr"] <= 2019], [1]), 1, 100, 0.5
    )
    models = {1: fit(frame, ModelConfig(max_iter=10), 1)}
    version_dir = registry.save(models, {"horizonte": {}}, tmp_path)
    assert registry.production_version(tmp_path) is None
    registry.promote(version_dir.name, tmp_path)
    loaded, meta = registry.load_production(tmp_path)
    assert meta["version"] == version_dir.name
    pd.testing.assert_frame_equal(loaded[1].predict(frame), models[1].predict(frame))


def test_promote_unknown_version(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        registry.promote("missing", tmp_path)


def test_load_without_production(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        registry.load_production(tmp_path)


@pytest.mark.parametrize(
    ("candidate", "production", "expected"),
    [(0.05, None, True), (0.05, 0.05, True), (0.051, 0.05, True), (0.06, 0.05, False)],
)
def test_quality_gate(candidate: float, production: float | None, expected: bool) -> None:
    assert registry.should_promote(candidate, production, tolerance=0.02) is expected
