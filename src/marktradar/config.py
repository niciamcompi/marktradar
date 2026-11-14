"""Typed configuration loaded from ``config/config.yaml``.

Pydantic validates the file on load, so a typo or a wrong type fails immediately
instead of halfway through a training run.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    catalog_url: str
    names_url: str | None = None
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    manifest_path: Path = Path("config/data_manifest.json")
    first_year: int = 2011


class FeatureConfig(BaseModel):
    min_versicherte: float = 500
    max_abs_target_per_year: float = Field(0.5, gt=0)


class ModelConfig(BaseModel):
    learning_rate: float = Field(0.05, gt=0, le=1)
    max_iter: int = Field(300, gt=0)
    max_depth: int | None = 3
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    quantiles: tuple[float, float] = (0.1, 0.9)
    random_state: int = 42
    n_threads: int | None = 1


class BacktestConfig(BaseModel):
    first_test_year: int = 2016


class RegistryConfig(BaseModel):
    path: Path = Path("models")
    promotion_tolerance: float = Field(0.02, ge=0)


class Settings(BaseModel):
    data: DataConfig
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    horizons: list[int] = [1, 2]
    backtest: BacktestConfig = BacktestConfig()
    registry: RegistryConfig = RegistryConfig()

    def resolve_paths(self, base: Path) -> Settings:
        def fix(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        self.data.raw_dir = fix(self.data.raw_dir)
        self.data.processed_dir = fix(self.data.processed_dir)
        self.data.manifest_path = fix(self.data.manifest_path)
        self.registry.path = fix(self.registry.path)
        return self

    @property
    def premiums_path(self) -> Path:
        return self.data.processed_dir / "praemien.parquet"

    @property
    def insured_path(self) -> Path:
        return self.data.processed_dir / "bestand.parquet"

    @property
    def names_path(self) -> Path:
        return self.data.processed_dir / "versicherer.parquet"


def config_path_from(path: Path | str | None = None) -> Path:
    return Path(path or os.environ.get("MARKTRADAR_CONFIG") or DEFAULT_CONFIG)


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings. Precedence: argument > $MARKTRADAR_CONFIG > default file."""
    config_path = config_path_from(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(raw)
    return settings.resolve_paths(config_path.resolve().parent.parent)
