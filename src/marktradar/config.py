"""Typisierte Konfiguration aus ``config/config.yaml``.

Pydantic validiert die YAML-Datei beim Laden: Tippfehler oder falsche Typen fallen
sofort auf und nicht erst mitten im Training.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    discovery_page: str | None = None
    source_url: str
    sha256: str | None = None
    raw_path: Path
    processed_path: Path
    first_year: int = 2006


class FeatureConfig(BaseModel):
    min_versicherte: float = 1000
    max_abs_target: float = Field(0.5, gt=0)


class ModelConfig(BaseModel):
    learning_rate: float = Field(0.05, gt=0, le=1)
    max_iter: int = Field(300, gt=0)
    max_depth: int | None = 3
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    quantiles: tuple[float, float] = (0.1, 0.9)
    random_state: int = 42


class BacktestConfig(BaseModel):
    first_test_year: int = 2014


class RegistryConfig(BaseModel):
    path: Path = Path("models")
    promotion_tolerance: float = Field(0.02, ge=0)


class Settings(BaseModel):
    data: DataConfig
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    backtest: BacktestConfig = BacktestConfig()
    registry: RegistryConfig = RegistryConfig()

    def resolve_paths(self, base: Path) -> Settings:
        """Relative Pfade gegen ``base`` auflösen (Projektordner)."""

        def fix(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        self.data.raw_path = fix(self.data.raw_path)
        self.data.processed_path = fix(self.data.processed_path)
        self.registry.path = fix(self.registry.path)
        return self


def config_path_from(path: Path | str | None = None) -> Path:
    return Path(path or os.environ.get("MARKTRADAR_CONFIG") or DEFAULT_CONFIG)


def update_data_source(config_path: Path, url: str, sha256: str) -> None:
    """URL und Prüfsumme in config.yaml ersetzen, ohne Kommentare zu verlieren."""
    text = config_path.read_text(encoding="utf-8")
    text, n_url = re.subn(r"(?m)^(\s*source_url:\s*).*$", rf'\g<1>"{url}"', text, count=1)
    text, n_sha = re.subn(r"(?m)^(\s*sha256:\s*).*$", rf'\g<1>"{sha256}"', text, count=1)
    if n_url != 1 or n_sha != 1:
        raise ValueError(f"source_url/sha256 nicht in {config_path} gefunden")
    config_path.write_text(text, encoding="utf-8")


def load_settings(path: Path | str | None = None) -> Settings:
    """Konfiguration laden. Reihenfolge: Argument > $MARKTRADAR_CONFIG > Standard."""
    config_path = config_path_from(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(raw)
    return settings.resolve_paths(config_path.resolve().parent.parent)
