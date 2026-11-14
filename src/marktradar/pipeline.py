"""The ML pipeline steps, shared by the CLI and the scheduled update.

download -> build -> train (backtest, calibrate, fit, quality gate) -> registry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from marktradar import data, registry
from marktradar.config import Settings
from marktradar.features import build_features, features_for, training_frame
from marktradar.model import backtest, conformal_margin, fit, summarize
from marktradar.registry import Models


def download(cfg: Settings) -> tuple[data.Manifest, bool]:
    """Sync raw files. Returns the new manifest and whether anything changed."""
    previous = data.Manifest.load(cfg.data.manifest_path)
    manifest = data.sync(
        cfg.data.catalog_url, cfg.data.names_url, cfg.data.raw_dir, previous, cfg.data.first_year
    )
    return manifest, manifest.fingerprint() != previous.fingerprint()


def build(cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    praemien, bestand = data.build_tables(cfg.data.raw_dir, cfg.data.first_year)
    data.validate(praemien, bestand)
    cfg.data.processed_dir.mkdir(parents=True, exist_ok=True)
    praemien.to_parquet(cfg.premiums_path, index=False)
    bestand.to_parquet(cfg.insured_path, index=False)
    names_file = cfg.data.raw_dir / data.NAMES_FILE
    names = (
        data.parse_names(names_file)
        if names_file.exists()
        else pd.DataFrame(columns=["bag_nr", "name"])
    )
    names.to_parquet(cfg.names_path, index=False)
    return praemien, bestand


def load_features(cfg: Settings) -> pd.DataFrame:
    return build_features(
        pd.read_parquet(cfg.premiums_path), pd.read_parquet(cfg.insured_path), cfg.horizons
    )


def load_names(cfg: Settings) -> pd.DataFrame:
    if cfg.names_path.exists():
        return pd.read_parquet(cfg.names_path)
    return pd.DataFrame({"bag_nr": pd.Series(dtype=int), "name": pd.Series(dtype=str)})


def training_frames(cfg: Settings, features: pd.DataFrame | None = None) -> dict[int, pd.DataFrame]:
    f = load_features(cfg) if features is None else features
    return {
        h: training_frame(f, h, cfg.features.min_versicherte, cfg.features.max_abs_target_per_year)
        for h in cfg.horizons
    }


@dataclass
class TrainOutcome:
    version: str
    promoted: bool
    summaries: dict[int, dict[str, float]]
    production_mae: dict[int, float | None]


def train(cfg: Settings, force: bool = False) -> TrainOutcome:
    with threadpool_limits(cfg.model.n_threads):
        return _train(cfg, force)


def _train(cfg: Settings, force: bool) -> TrainOutcome:
    frames = training_frames(cfg)
    coverage = cfg.model.quantiles[1] - cfg.model.quantiles[0]
    models: Models = {}
    horizons_meta: dict[str, Any] = {}
    summaries: dict[int, dict[str, float]] = {}
    for h, frame in frames.items():
        result = backtest(frame, cfg.model, h, cfg.backtest.first_test_year)
        margin = conformal_margin(result.scores, coverage)
        models[h] = fit(frame, cfg.model, h, conformal_margin=margin)
        summaries[h] = summarize(result.table)
        horizons_meta[str(h)] = {
            "trainiert_bis_basisjahr": int(frame["jahr"].max()),
            "n_trainingszeilen": len(frame),
            "features": features_for(h),
            "conformal_margin": round(margin, 5),
            "backtest": summaries[h],
            "backtest_pro_jahr": result.table.round(4).to_dict(orient="records"),
        }
    manifest = data.Manifest.load(cfg.data.manifest_path)
    metadata = {
        "horizonte": horizons_meta,
        "daten_fingerprint": manifest.fingerprint() if manifest.files else None,
        "daten_quelle": cfg.data.catalog_url,
        "konfiguration": cfg.model.model_dump(),
    }
    root = cfg.registry.path
    version_dir = registry.save(models, metadata, root)

    prod = registry.production_version(root)
    prod_meta = registry.load_metadata(root, prod) if prod else None
    prod_mae: dict[int, float | None] = {
        h: (
            prod_meta["horizonte"].get(str(h), {}).get("backtest", {}).get("mae_modell")
            if prod_meta and "horizonte" in prod_meta
            else None
        )
        for h in frames
    }
    promoted = force or all(
        registry.should_promote(
            summaries[h]["mae_modell"], prod_mae[h], cfg.registry.promotion_tolerance
        )
        for h in frames
    )
    if promoted:
        registry.promote(version_dir.name, root)
    return TrainOutcome(version_dir.name, promoted, summaries, prod_mae)
