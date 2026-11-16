"""Command line: ``uv run marktradar --help``

Typical flow (= the ML pipeline):
    marktradar download   # fetch BAG files (archives only once)
    marktradar build      # parse + validate -> data/processed/*.parquet
    marktradar train      # backtest, final training, quality gate, registry
    marktradar predict 8
    marktradar serve
Or all of it, only when the BAG has published something new:
    marktradar update
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from threadpoolctl import threadpool_limits

from marktradar import monitoring, pipeline, registry
from marktradar.config import Settings, load_settings
from marktradar.model import backtest, summarize
from marktradar.service import ForecastService

app = typer.Typer(help="Marktradar: market share forecasts for Swiss health insurers.")
ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to config.yaml")]


def _settings(config: Path | None) -> Settings:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return load_settings(config)


def _echo_json(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False))


@app.command()
def download(config: ConfigOpt = None) -> None:
    """Download the BAG files and record their versions in the data manifest."""
    cfg = _settings(config)
    manifest, changed = pipeline.download(cfg)
    manifest.save(cfg.data.manifest_path)
    typer.echo(f"{len(manifest.files)} files, {'changed' if changed else 'unchanged'}.")


@app.command()
def build(config: ConfigOpt = None) -> None:
    """Parse and validate the raw files into tidy tables."""
    cfg = _settings(config)
    praemien, bestand = pipeline.build(cfg)
    typer.echo(
        f"Premiums {praemien['jahr'].min()}-{praemien['jahr'].max()}, "
        f"insured {bestand['jahr'].min()}-{bestand['jahr'].max()}, "
        f"{bestand['bag_nr'].nunique()} insurers."
    )


@app.command("backtest")
def backtest_cmd(config: ConfigOpt = None) -> None:
    """Run only the time-correct backtest and print it (nothing is saved)."""
    cfg = _settings(config)
    for h, frame in pipeline.training_frames(cfg).items():
        with threadpool_limits(cfg.model.n_threads):
            table = backtest(frame, cfg.model, h, cfg.backtest.first_test_year).table
        typer.echo(f"\nHorizon {h} year(s)")
        typer.echo(table.round(4).to_string(index=False))
        _echo_json(summarize(table))


def _report(outcome: pipeline.TrainOutcome) -> None:
    for h, s in outcome.summaries.items():
        typer.echo(
            f"Horizon {h}: error {s['mae_modell']} vs. baseline {s['mae_konstant']} "
            f"({s['verbesserung_vs_konstant']:.0%} better), rank corr. {s['rangkorrelation']}, "
            f"interval coverage {s['intervall_abdeckung']:.0%}"
        )
    if outcome.promoted:
        typer.echo(f"OK: version {outcome.version} is now in production.")
    else:
        typer.echo(
            f"REJECTED: version {outcome.version} saved but NOT promoted "
            f"(production error: {outcome.production_mae})."
        )


@app.command()
def train(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Skip the quality gate")] = False,
) -> None:
    """Backtest + final training + quality gate + registry."""
    outcome = pipeline.train(_settings(config), force)
    _report(outcome)
    if not outcome.promoted:
        raise typer.Exit(code=2)


@app.command()
def update(
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option(help="Retrain even without new data")] = False,
    report: Annotated[Path | None, typer.Option(help="Write a Markdown summary here")] = None,
) -> None:
    """Self-update: look for new BAG data -> measure live accuracy of the current model
    -> retrain -> quality gate -> record the new data version.

    Exit code 0 = nothing new or new model promoted, 2 = candidate rejected.
    """
    cfg = _settings(config)
    manifest, changed = pipeline.download(cfg)
    root = cfg.registry.path
    has_production = registry.production_version(root) is not None
    if not changed and has_production and not force:
        typer.echo("No new data. Production model unchanged.")
        if report:
            report.write_text("No new BAG data.\n", encoding="utf-8")
        return

    typer.echo("New data found, rebuilding.")
    pipeline.build(cfg)
    live: list[dict[str, Any]] = []
    if has_production:
        models, metadata = registry.load_production(root)
        if "horizonte" in metadata:
            live = monitoring.live_evaluation(models, metadata, pipeline.training_frames(cfg))
            monitoring.append(root, live)
        for r in live:
            typer.echo(
                f"Live accuracy of {r['modell_version']} for {r['prognosejahr']} "
                f"(horizon {r['horizont']}): error {r['mae_modell']} "
                f"vs. baseline {r['mae_konstant']}"
            )

    outcome = pipeline.train(cfg)
    _report(outcome)
    if outcome.promoted:
        manifest.save(cfg.data.manifest_path)
        typer.echo(f"Data manifest updated: {cfg.data.manifest_path}")
    if report:
        report.write_text(_markdown_report(outcome, live), encoding="utf-8")
    if not outcome.promoted:
        raise typer.Exit(code=2)


def _markdown_report(outcome: pipeline.TrainOutcome, live: list[dict[str, Any]]) -> str:
    lines = [
        "## New BAG data processed",
        "",
        "| Horizon | Error new model | Error production | Better than baseline | Rank corr. |",
        "|---|---|---|---|---|",
    ]
    for h, s in outcome.summaries.items():
        lines.append(
            f"| {h} year(s) | {s['mae_modell']} | {outcome.production_mae.get(h)} | "
            f"{s['verbesserung_vs_konstant']:.0%} | {s['rangkorrelation']} |"
        )
    lines += ["", f"**Quality gate:** {'passed' if outcome.promoted else 'NOT passed'}"]
    if live:
        lines += [
            "",
            "### Live accuracy of the previous model (years it never saw)",
            "",
            "| Forecast year | Horizon | Error model | Error baseline | Rank corr. |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['prognosejahr']} | {r['horizont']} | {r['mae_modell']} | "
            f"{r['mae_konstant']} | {r['rangkorrelation']} |"
            for r in live
        ]
    return "\n".join(lines) + "\n"


@app.command()
def compare(
    config: ConfigOpt = None,
    models: Annotated[
        str | None, typer.Option(help="Comma-separated model keys (default: all except slow ones)")
    ] = None,
    slow: Annotated[bool, typer.Option(help="Also run slow models (Chronos fine-tuning)")] = False,
    report: Annotated[Path, typer.Option(help="Markdown report")] = Path(
        "docs/model_comparison.md"
    ),
) -> None:
    """Compare models from simple baselines to transformers in the same backtest."""
    from marktradar import benchmark

    cfg = _settings(config)
    candidates = benchmark.select(models.split(",") if models else None, slow)
    missing = [c.name for c in candidates if c.heavy and not _benchmark_extra_installed()]
    if missing:
        typer.echo(f"Skipping {missing}: run `uv sync --extra benchmark` first.")
        candidates = [c for c in candidates if not c.heavy]
    ctx = benchmark.Context(
        pd.read_parquet(cfg.premiums_path), pd.read_parquet(cfg.insured_path), cfg
    )
    with threadpool_limits(cfg.model.n_threads):
        results = benchmark.run(
            ctx, pipeline.training_frames(cfg), candidates, cfg.backtest.first_test_year, typer.echo
        )
    summary = benchmark.summarize(results)
    benchmark.save(results, summary, cfg.registry.path / "benchmark.json", report)
    typer.echo(benchmark.markdown_report(summary))


def _benchmark_extra_installed() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) for m in ("torch", "chronos", "tabpfn", "lightgbm"))


@app.command()
def promote(version: str, config: ConfigOpt = None) -> None:
    """Put a specific version into production (e.g. a rollback)."""
    cfg = _settings(config)
    registry.promote(version, cfg.registry.path)
    typer.echo(f"Version {version} is now in production.")


@app.command()
def predict(
    bag_nr: int,
    pp: Annotated[
        float | None,
        typer.Option(help="Premium change vs. market in pp (default: approved tariffs)"),
    ] = None,
    horizont: Annotated[int | None, typer.Option(help="1 or 2 (default: latest year)")] = None,
    curve: Annotated[bool, typer.Option(help="Elasticity curve from -5 to +5 pp")] = False,
    config: ConfigOpt = None,
) -> None:
    """Forecast for one insurer (BAG number, e.g. 8 = CSS)."""
    cfg = _settings(config)
    models, _ = registry.load_production(cfg.registry.path)
    svc = ForecastService(models, pipeline.load_features(cfg), pipeline.load_names(cfg))
    if curve:
        rows = svc.elastizitaet(bag_nr, horizont, -5, 5, 1)
        cols = ["praemienanpassung_rel_pp", "anteilswachstum_pct", "bestand_prognose"]
        typer.echo(pd.DataFrame(rows)[cols].to_string(index=False))
    else:
        result = svc.prognose(bag_nr, horizont, pp)
        typer.echo(pd.DataFrame(result.pop("kantone")).head(10).to_string(index=False))
        _echo_json(result)


@app.command()
def serve(
    host: str = "0.0.0.0",  # noqa: S104 - intended inside the container
    port: int = 8000,
    config: ConfigOpt = None,
) -> None:
    """Start the web UI and REST API."""
    import uvicorn

    from marktradar.api import create_app

    uvicorn.run(create_app(_settings(config)), host=host, port=port)


if __name__ == "__main__":
    app()
