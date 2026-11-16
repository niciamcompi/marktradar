"""REST API and web UI (FastAPI).

Run locally:  uv run marktradar serve   -> UI at http://localhost:8000, API docs at /docs
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from marktradar import __version__, monitoring
from marktradar.config import Settings, load_settings
from marktradar.pipeline import load_features, load_names
from marktradar.registry import load_production
from marktradar.service import ForecastService, UnknownInsurerError

WEB_DIR = Path(__file__).parent / "web"
PP_LIMIT = 15.0  # scenarios beyond ±15 pp are far outside anything in the training data


class PrognoseRequest(BaseModel):
    bag_nr: int = Field(..., examples=[8])
    horizont: int | None = Field(
        None, description="1 or 2 years after the base year; default: latest"
    )
    praemienanpassung_rel_pp: float | None = Field(
        None,
        ge=-PP_LIMIT,
        le=PP_LIMIT,
        description="Premium change relative to the canton markets in percentage points "
        "(+2 = 2 pp more than the market). Empty = the actually approved tariffs.",
    )


class ElastizitaetRequest(BaseModel):
    bag_nr: int = Field(..., examples=[8])
    horizont: int | None = None
    von_pp: float = Field(-5.0, ge=-PP_LIMIT, le=PP_LIMIT)
    bis_pp: float = Field(5.0, ge=-PP_LIMIT, le=PP_LIMIT)
    schritt_pp: float = Field(1.0, gt=0.1, le=5)


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or load_settings()
        models, metadata = load_production(cfg.registry.path)
        app.state.service = ForecastService(models, load_features(cfg), load_names(cfg))
        app.state.metadata = metadata
        app.state.registry_path = cfg.registry.path
        yield

    app = FastAPI(
        title="Marktradar",
        version=__version__,
        description="Market share forecasts for Swiss health insurers (mandatory basic "
        "insurance), built on public BAG premium and insured-population data.",
        lifespan=lifespan,
    )

    def svc(request: Request) -> ForecastService:
        service: ForecastService = request.app.state.service
        return service

    def check_horizon(request: Request, h: int | None) -> None:
        if h is not None and h not in svc(request).horizons:
            raise HTTPException(422, f"Horizon must be one of {svc(request).horizons}")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "modell_version": request.app.state.metadata["version"]}

    @app.get("/modell")
    def modell(request: Request) -> dict[str, Any]:
        """Version, data, backtest metrics and live monitoring of the production model."""
        meta: dict[str, Any] = dict(request.app.state.metadata)
        meta["live_monitoring"] = monitoring.load(request.app.state.registry_path)
        meta["horizonte_verfuegbar"] = svc(request).horizons
        meta["basisjahr"] = svc(request).basisjahr
        return meta

    @app.get("/modellvergleich")
    def modellvergleich(request: Request) -> dict[str, Any]:
        """Results of `marktradar compare` (empty if it has not been run)."""
        path = request.app.state.registry_path / "benchmark.json"
        if not path.exists():
            return {"zusammenfassung": [], "pro_jahr": []}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    @app.get("/versicherer")
    def versicherer(request: Request) -> list[dict[str, Any]]:
        return svc(request).versicherer()

    @app.get("/kantone")
    def kantone(request: Request) -> list[dict[str, Any]]:
        return svc(request).kantone()

    @app.post("/prognose")
    def prognose(body: PrognoseRequest, request: Request) -> dict[str, Any]:
        check_horizon(request, body.horizont)
        try:
            return svc(request).prognose(body.bag_nr, body.horizont, body.praemienanpassung_rel_pp)
        except UnknownInsurerError:
            raise HTTPException(404, f"Unknown BAG number {body.bag_nr}") from None

    @app.post("/elastizitaet")
    def elastizitaet(body: ElastizitaetRequest, request: Request) -> list[dict[str, Any]]:
        check_horizon(request, body.horizont)
        if body.von_pp >= body.bis_pp:
            raise HTTPException(422, "von_pp must be smaller than bis_pp")
        try:
            return svc(request).elastizitaet(
                body.bag_nr, body.horizont, body.von_pp, body.bis_pp, body.schritt_pp
            )
        except UnknownInsurerError:
            raise HTTPException(404, f"Unknown BAG number {body.bag_nr}") from None

    @app.get("/markt")
    def markt(
        request: Request,
        horizont: int | None = None,
        kanton: str | None = Query(None, min_length=2, max_length=2),
    ) -> list[dict[str, Any]]:
        """All insurers, sorted by forecast market share growth (approved tariffs)."""
        check_horizon(request, horizont)
        return svc(request).markt(horizont, kanton.upper() if kanton else None)

    return app
