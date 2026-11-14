"""Forecasts and scenarios, independent of API or CLI.

Both the API and the CLI only call this class, so there is exactly one place where
forecasts are computed, and it can be tested without a web server.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from marktradar.features import lever, lever_to_pp, pp_to_lever
from marktradar.model import national
from marktradar.registry import Models


class UnknownInsurerError(KeyError):
    pass


def _pct(log_change: float) -> float:
    return round(float(np.expm1(log_change) * 100), 2)


class ForecastService:
    def __init__(self, models: Models, features: pd.DataFrame, names: pd.DataFrame) -> None:
        self.models = models
        self.basisjahr = int(features["jahr"].max())
        latest = features[features["jahr"] == self.basisjahr]
        latest = latest.merge(names, on="bag_nr", how="left")
        latest["name"] = latest["name"].fillna("BAG-Nr. " + latest["bag_nr"].astype(str))
        self.latest = latest.reset_index(drop=True)
        self.total = float(self.latest["versicherte"].sum())

    @property
    def horizons(self) -> list[int]:
        return sorted(self.models)

    def _active(self, h: int) -> pd.DataFrame:
        """Canton rows of insurers that still offer premiums in every forecast year."""
        levers = [lever(k) for k in range(1, h + 1)]
        return self.latest.dropna(subset=levers)

    def _predict(self, rows: pd.DataFrame, h: int, pp: float | None) -> pd.DataFrame:
        X = rows.copy()
        if pp is not None:
            X[lever(h)] = pp_to_lever(pp)
        return self.models[h].predict(X)

    def _weighted_lever_pp(self, rows: pd.DataFrame, h: int) -> float:
        return round(lever_to_pp(float(np.average(rows[lever(h)], weights=rows["versicherte"]))), 2)

    def _rows(self, bag_nr: int, h: int) -> pd.DataFrame:
        rows = self._active(h)
        rows = rows[rows["bag_nr"] == bag_nr]
        if rows.empty:
            raise UnknownInsurerError(bag_nr)
        return rows

    def versicherer(self) -> list[dict[str, Any]]:
        h = max(self.horizons)
        active = self._active(h)
        out = []
        for bag_nr, g in active.groupby("bag_nr"):
            out.append(
                {
                    "bag_nr": int(str(bag_nr)),
                    "name": str(g["name"].iloc[0]),
                    "versicherte": int(g["versicherte"].sum()),
                    "marktanteil_pct": round(float(g["versicherte"].sum() / self.total * 100), 2),
                    "kantone": int(len(g)),
                    "praemienanpassung_tatsaechlich_pp": self._weighted_lever_pp(g, h),
                }
            )
        return sorted(out, key=lambda r: r["versicherte"], reverse=True)

    def kantone(self) -> list[dict[str, Any]]:
        g = self.latest.groupby("kanton")["versicherte"].sum().sort_values(ascending=False)
        return [{"kanton": str(k), "versicherte": int(v)} for k, v in g.items()]

    def prognose(
        self, bag_nr: int, horizont: int | None = None, pp: float | None = None
    ) -> dict[str, Any]:
        """National forecast plus canton breakdown for one insurer.

        ``pp``: premium change vs. the canton markets in the final forecast year, in
        percentage points. ``None`` = the actually approved tariffs.
        Assumption: the size of each canton market stays constant.
        """
        h = horizont or max(self.horizons)
        rows = self._rows(bag_nr, h)
        pred = self._predict(rows, h, pp)
        n = national(pred, rows).iloc[0]
        bestand = float(rows["versicherte"].sum())
        kantone = []
        for (_, r), (_, p) in zip(rows.iterrows(), pred.iterrows(), strict=True):
            kantone.append(
                {
                    "kanton": r["kanton"],
                    "versicherte": int(r["versicherte"]),
                    "marktanteil_pct": round(float(np.exp(r["log_anteil"]) * 100), 2),
                    "praemienanpassung_pp": round(lever_to_pp(r[lever(h)]), 2),
                    "anteilswachstum_pct": _pct(p["prognose"]),
                    "untergrenze_pct": _pct(p["untergrenze"]),
                    "obergrenze_pct": _pct(p["obergrenze"]),
                    "bestand_veraenderung": round(
                        float(r["versicherte"] * np.expm1(p["prognose"]))
                    ),
                }
            )
        return {
            "bag_nr": int(bag_nr),
            "name": str(rows["name"].iloc[0]),
            "horizont": h,
            "basisjahr": self.basisjahr,
            "prognosejahr": self.basisjahr + h,
            "praemienanpassung_rel_pp": pp,
            "praemienanpassung_tatsaechlich_pp": self._weighted_lever_pp(rows, h),
            "marktanteil_basis_pct": round(bestand / self.total * 100, 3),
            "anteilswachstum_pct": _pct(n["prognose"]),
            "anteilswachstum_untergrenze_pct": _pct(n["untergrenze"]),
            "anteilswachstum_obergrenze_pct": _pct(n["obergrenze"]),
            "bestand_basis": round(bestand),
            "bestand_prognose": round(bestand * float(np.exp(n["prognose"]))),
            "kantone": sorted(kantone, key=lambda k: k["versicherte"], reverse=True),
        }

    def elastizitaet(
        self, bag_nr: int, horizont: int | None, von_pp: float, bis_pp: float, schritt_pp: float
    ) -> list[dict[str, Any]]:
        """Premium elasticity curve: national forecast over a range of premium scenarios."""
        steps = np.arange(von_pp, bis_pp + schritt_pp / 2, schritt_pp)
        out = []
        for s in steps:
            res = self.prognose(bag_nr, horizont, round(float(s), 4))
            res.pop("kantone")
            out.append(res)
        return out

    def markt(self, horizont: int | None = None, kanton: str | None = None) -> list[dict[str, Any]]:
        """All insurers (nationally or in one canton), sorted by expected share growth,
        based on the actually approved tariffs."""
        h = horizont or max(self.horizons)
        rows = self._active(h)
        if kanton:
            rows = rows[rows["kanton"] == kanton]
        pred = self._predict(rows, h, None)
        n = national(pred, rows)
        total = float(rows["versicherte"].sum())
        names = rows.groupby("bag_nr")["name"].first()
        levers = rows.groupby("bag_nr")[[lever(h), "versicherte"]].apply(
            lambda g: self._weighted_lever_pp(g, h)
        )
        out = [
            {
                "bag_nr": int(str(b)),
                "name": str(names.to_dict()[b]),
                "basisjahr": self.basisjahr,
                "prognosejahr": self.basisjahr + h,
                "marktanteil_basis_pct": round(float(r["versicherte"] / total * 100), 2),
                "praemienanpassung_tatsaechlich_pp": float(levers.to_dict()[b]),
                "anteilswachstum_pct": _pct(r["prognose"]),
                "anteilswachstum_untergrenze_pct": _pct(r["untergrenze"]),
                "anteilswachstum_obergrenze_pct": _pct(r["obergrenze"]),
                "bestand_basis": round(float(r["versicherte"])),
                "bestand_prognose": round(float(r["versicherte"] * np.exp(r["prognose"]))),
            }
            for b, r in n.iterrows()
        ]
        return sorted(out, key=lambda r: r["anteilswachstum_pct"], reverse=True)
