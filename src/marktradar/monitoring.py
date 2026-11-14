"""Live monitoring: how good was the model on years it has never seen?

The backtest estimates accuracy *in hindsight*. Monitoring measures it *for real*:
when the BAG publishes a new year of insured data, the current production model is
applied to exactly those new cases and compared with what actually happened, *before*
it is retrained. Results go to ``models/monitoring.json`` and appear in the web UI.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from marktradar.model import evaluate
from marktradar.registry import Models

MONITORING_FILE = "monitoring.json"


def live_evaluation(
    models: Models, metadata: dict[str, Any], frames: dict[int, pd.DataFrame]
) -> list[dict[str, Any]]:
    """Evaluate each production horizon on base years after its training data."""
    records = []
    for h, model in models.items():
        last_trained = int(metadata["horizonte"][str(h)]["trainiert_bis_basisjahr"])
        frame = frames[h]
        for year in sorted(y for y in frame["jahr"].unique() if y > last_trained):
            test = frame[frame["jahr"] == year]
            if len(test) < 20:
                continue
            metrics = evaluate(model.predict(test), test, h)
            records.append(
                {
                    "ausgewertet_am": datetime.now(UTC).date().isoformat(),
                    "modell_version": metadata["version"],
                    "horizont": h,
                    "basisjahr": int(year),
                    "prognosejahr": int(year) + h,
                    **{k: round(v, 4) for k, v in metrics.items()},
                }
            )
    return records


def load(root: Path) -> list[dict[str, Any]]:
    path = root / MONITORING_FILE
    if not path.exists():
        return []
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def append(root: Path, records: list[dict[str, Any]]) -> None:
    """Append new results; each (model version, horizon, base year) only once."""
    existing = load(root)

    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (r["modell_version"], r.get("horizont", 1), r["basisjahr"])

    seen = {key(r) for r in existing}
    existing += [r for r in records if key(r) not in seen]
    root.mkdir(parents=True, exist_ok=True)
    (root / MONITORING_FILE).write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
