"""Live-Monitoring: Wie gut war das Modell bei Jahren, die es nie gesehen hat?

Der Backtest schätzt die Güte *rückwirkend*. Das Monitoring misst sie *echt*:
Sobald das BAG ein neues Datenjahr publiziert, wird das bisherige Produktionsmodell
auf genau diese neuen Fälle angewendet und mit der Realität verglichen, *bevor*
es neu trainiert wird. Die Resultate landen in ``models/monitoring.json`` und sind
in der Oberfläche unter "Modellgüte" sichtbar.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from marktradar.model import MarktradarModel, evaluate

MONITORING_FILE = "monitoring.json"


def live_evaluation(
    model: MarktradarModel, metadata: dict[str, Any], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """Produktionsmodell auf allen Basisjahren auswerten, die nach seinem Training liegen."""
    last_trained = int(metadata["trainiert_auf_basisjahre"][1])
    records = []
    for year in sorted(y for y in frame["jahr"].unique() if y > last_trained):
        test = frame[frame["jahr"] == year]
        if len(test) < 5:
            continue
        metrics = evaluate(model.predict(test), test)
        records.append(
            {
                "ausgewertet_am": datetime.now(UTC).date().isoformat(),
                "modell_version": metadata["version"],
                "basisjahr": int(year),
                "prognosejahr": int(year) + 1,
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
    """Neue Auswertungen anhängen; je (Modellversion, Basisjahr) nur einmal."""
    existing = load(root)
    seen = {(r["modell_version"], r["basisjahr"]) for r in existing}
    existing += [r for r in records if (r["modell_version"], r["basisjahr"]) not in seen]
    root.mkdir(parents=True, exist_ok=True)
    (root / MONITORING_FILE).write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
