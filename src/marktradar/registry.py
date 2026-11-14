"""A minimal file-based model registry.

    models/
      20261114-093012/          <- one immutable version per training run
        model.joblib            <- {horizon: HorizonModel}
        metadata.json           <- metrics, data fingerprint, config, git commit
      production.json           <- points at the version the API serves
      monitoring.json           <- live accuracy of past production models

Deliberately simple (no MLflow server needed), but built on the same ideas: versioned
artifacts plus a pointer that only moves after a quality gate. Rollback = move the pointer.

Security: joblib/pickle executes code on load. Only load artifacts you created yourself.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from marktradar.model import HorizonModel

Models = dict[int, HorizonModel]
PRODUCTION_POINTER = "production.json"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save(models: Models, metadata: dict[str, Any], root: Path) -> Path:
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    target = root / version
    target.mkdir(parents=True, exist_ok=False)
    joblib.dump(models, target / "model.joblib")
    meta = {"version": version, "git_commit": _git_commit(), **metadata}
    (target / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return target


def promote(version: str, root: Path) -> None:
    if not (root / version / "model.joblib").exists():
        raise FileNotFoundError(f"Version {version} does not exist in {root}")
    (root / PRODUCTION_POINTER).write_text(
        json.dumps({"version": version}, indent=2), encoding="utf-8"
    )


def production_version(root: Path) -> str | None:
    pointer = root / PRODUCTION_POINTER
    if not pointer.exists():
        return None
    return str(json.loads(pointer.read_text(encoding="utf-8"))["version"])


def load_metadata(root: Path, version: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (root / version / "metadata.json").read_text(encoding="utf-8")
    )
    return data


def load_production(root: Path) -> tuple[Models, dict[str, Any]]:
    version = production_version(root)
    if version is None:
        raise FileNotFoundError(f"No production model in {root}. Run `marktradar train` first.")
    models: Models = joblib.load(root / version / "model.joblib")
    return models, load_metadata(root, version)


def should_promote(candidate_mae: float, production_mae: float | None, tolerance: float) -> bool:
    """Quality gate: the candidate may be at most ``tolerance`` (relative) worse."""
    if production_mae is None:
        return True
    return candidate_mae <= production_mae * (1 + tolerance)
