"""Artifact manifest helpers.

The project has several strategy scripts, but dashboards and API endpoints need
one stable contract for "what is in this run directory".  These helpers write a
small manifest next to existing artifacts without changing the older files.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

SCHEMA_VERSION = "artifact-manifest/v1"


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)
    return path


def existing_artifacts(run_dir: str | Path, relative_paths: Iterable[str]) -> Dict[str, str]:
    run_dir = Path(run_dir)
    out: Dict[str, str] = {}
    for rel in relative_paths:
        if (run_dir / rel).exists():
            out[rel] = rel
    return out


def write_training_manifest(
    run_dir: str | Path,
    *,
    strategy: str,
    run_id: str | None = None,
    bundle_types: Iterable[str] = (),
    summary: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "training_bundle",
        "strategy": strategy,
        "run_id": run_id or run_dir.name,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_types": list(bundle_types),
        "required_files": [
            "manifest.json",
            "shared/shared_meta.json",
            "training_summary.json",
        ],
        "artifacts": existing_artifacts(
            run_dir,
            [
                "shared/shared_meta.json",
                "shared/router.pkl",
                "shared/prototypes.pkl",
                "shared/cluster_stats.csv",
                "training_summary.json",
                "bundle_per_cluster/bundle_meta.json",
                "bundle_per_cluster/usability_flags.json",
                "bundle_single_with_cf/bundle_meta.json",
            ],
        ),
        "summary": dict(summary or {}),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return write_json(run_dir / "manifest.json", manifest)


def write_backtest_manifest(
    run_dir: str | Path,
    *,
    strategy: str,
    run_id: str | None = None,
    bundle_type: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "backtest_run",
        "strategy": strategy,
        "run_id": run_id or run_dir.name,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_type": bundle_type,
        "required_files": [
            "manifest.json",
            "backtest_metrics.json",
            "run_config.json",
            "summary.json",
        ],
        "artifacts": existing_artifacts(
            run_dir,
            [
                "backtest_metrics.json",
                "run_config.json",
                "summary.json",
                "signals.csv",
                "trades.csv",
                "equity_curve.csv",
            ],
        ),
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return write_json(run_dir / "manifest.json", manifest)
