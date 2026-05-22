from __future__ import annotations

import json

from src.utils.artifact_schema import (
    SCHEMA_VERSION,
    write_backtest_manifest,
    write_training_manifest,
)


def test_write_training_manifest(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "shared_meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training_summary.json").write_text("{}", encoding="utf-8")

    path = write_training_manifest(
        tmp_path,
        strategy="pattern_cluster",
        run_id="run_x",
        bundle_types=["per_cluster"],
        summary={"best_k": 3},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["artifact_type"] == "training_bundle"
    assert data["bundle_types"] == ["per_cluster"]
    assert data["artifacts"]["shared/shared_meta.json"] == "shared/shared_meta.json"
    assert data["summary"]["best_k"] == 3


def test_write_backtest_manifest(tmp_path):
    (tmp_path / "backtest_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    path = write_backtest_manifest(
        tmp_path,
        strategy="pattern_cluster",
        run_id="run_y",
        bundle_type="per_cluster",
        metrics={"total_return": 0.12},
        config={"topk": 5},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["artifact_type"] == "backtest_run"
    assert data["bundle_type"] == "per_cluster"
    assert data["artifacts"]["backtest_metrics.json"] == "backtest_metrics.json"
    assert data["metrics"]["total_return"] == 0.12
    assert data["config"]["topk"] == 5
