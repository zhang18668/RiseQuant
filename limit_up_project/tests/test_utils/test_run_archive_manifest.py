from __future__ import annotations

import json

import pandas as pd

from src.utils.run_archive import RunArchive


class FakeBacktester:
    equity_curve = pd.DataFrame({"date": ["2024-01-02"], "equity": [1.0]})

    def get_metrics(self):
        return {"total_return": 0.1}

    def get_trades(self):
        return pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "code": ["600000"],
                "action": ["SELL_TIMEOUT"],
                "return_pct": [0.03],
            }
        )


def test_run_archive_write_summary_writes_manifest(tmp_path):
    archive = RunArchive(base_dir=str(tmp_path), strategy="wash_second", run_id="t1")
    archive.save_config({"x": 1})
    archive.save_backtest(FakeBacktester())
    archive.write_summary()

    manifest = archive.run_dir / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema_version"] == "artifact-manifest/v1"
    assert data["artifact_type"] == "backtest_run"
    assert data["strategy"] == "wash_second"
    assert data["metrics"]["total_return"] == 0.1
    assert data["artifacts"]["backtest_metrics.json"] == "backtest_metrics.json"
