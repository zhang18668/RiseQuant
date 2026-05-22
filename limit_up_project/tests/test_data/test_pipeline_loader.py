from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import pipeline_loader


class StubConfig:
    def __init__(self, data_section: dict):
        self._data_section = data_section

    def get_section(self, name: str) -> dict:
        if name == "data":
            return self._data_section
        return {}


def make_daily(code: str = "600000") -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "date": dates,
            "code": str(code).zfill(6),
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "turnover": [10000, 11000, 12000, 13000, 14000],
            "change_pct": [0, 1, 1, 1, 1],
        }
    )


def make_config(tmp_path: Path) -> StubConfig:
    return StubConfig(
        {
            "start_date": "2024-01-02",
            "end_date": "2024-01-08",
            "daily_cache_dir": str(tmp_path / "daily_cache"),
            "zt_pool_cache_dir": str(tmp_path / "zt_pool_cache"),
            "tdx_vipdoc": str(tmp_path / "missing_tdx"),
        }
    )


def test_load_daily_data_cache_reads_existing_parquet(tmp_path: Path):
    pytest.importorskip("pyarrow")
    daily_cache = tmp_path / "daily_cache"
    daily_cache.mkdir()
    make_daily("600000").to_parquet(daily_cache / "600000.parquet", index=False)

    data = pipeline_loader.load_daily_data(make_config(tmp_path), source="cache")

    assert len(data) == 5
    assert data["code"].unique().tolist() == ["600000"]
    assert data["date"].min() == pd.Timestamp("2024-01-02")
    assert data["date"].max() == pd.Timestamp("2024-01-08")


def test_load_daily_data_cache_fails_with_guidance_when_cache_missing(tmp_path: Path):
    with pytest.raises(RuntimeError, match="build_daily_cache.py"):
        pipeline_loader.load_daily_data(make_config(tmp_path), source="cache")


def test_load_daily_data_auto_falls_back_to_tdx(monkeypatch, tmp_path: Path):
    expected = make_daily("000001")

    def fake_load_from_tdx(tdx_path, start_date, end_date):
        assert start_date == "2024-01-02"
        assert end_date == "2024-01-08"
        return expected

    monkeypatch.setattr(pipeline_loader, "_load_from_tdx", fake_load_from_tdx)

    data = pipeline_loader.load_daily_data(make_config(tmp_path), source="auto")

    assert data.equals(expected)
