"""F-006 FeatureCalculator 集成测试。"""
from __future__ import annotations

import pandas as pd

from src.feature.feature_calculator import FeatureCalculator


def test_calculate_smoke(sample_daily_data):
    fc = FeatureCalculator()
    events = pd.DataFrame({
        "code": ["000001"],
        "event_date": [sample_daily_data[sample_daily_data["code"] == "000001"]["date"].iloc[-2]],
    })
    out = fc.calculate(sample_daily_data, events)
    assert not out.empty
    assert "code" in out.columns
    assert "event_date" in out.columns


def test_empty_events_returns_empty(sample_daily_data):
    fc = FeatureCalculator()
    out = fc.calculate(sample_daily_data, pd.DataFrame(columns=["code", "event_date"]))
    assert out.empty


def test_calculate_at_event(stock_a_data):
    fc = FeatureCalculator()
    feats = fc.calculate_at_event("000001", stock_a_data["date"].iloc[-2], stock_a_data)
    assert isinstance(feats, pd.Series)
