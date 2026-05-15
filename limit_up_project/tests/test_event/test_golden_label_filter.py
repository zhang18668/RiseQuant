"""单测: GoldenLabelFilter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.event.golden_label_filter import GoldenLabelFilter, OUTPUT_EXTRA_COLS


def _make_daily(code: str, dates, closes):
    return pd.DataFrame({
        "date": dates,
        "code": code,
        "close": list(closes),
    })


def test_empty_events_returns_empty_with_cols():
    daily = _make_daily("A", pd.bdate_range("2024-01-02", periods=30), np.linspace(10, 15, 30))
    events = pd.DataFrame(columns=["code", "second_date", "second_close"])
    out = GoldenLabelFilter().filter(events, daily)
    assert out.empty
    for c in OUTPUT_EXTRA_COLS:
        assert c in out.columns


def test_basic_golden_in_range():
    # second_date = dates[10], second_close = 10.0; 之后 22 日内涨幅 25%
    dates = pd.bdate_range("2024-01-02", periods=40)
    closes = [10.0] * 11
    for i in range(22):
        closes.append(10.0 + (i + 1) * (2.5 / 22))
    closes += [12.5] * (40 - len(closes))
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    assert len(out) == 1
    row = out.iloc[0]
    assert pytest.approx(row["future_max_pct"], abs=1e-6) == 0.25
    assert bool(row["is_golden"]) is True
    assert bool(row["has_full_horizon"]) is True


def test_out_of_range_low_not_golden():
    dates = pd.bdate_range("2024-01-02", periods=40)
    closes = [10.0] * 11 + [10.0 * 1.05] * 29
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    assert bool(out.iloc[0]["is_golden"]) is False


def test_out_of_range_high_not_golden():
    dates = pd.bdate_range("2024-01-02", periods=40)
    closes = [10.0] * 11 + [15.0] * 29
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    assert bool(out.iloc[0]["is_golden"]) is False
    assert pytest.approx(out.iloc[0]["future_max_pct"], abs=1e-6) == 0.50


def test_insufficient_horizon_marks_has_full_false_but_can_still_be_golden():
    dates = pd.bdate_range("2024-01-02", periods=15)
    closes = [10.0] * 11 + [12.0] * 4
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35, horizon=22).filter(events, daily)
    row = out.iloc[0]
    assert bool(row["has_full_horizon"]) is False
    assert bool(row["is_golden"]) is True
    assert pytest.approx(row["future_max_pct"], abs=1e-6) == 0.20


def test_second_date_at_end_returns_nan():
    dates = pd.bdate_range("2024-01-02", periods=10)
    daily = _make_daily("A", dates, [10.0] * 10)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[-1], "second_close": 10.0,
    }])
    out = GoldenLabelFilter().filter(events, daily)
    assert pd.isna(out.iloc[0]["future_max_pct"])
    assert bool(out.iloc[0]["is_golden"]) is False


def test_second_date_not_in_daily_returns_nan():
    dates = pd.bdate_range("2024-01-02", periods=10)
    daily = _make_daily("A", dates, [10.0] * 10)
    events = pd.DataFrame([{
        "code": "A",
        "second_date": pd.Timestamp("2099-01-01"),
        "second_close": 10.0,
    }])
    out = GoldenLabelFilter().filter(events, daily)
    assert pd.isna(out.iloc[0]["future_max_pct"])
    assert bool(out.iloc[0]["is_golden"]) is False


def test_unknown_code_returns_nan():
    dates = pd.bdate_range("2024-01-02", periods=10)
    daily = _make_daily("A", dates, [10.0] * 10)
    events = pd.DataFrame([{
        "code": "B", "second_date": dates[3], "second_close": 10.0,
    }])
    out = GoldenLabelFilter().filter(events, daily)
    assert pd.isna(out.iloc[0]["future_max_pct"])


def test_horizon_uses_max_close_not_last_close():
    # 未来先冲到 30% 再回落到 5%, future_max_pct 应该是 30%
    dates = pd.bdate_range("2024-01-02", periods=40)
    closes = [10.0] * 11
    closes += [10.5, 11.0, 12.0, 13.0, 12.0, 11.0, 10.5]
    closes += [10.5] * (40 - len(closes))
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    row = out.iloc[0]
    fmd = pd.Timestamp(row["future_max_date"])
    expected = dates[14]
    assert pytest.approx(row["future_max_pct"], abs=1e-6) == 0.30
    assert bool(row["is_golden"]) is True
    assert fmd == expected, f"got {fmd} expected {expected}"


def test_missing_required_column_raises():
    daily = _make_daily("A", pd.bdate_range("2024-01-02", periods=10), [10.0] * 10)
    events = pd.DataFrame([{"code": "A"}])
    with pytest.raises(ValueError):
        GoldenLabelFilter().filter(events, daily)


def test_preserves_input_columns():
    dates = pd.bdate_range("2024-01-02", periods=40)
    daily = _make_daily("A", dates, [10.0] * 11 + [12.0] * 29)
    events = pd.DataFrame([{
        "code": "A",
        "second_date": dates[10],
        "second_close": 10.0,
        "first_date": dates[5],
        "gap_days": 5,
        "stop_loss_price": 9.5,
    }])
    out = GoldenLabelFilter().filter(events, daily)
    for c in ["code", "second_date", "second_close", "first_date", "gap_days", "stop_loss_price"]:
        assert c in out.columns
    assert out.iloc[0]["gap_days"] == 5
