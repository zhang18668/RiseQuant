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
    flt = GoldenLabelFilter()
    out = flt.filter(events, daily)
    assert out.empty
    for c in OUTPUT_EXTRA_COLS:
        assert c in out.columns


def test_basic_golden_in_range():
    # second_date 之后 22 天 close 涨到 25% -> is_golden=True
    dates = pd.bdate_range("2024-01-02", periods=40)
    # 第 10 天是 second_date, close=10; 后面缓慢涨到 12.5 (25%)
    closes = [10.0] * 10
    for i in range(22):
        closes.append(10.0 + (i + 1) * (2.5 / 22))   # 第 i+1 天 close 线性升到 12.5
    closes += [12.5] * (40 - len(closes))
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    assert len(out) == 1
    row = out.iloc[0]
    assert pytest.approx(row["future_max_pct"], abs=1e-6) == 0.25
    assert row["is_golden"] is True or row["is_golden"] == True  # noqa: E712
    assert row["has_full_horizon"] is True or row["has_full_horizon"] == True  # noqa: E712


def test_out_of_range_low_not_golden():
    # 未来涨幅只有 5%
    dates = pd.bdate_range("2024-01-02", periods=40)
    closes = [10.0] * 10 + [10.0 * 1.05] * 30
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    assert out.iloc[0]["is_golden"] == False  # noqa: E712


def test_out_of_range_high_not_golden():
    # 未来直接涨 50%, 超过 hi
    dates = pd.bdate_range("2024-01-02", periods=40)
    closes = [10.0] * 10 + [15.0] * 30
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    assert out.iloc[0]["is_golden"] == False  # noqa: E712
    assert pytest.approx(out.iloc[0]["future_max_pct"], abs=1e-6) == 0.50


def test_insufficient_horizon_marks_has_full_false_but_can_still_be_golden():
    # second_date 之后只剩 5 天就涨到 20%
    dates = pd.bdate_range("2024-01-02", periods=15)
    closes = [10.0] * 10 + [12.0] * 5
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35, horizon=22).filter(events, daily)
    row = out.iloc[0]
    assert row["has_full_horizon"] == False  # noqa: E712
    assert row["is_golden"] == True  # noqa: E712
    assert pytest.approx(row["future_max_pct"], abs=1e-6) == 0.20


def test_second_date_at_end_returns_nan():
    # second_date 是最后一天, 之后没有数据
    dates = pd.bdate_range("2024-01-02", periods=10)
    closes = [10.0] * 10
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[-1], "second_close": 10.0,
    }])
    out = GoldenLabelFilter().filter(events, daily)
    assert pd.isna(out.iloc[0]["future_max_pct"])
    assert out.iloc[0]["is_golden"] == False  # noqa: E712


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
    assert out.iloc[0]["is_golden"] == False  # noqa: E712


def test_unknown_code_returns_nan():
    dates = pd.bdate_range("2024-01-02", periods=10)
    daily = _make_daily("A", dates, [10.0] * 10)
    events = pd.DataFrame([{
        "code": "B",
        "second_date": dates[3],
        "second_close": 10.0,
    }])
    out = GoldenLabelFilter().filter(events, daily)
    assert pd.isna(out.iloc[0]["future_max_pct"])


def test_horizon_uses_max_close_not_last_close():
    # second 之后 close 先冲到 1.30 然后回落到 1.05 -> future_max_pct=0.30
    dates = pd.bdate_range("2024-01-02", periods=40)
    # 注意: closes[10] 是 second_date 当天的 close, 必须等于 second_close=10.0
    closes = [10.0] * 11    # 含 dates[10] = second_date 当天的 close
    after_pattern = [1.05, 1.10, 1.20, 1.30, 1.20, 1.10, 1.05]
    after = [10.0 * v for v in after_pattern]
    closes += after
    # pad to 40
    closes += [after[-1]] * (40 - len(closes))
    daily = _make_daily("A", dates, closes)
    events = pd.DataFrame([{
        "code": "A", "second_date": dates[10], "second_close": 10.0,
    }])
    out = GoldenLabelFilter(lo=0.15, hi=0.35).filter(events, daily)
    row = out.iloc[0]
    print("DEBUG row:", dict(row))
    print("DEBUG dates[14]:", dates[14])
    print("DEBUG dates[13]:", dates[13])
    assert pytest.approx(row["future_max_pct"], abs=1e-6) == 0.30
    assert row["is_golden"] == True  # noqa: E712
    # after[3] = 1.30 在 dates[10+1+3] = dates[14]
    fmd = pd.Timestamp(row["future_max_date"])
    d14 = dates[14]
    assert fmd == d14, f"{fmd} != {d14}"


def test_missing_required_column_raises():
    daily = _make_daily("A", pd.bdate_range("2024-01-02", periods=10), [10.0] * 10)
    events = pd.DataFrame([{"code": "A"}])  # 缺 second_date / second_close
    with pytest.raises(ValueError):
        GoldenLabelFilter().filter(events, daily)


def test_preserves_input_columns():
    # 原 events 的列不应丢失
    dates = pd.bdate_range("2024-01-02", periods=40)
    daily = _make_daily("A", dates, [10.0] * 10 + [12.0] * 30)
    events = pd.DataFrame([{
        "code": "A",
   