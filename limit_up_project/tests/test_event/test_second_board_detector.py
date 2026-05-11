"""E-002 SecondBoardDetector 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.event.limit_up_detector import LimitUpEventDetector
from src.event.second_board_detector import SecondBoardDetector


class TestSecondBoardDetector:
    def test_within_window(self, sample_daily_data):
        lu_det = LimitUpEventDetector()
        events = lu_det.detect(sample_daily_data)
        det = SecondBoardDetector(n_days=5)
        out = det.detect(events, sample_daily_data)
        assert not out.empty
        # 000001 在样本中有 1 对 (首板+二板)
        a_rows = out[out["code"] == "000001"]
        assert len(a_rows) >= 1
        assert a_rows.iloc[0]["gap_days"] == 2

    def test_outside_window(self):
        # 一个涨停 + 一个 8 天后涨停 -> n_days=5 时不算二板
        dates = pd.bdate_range("2023-01-02", periods=10)
        events = pd.DataFrame({
            "date": [dates[0], dates[8]],
            "code": ["000001", "000001"],
        })
        daily = pd.DataFrame({
            "date": dates,
            "code": "000001",
            "close": 10.0,
        })
        det = SecondBoardDetector(n_days=5)
        out = det.detect(events, daily)
        assert out.empty

    def test_empty_input(self):
        det = SecondBoardDetector()
        out = det.detect(
            pd.DataFrame(columns=["date", "code"]),
            pd.DataFrame(columns=["date", "code"]),
        )
        assert out.empty
        assert set(out.columns) == {"code", "first_date", "second_date", "gap_days"}

    def test_single_event_no_second(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        events = pd.DataFrame({"date": [dates[0]], "code": ["000001"]})
        daily = pd.DataFrame({"date": dates, "code": "000001", "close": 10.0})
        det = SecondBoardDetector(n_days=5)
        assert det.detect(events, daily).empty
