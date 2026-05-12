"""E-001 LimitUpEventDetector 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.event.limit_up_detector import LimitUpEventDetector


class TestLimitUpDetector:
    def test_is_limit_up_true(self):
        det = LimitUpEventDetector()
        assert det.is_limit_up(9.95) is True

    def test_is_limit_up_false_below_threshold(self):
        det = LimitUpEventDetector()
        assert det.is_limit_up(9.8) is False

    def test_is_limit_up_nan(self):
        det = LimitUpEventDetector()
        assert det.is_limit_up(float("nan")) is False

    def test_st_excluded(self):
        det = LimitUpEventDetector(exclude_st=True)
        assert det.is_limit_up(10.0, is_st=True) is False

    def test_st_not_excluded_when_flag_off(self):
        det = LimitUpEventDetector(exclude_st=False)
        assert det.is_limit_up(10.0, is_st=True) is True

    def test_detect_returns_correct_columns(self, sample_daily_data):
        det = LimitUpEventDetector()
        events = det.detect(sample_daily_data)
        # 至少包含核心列, 允许额外的 limit_up_type / consecutive_n 等扩展列
        assert set(events.columns) >= {"date", "code", "change_pct", "is_limit_up"}

    def test_detect_filters_correctly(self, sample_daily_data):
        det = LimitUpEventDetector()
        events = det.detect(sample_daily_data)
        # 所有保留行都达到阈值
        assert (events["change_pct"] >= 9.9).all()
        # 000001 至少 2 个涨停 (首板+二板)
        assert (events["code"] == "000001").sum() >= 2

    def test_detect_empty(self):
        det = LimitUpEventDetector()
        events = det.detect(pd.DataFrame(columns=["date", "code", "change_pct"]))
        assert events.empty
        assert set(events.columns) >= {"date", "code", "change_pct", "is_limit_up"}

    def test_detect_excludes_st_rows(self, sample_daily_data):
        df = sample_daily_data.copy()
        df.loc[df["code"] == "000001", "is_st"] = True
        det = LimitUpEventDetector(exclude_st=True)
        events = det.detect(df)
        assert (events["code"] == "000001").sum() == 0

    def test_detect_missing_columns_raises(self):
        det = LimitUpEventDetector()
        with pytest.raises(Exception):
            det.detect(pd.DataFrame({"date": []}))

    def test_threshold_custom(self):
        det = LimitUpEventDetector(threshold=5.0)
        assert det.is_limit_up(5.0) is True
        assert det.is_limit_up(4.9) is False
