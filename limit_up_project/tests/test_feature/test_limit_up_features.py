"""F-002 LimitUpFeatures 测试。"""
from __future__ import annotations

import pandas as pd

from src.feature.limit_up_features import LimitUpFeatures


class TestLimitUpFeatures:
    def test_limit_up_count(self):
        cp = pd.Series([10.0, 5.0, 10.0, 0.0, 10.0])
        out = LimitUpFeatures().limit_up_count(cp, window=3)
        # at idx 4: last 3 = [10, 0, 10] -> 2
        assert out.iloc[-1] == 2

    def test_consecutive_limit_up(self):
        cp = pd.Series([10.0, 10.0, 5.0, 10.0])
        out = LimitUpFeatures().consecutive_limit_up(cp)
        assert out.tolist() == [1, 2, 0, 1]

    def test_consecutive_up_days(self):
        cp = pd.Series([1.0, 2.0, -1.0, 1.0, 1.0])
        out = LimitUpFeatures().consecutive_up_days(cp)
        assert out.tolist() == [1, 2, 0, 1, 2]

    def test_days_since_last_limit_up(self):
        cp = pd.Series([0.0, 10.0, 0.0, 0.0, 0.0])
        out = LimitUpFeatures().days_since_last_limit_up(cp)
        # 第一个无涨停 -> NaN, 第二个涨停日 -> 0, 之后 1,2,3
        assert pd.isna(out.iloc[0])
        assert out.iloc[1] == 0
        assert out.iloc[-1] == 3

    def test_calculate_all_smoke(self, stock_a_data):
        out = LimitUpFeatures().calculate_all(stock_a_data, windows=[5, 10])
        assert "f_lu_count_5" in out.columns
        assert "f_lu_consec" in out.columns
        assert "f_up_consec" in out.columns
