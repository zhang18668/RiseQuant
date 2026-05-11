"""F-005 PreTrendFeatures 测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature.pre_trend_features import PreTrendFeatures


def _build_uptrend_stock(n=60, base=10.0, daily_change=0.005):
    dates = pd.bdate_range("2023-01-02", periods=n)
    closes = base * (1 + daily_change) ** np.arange(n)
    df = pd.DataFrame({
        "date": dates,
        "code": "000001",
        "close": closes,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "volume": np.linspace(1_000_000, 800_000, n),
    })
    return df


class TestPreTrendFeatures:
    def test_returns_series_basic(self):
        df = _build_uptrend_stock()
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[40], df
        )
        # 上涨趋势 -> 1 个月、2 个月累计收益均 > 0
        assert feats["f_pt_ret_20"] > 0
        assert feats["f_pt_ret_40"] > 0

    def test_event_too_early_returns_nan(self):
        df = _build_uptrend_stock(n=30)
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[5], df
        )
        # 不足 1 个月 -> 全 NaN
        assert pd.isna(feats["f_pt_ret_20"])

    def test_ma_alignment_bullish_in_uptrend(self):
        df = _build_uptrend_stock()
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[40], df
        )
        assert feats["f_pt_ma_bullish"] == 1.0

    def test_consolidation_days_in_flat_stock(self):
        df = _build_uptrend_stock(daily_change=0.0)
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[40], df
        )
        # 横盘 -> 振幅<3% 的天数较多
        assert feats["f_pt_consol_days"] >= 10

    def test_volume_convergence_basic(self):
        df = _build_uptrend_stock()
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[40], df
        )
        # volume 递减 -> recent < full -> ratio < 1
        assert feats["f_pt_vol_conv"] < 1

    def test_support_resistance(self):
        df = _build_uptrend_stock()
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[40], df
        )
        # 上涨趋势 -> 接近 2m 高位
        assert feats["f_pt_pos_in_range_2m"] > 0.5

    def test_comprehensive_score_in_range(self):
        df = _build_uptrend_stock()
        feats = PreTrendFeatures(window_1m=20, window_2m=40).calculate_at_event(
            "000001", df["date"].iloc[40], df
        )
        assert 0.0 <= feats["f_pt_score_comp"] <= 1.0

    def test_to_series_dict_helpers(self):
        df = _build_uptrend_stock()
        ptf = PreTrendFeatures()
        feats = ptf.calculate_at_event("000001", df["date"].iloc[40], df)
        d = ptf.to_dict(feats)
        assert "f_pt_score_comp" in d
        s = ptf.to_series(feats)
        assert isinstance(s, pd.Series)
