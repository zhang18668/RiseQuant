"""E-003 MainWaveDetector 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.event.main_wave_detector import MainWaveDetector


class TestMainWaveDetector:
    def _make_daily(self, closes):
        return pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=len(closes)),
            "code": "000001",
            "close": closes,
        })

    def test_main_wave_success(self):
        # 二板当天 close=10, 之后窗口内最高 12 -> 20% 涨幅
        closes = [10.0] * 5 + [12.0] * 5
        daily = self._make_daily([10.0] + closes)  # 11 天
        sb = pd.DataFrame({
            "code": ["000001"],
            "second_date": [daily["date"].iloc[0]],
        })
        det = MainWaveDetector(n_days=10, return_threshold=0.15)
        out = det.detect(sb, daily)
        assert len(out) == 1
        assert out.iloc[0]["is_main_wave"]
        assert out.iloc[0]["period_return"] == pytest.approx(0.2)

    def test_main_wave_fail(self):
        # 二板后窗口内最大涨幅仅 10%
        closes = [10.0, 10.5, 11.0, 10.8, 11.0]
        daily = self._make_daily(closes)
        sb = pd.DataFrame({
            "code": ["000001"],
            "second_date": [daily["date"].iloc[0]],
        })
        det = MainWaveDetector(n_days=4, return_threshold=0.15)
        out = det.detect(sb, daily)
        assert not bool(out.iloc[0]["is_main_wave"])

    def test_boundary_exact_threshold(self):
        closes = [10.0, 11.5]  # +15% 恰好
        daily = self._make_daily(closes)
        sb = pd.DataFrame({"code": ["000001"], "second_date": [daily["date"].iloc[0]]})
        det = MainWaveDetector(n_days=5, return_threshold=0.15)
        out = det.detect(sb, daily)
        assert bool(out.iloc[0]["is_main_wave"])

    def test_empty_input(self):
        det = MainWaveDetector()
        out = det.detect(
            pd.DataFrame(columns=["code", "second_date"]),
            pd.DataFrame(columns=["date", "code", "close"]),
        )
        assert out.empty

    def test_calc_period_return(self):
        daily = self._make_daily([10.0, 12.0, 11.0])
        det = MainWaveDetector(n_days=5)
        r = det.calc_period_return("000001", daily["date"].iloc[0], 5, daily)
        assert r == pytest.approx(0.2)
