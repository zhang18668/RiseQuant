"""F-001 PriceVolumeFactors 测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature.price_volume import PriceVolumeFactors


class TestPriceVolumeFactors:
    def test_returns_1d(self):
        s = pd.Series([100.0, 105.0, 110.0])
        out = PriceVolumeFactors.returns(s, 1)
        assert out.iloc[1] == pytest.approx(0.05)
        assert out.iloc[2] == pytest.approx(110.0 / 105.0 - 1)

    def test_returns_negative(self):
        s = pd.Series([100.0, 90.0])
        out = PriceVolumeFactors.returns(s, 1)
        assert out.iloc[1] == pytest.approx(-0.1)

    def test_returns_n(self):
        s = pd.Series([100, 105, 110, 115, 120], dtype=float)
        out = PriceVolumeFactors.returns(s, 3)
        assert out.iloc[3] == pytest.approx(115 / 100 - 1)

    def test_volume_ratio(self):
        # vol [100,200,400,200,100,500]; ratio at idx 5 = 500 / mean([100,200,400,200,100]) = 500/200 = 2.5
        v = pd.Series([100, 200, 400, 200, 100, 500], dtype=float)
        out = PriceVolumeFactors.volume_ratio(v, n=5)
        assert out.iloc[5] == pytest.approx(2.5)

    def test_volatility_positive(self):
        s = pd.Series([100.0, 102.0, 98.0, 101.0, 99.0, 103.0])
        out = PriceVolumeFactors.volatility(s, n=3)
        assert out.iloc[-1] > 0

    def test_ma(self):
        s = pd.Series([10, 12, 14, 16, 18], dtype=float)
        out = PriceVolumeFactors.ma(s, 3)
        assert out.iloc[-1] == pytest.approx((14 + 16 + 18) / 3)

    def test_ma_position(self):
        s = pd.Series([10, 10, 10, 10, 11], dtype=float)
        out = PriceVolumeFactors.ma_position(s, 5)
        # mean = (10+10+10+10+11)/5 = 10.2; ratio - 1 = 11/10.2 - 1
        assert out.iloc[-1] == pytest.approx(11 / 10.2 - 1)

    def test_amplitude_single_day(self):
        h = pd.Series([11.0, 12.0])
        l = pd.Series([9.0, 10.0])
        amp = PriceVolumeFactors.amplitude(h, l, n=1)
        assert amp.iloc[0] == pytest.approx((11 - 9) / 9)

    def test_calculate_all(self, stock_a_data):
        out = PriceVolumeFactors.calculate_all(
            stock_a_data,
            returns_n=[1, 5],
            volume_n=[5],
            ma_n=[5],
            volatility_n=[5],
        )
        assert "f_ret_1" in out.columns
        assert "f_ret_5" in out.columns
        assert "f_vol_ratio_5" in out.columns
        assert "f_ma_pos_5" in out.columns
        assert "f_vol_5" in out.columns
        assert "f_amplitude" in out.columns

    def test_calculate_all_missing_column(self):
        df = pd.DataFrame({"close": [1.0]})
        with pytest.raises(ValueError):
            PriceVolumeFactors.calculate_all(df)
