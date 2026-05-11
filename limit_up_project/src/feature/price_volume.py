"""F-001 基础量价因子

所有函数都是 *无副作用* 的纯静态方法，输入 ``pd.Series``，输出 ``pd.Series``。
这是为了让上层的特征计算器能够在不同的股票上批量应用。
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


class PriceVolumeFactors:
    """量价因子计算器。"""

    # ------------------------------------------------------------------
    # 单因子
    # ------------------------------------------------------------------
    @staticmethod
    def returns(close: pd.Series, n: int = 1) -> pd.Series:
        """N 日简单收益率。"""
        return close.pct_change(periods=n, fill_method=None)

    @staticmethod
    def log_returns(close: pd.Series, n: int = 1) -> pd.Series:
        return np.log(close / close.shift(n))

    @staticmethod
    def volume_ratio(volume: pd.Series, n: int = 5) -> pd.Series:
        """量比 = 当日成交量 / 前 N 日 *不含当日* 的均量。"""
        ref = volume.shift(1).rolling(window=n, min_periods=1).mean()
        return volume / ref

    @staticmethod
    def volatility(close: pd.Series, n: int = 5) -> pd.Series:
        """N 日收益率的样本标准差。"""
        return close.pct_change(fill_method=None).rolling(window=n).std()

    @staticmethod
    def ma(close: pd.Series, n: int) -> pd.Series:
        """N 日简单移动均线。"""
        return close.rolling(window=n).mean()

    @staticmethod
    def ma_position(close: pd.Series, n: int) -> pd.Series:
        """收盘价相对 N 日均线的位置 (close/ma - 1)。"""
        return close / close.rolling(window=n).mean() - 1.0

    @staticmethod
    def amplitude(high: pd.Series, low: pd.Series, n: int = 1) -> pd.Series:
        """N 日振幅 = (rolling_max(high) - rolling_min(low)) / rolling_min(low)。

        ``n=1`` 退化为当日振幅 (high-low)/low。
        """
        h = high.rolling(window=n, min_periods=1).max()
        l = low.rolling(window=n, min_periods=1).min()
        return (h - l) / l

    @staticmethod
    def turnover_ratio(volume: pd.Series, float_shares: float) -> pd.Series:
        """换手率 = volume / float_shares (此处保留接口，由上层填充流通股本)。"""
        if float_shares <= 0:
            return pd.Series(np.nan, index=volume.index)
        return volume / float_shares

    # ------------------------------------------------------------------
    # 组合特征
    # ------------------------------------------------------------------
    @classmethod
    def calculate_all(
        cls,
        df: pd.DataFrame,
        returns_n: Iterable[int] = (1, 3, 5, 10, 20),
        volume_n: Iterable[int] = (5, 10),
        ma_n: Iterable[int] = (5, 10, 20),
        volatility_n: Iterable[int] = (5, 10, 20),
    ) -> pd.DataFrame:
        """对单只股票 (date 升序排列) 计算一组量价因子。

        Parameters
        ----------
        df : DataFrame
            列 ``close, high, low, volume`` 至少需要。

        Returns
        -------
        DataFrame
            原始列 + 新增的 ``f_*`` 因子列。
        """
        required: List[str] = ["close", "high", "low", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")

        out = df.copy()
        close = out["close"]
        for n in returns_n:
            out[f"f_ret_{n}"] = cls.returns(close, n)
        for n in volume_n:
            out[f"f_vol_ratio_{n}"] = cls.volume_ratio(out["volume"], n)
        for n in ma_n:
            out[f"f_ma_pos_{n}"] = cls.ma_position(close, n)
        for n in volatility_n:
            out[f"f_vol_{n}"] = cls.volatility(close, n)
        out["f_amplitude"] = cls.amplitude(out["high"], out["low"], n=1)
        return out
