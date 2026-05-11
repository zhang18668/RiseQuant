"""F-002 涨停序列衍生特征

围绕涨停相关的衍生统计量：
- 历史 N 日内涨停次数；
- 当前已连板数；
- 上次涨停距今的交易日数；
- 连续上涨天数（不一定涨停）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class LimitUpFeatures:
    """涨停衍生特征计算器。"""

    def __init__(self, threshold: float = 9.9) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    def _is_limit_up_series(self, change_pct: pd.Series) -> pd.Series:
        return change_pct >= self.threshold

    # ------------------------------------------------------------------
    def limit_up_count(self, change_pct: pd.Series, window: int) -> pd.Series:
        """过去 ``window`` 日内的涨停次数（含当日）。"""
        is_lu = self._is_limit_up_series(change_pct).astype(int)
        return is_lu.rolling(window=window, min_periods=1).sum()

    def consecutive_limit_up(self, change_pct: pd.Series) -> pd.Series:
        """截至当前的连续涨停天数。"""
        is_lu = self._is_limit_up_series(change_pct).astype(int)
        groups = (is_lu == 0).cumsum()
        return is_lu.groupby(groups).cumsum()

    def consecutive_up_days(self, change_pct: pd.Series) -> pd.Series:
        """截至当前的连续上涨天数 (change_pct>0)。"""
        is_up = (change_pct > 0).astype(int)
        groups = (is_up == 0).cumsum()
        return is_up.groupby(groups).cumsum()

    def days_since_last_limit_up(self, change_pct: pd.Series) -> pd.Series:
        """距离上次涨停的交易日数；从未出现过则填 NaN。"""
        is_lu = self._is_limit_up_series(change_pct)
        last_idx = pd.Series(np.where(is_lu, np.arange(len(is_lu)), np.nan), index=is_lu.index)
        last_idx = last_idx.ffill()
        positions = pd.Series(np.arange(len(is_lu)), index=is_lu.index, dtype=float)
        return positions - last_idx

    # ------------------------------------------------------------------
    def calculate_all(self, df: pd.DataFrame, windows: Iterable[int] = (20, 60)) -> pd.DataFrame:
        """按单只股票（已按日期升序）一次计算所有特征。

        必须包含 ``change_pct`` 列。
        """
        if "change_pct" not in df.columns:
            raise ValueError("missing column: change_pct")
        out = df.copy()
        cp = out["change_pct"]
        for w in windows:
            out[f"f_lu_count_{w}"] = self.limit_up_count(cp, w)
        out["f_lu_consec"] = self.consecutive_limit_up(cp)
        out["f_up_consec"] = self.consecutive_up_days(cp)
        out["f_lu_days_since"] = self.days_since_last_limit_up(cp)
        return out
