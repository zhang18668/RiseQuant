"""F-004 资金流特征

本期未接入主力/融资融券真实数据，使用近似指标作为占位：
- 净金额 = 当日成交额 * sign(change_pct)（简化代理）；
- 累计净金额：N 日滑窗。

接口给上层使用，真实数据接入后只需在此处替换实现。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class MoneyFlowFeatures:
    """资金流特征计算器。"""

    # ------------------------------------------------------------------
    def net_amount(self, change_pct: pd.Series, turnover: pd.Series) -> pd.Series:
        """以 change_pct 的方向作为资金流方向的近似。"""
        direction = np.sign(change_pct.fillna(0))
        return direction * turnover.fillna(0.0)

    def rolling_net_amount(
        self,
        df: pd.DataFrame,
        windows: Iterable[int] = (3, 5, 10),
    ) -> pd.DataFrame:
        """需要 ``change_pct`` 和 ``turnover`` 两列。"""
        if "change_pct" not in df.columns or "turnover" not in df.columns:
            raise ValueError("require columns: change_pct, turnover")
        out = df.copy()
        net = self.net_amount(out["change_pct"], out["turnover"])
        for w in windows:
            out[f"f_mf_net_{w}"] = net.rolling(window=w, min_periods=1).sum()
        return out
