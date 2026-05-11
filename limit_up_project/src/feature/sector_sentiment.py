"""F-003 板块情绪特征

由于本期暂不接入实时板块数据，这里实现一组与"全市场情绪"近似的指标：
- 当日上涨股票占比；
- 当日涨停股票数；
- 历史 N 日的涨停股票均值。

接口设计为 *跨股票* 的，输入是宽表（按日期聚合）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class SectorSentimentFeatures:
    """市场/板块情绪特征计算器。"""

    def __init__(self, threshold: float = 9.9) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    def market_sentiment(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """按日期聚合，计算市场广度指标。

        必须包含列 ``date, change_pct``。
        返回列 ``date, f_mkt_up_ratio, f_mkt_limit_up_count``。
        """
        if "date" not in daily_data.columns or "change_pct" not in daily_data.columns:
            raise ValueError("daily_data must contain 'date' and 'change_pct'")

        df = daily_data.copy()
        df["_up"] = (df["change_pct"] > 0).astype(int)
        df["_lu"] = (df["change_pct"] >= self.threshold).astype(int)

        agg = df.groupby("date").agg(
            f_mkt_up_ratio=("_up", "mean"),
            f_mkt_limit_up_count=("_lu", "sum"),
        ).reset_index()
        return agg

    def rolling_limit_up_count(
        self,
        sentiment: pd.DataFrame,
        windows: Iterable[int] = (3, 5, 10),
    ) -> pd.DataFrame:
        """对 :meth:`market_sentiment` 的输出进一步做滚动统计。"""
        if "f_mkt_limit_up_count" not in sentiment.columns:
            raise ValueError("expect column f_mkt_limit_up_count")
        df = sentiment.sort_values("date").copy()
        for w in windows:
            df[f"f_mkt_lu_ma_{w}"] = df["f_mkt_limit_up_count"].rolling(window=w, min_periods=1).mean()
        return df.reset_index(drop=True)
