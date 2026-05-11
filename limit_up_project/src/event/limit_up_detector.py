"""E-001 涨停事件检测器

判定原则：日涨幅 >= ``threshold`` (默认 9.9%)，可选排除 ST 股 (注册制相关
板块 ±20% 限制留待后续扩展，本期不区分主板/创业板)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.utils.validator import DataValidator


@dataclass
class LimitUpEventDetector:
    """涨停事件检测器。"""

    threshold: float = 9.9
    exclude_st: bool = True

    # ------------------------------------------------------------------
    @staticmethod
    def _is_st_row(row: pd.Series) -> bool:
        if "is_st" in row.index and pd.notna(row.get("is_st")):
            return bool(row["is_st"])
        name = row.get("name") or row.get("stock_name")
        if isinstance(name, str) and ("ST" in name.upper()):
            return True
        return False

    def is_limit_up(self, change_pct: float, is_st: Optional[bool] = None) -> bool:
        """判断单条记录是否涨停。``change_pct`` 单位为 ``%``。"""
        if change_pct is None or pd.isna(change_pct):
            return False
        if self.exclude_st and bool(is_st):
            return False
        return float(change_pct) >= self.threshold

    # ------------------------------------------------------------------
    def detect(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """对一份日线数据扫描涨停事件。

        必须包含列 ``date, code, change_pct``。可选列 ``is_st``。
        返回列：``date, code, change_pct, is_limit_up``。
        """
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "change_pct"], raise_error=True
        )
        if daily_data.empty:
            return pd.DataFrame(columns=["date", "code", "change_pct", "is_limit_up"])

        df = daily_data.copy()
        if self.exclude_st and "is_st" in df.columns:
            st_mask = df["is_st"].astype(bool).fillna(False)
        else:
            st_mask = pd.Series(False, index=df.index)

        df["is_limit_up"] = (df["change_pct"] >= self.threshold) & ~st_mask
        events = df[df["is_limit_up"]].copy()
        events = events[["date", "code", "change_pct", "is_limit_up"]]
        events = events.sort_values(["code", "date"]).reset_index(drop=True)
        return events
