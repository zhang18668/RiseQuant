"""E-001 涨停事件检测器

判定原则: 日涨幅 >= ``threshold`` (默认 9.9%), 可选排除 ST 股
(注册制相关板块 +/-20% 限制留待后续扩展, 本期不区分主板/创业板).

输出列
------
``date | code | change_pct | is_limit_up | limit_up_type | consecutive_n``

其中:
- ``limit_up_type``
    - ``first_board``  当日涨停, 且前一交易日不是涨停;
    - ``continuation`` 当日涨停, 且前一交易日也是涨停 (=连板);
- ``consecutive_n`` 截至当日已经连续涨停的天数, 首板为 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.utils.validator import DataValidator


OUTPUT_COLUMNS = [
    "date", "code", "change_pct", "is_limit_up",
    "limit_up_type", "consecutive_n",
]


@dataclass
class LimitUpEventDetector:
    """涨停事件检测器."""

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
        """判断单条记录是否涨停. ``change_pct`` 单位为 ``%``."""
        if change_pct is None or pd.isna(change_pct):
            return False
        if self.exclude_st and bool(is_st):
            return False
        return float(change_pct) >= self.threshold

    # ------------------------------------------------------------------
    def detect(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """对一份日线数据扫描涨停事件.

        必须包含列 ``date, code, change_pct``. 可选列 ``is_st``.
        返回 :data:`OUTPUT_COLUMNS` 所示列.
        """
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "change_pct"], raise_error=True
        )
        if daily_data.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df = daily_data[["date", "code", "change_pct"] +
                        (["is_st"] if "is_st" in daily_data.columns else [])].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        if self.exclude_st and "is_st" in df.columns:
            st_mask = df["is_st"].astype(bool).fillna(False)
        else:
            st_mask = pd.Series(False, index=df.index)

        df["is_limit_up"] = (df["change_pct"] >= self.threshold) & ~st_mask

        # 在原始 (code, date) 维度上计算连板天数 -----------------------------
        is_lu = df["is_limit_up"].astype(int)
        # 对每只股票, "连板组" 由 0 的累计和切分
        group_id = (is_lu == 0).astype(int).groupby(df["code"]).cumsum()
        consecutive_n = is_lu.groupby([df["code"], group_id]).cumsum()
        df["consecutive_n"] = consecutive_n

        # 仅保留涨停事件
        events = df[df["is_limit_up"]].copy()
        if events.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        events["limit_up_type"] = events["consecutive_n"].apply(
            lambda n: "first_board" if int(n) == 1 else "continuation"
        )
        events = events[OUTPUT_COLUMNS].sort_values(
            ["code", "date"]
        ).reset_index(drop=True)
        return events
