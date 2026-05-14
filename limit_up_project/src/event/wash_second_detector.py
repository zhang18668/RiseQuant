"""E-005 "首板 → 震荡洗盘 → 第二涨停" 事件检测器

业务定义
----------
**首板 (first board)**:
    当日涨停, 且 *前 ``cooldown_days`` 个交易日 (默认 3)* 内没有涨停.

**第二涨停 (second board)**:
    在首板之后的 [``min_gap``, ``max_gap``] 个交易日内 (默认 [3, 20]) 找到的下一个涨停,
    且必须满足:
    - 中间不能出现其他涨停 (否则那个涨停才是"真正的二板")
    - 第二涨停 close > 首板 close (创新高)
    - 与首板间隔 >= ``min_gap`` (>=3 -> 排除连板)

输出
----
``[code, first_date, second_date, gap_days,
   first_open, first_close, second_open, second_close,
   wash_max_high, wash_min_low, stop_loss_price]``

``stop_loss_price`` = 首板 ``open`` (策略侧建议的动态止损位).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


OUTPUT_COLS = [
    "code", "first_date", "second_date", "gap_days",
    "first_open", "first_close", "second_open", "second_close",
    "wash_max_high", "wash_min_low", "stop_loss_price",
]


@dataclass
class WashSecondDetector:
    """首板-震荡-第二涨停事件检测器."""

    threshold: float = 9.9         # 涨停阈值 (%)
    cooldown_days: int = 3         # 首板前 N 日不能有涨停
    min_gap: int = 3               # 首板到第二涨停最小交易日间隔 (排除连板)
    max_gap: int = 30              # 首板到第二涨停最大交易日间隔 (放宽到 30, 容纳更长的洗盘)
    exclude_st: bool = True

    # ------------------------------------------------------------------
    def detect(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """扫描日线找出所有 (first, second) 对."""
        if daily_data.empty:
            return self._empty()

        required = {"date", "code", "open", "close", "high", "low", "change_pct"}
        missing = required - set(daily_data.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")

        df = daily_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        st_mask = (
            df["is_st"].astype(bool).fillna(False)
            if self.exclude_st and "is_st" in df.columns
            else pd.Series(False, index=df.index)
        )
        df["is_lu"] = (df["change_pct"] >= self.threshold) & ~st_mask

        out_rows: List[dict] = []
        for code, g in df.groupby("code", sort=False):
            g = g.reset_index(drop=True)
            lu_arr = g["is_lu"].to_numpy()
            n = len(g)

            # 找首板候选: 当日涨停且前 cooldown_days 内无涨停
            for i in range(n):
                if not lu_arr[i]:
                    continue
                lo = max(0, i - self.cooldown_days)
                if lu_arr[lo:i].any():
                    continue  # 前 N 日内有涨停, 不是严格首板

                first_close = float(g["close"].iloc[i])
                first_open  = float(g["open"].iloc[i])

                # 在 [i + min_gap, i + max_gap] 区间找第二涨停, 且中间不能有涨停
                j_start = i + self.min_gap
                j_end   = min(n - 1, i + self.max_gap)
                if j_start > n - 1:
                    continue

                # 中间窗口 (i, i+min_gap) 已自动排除 (gap >= min_gap)
                # 但要保证 [i+1, j-1] 内没有其他涨停 (即 j 是 i 之后的第一个涨停)
                second_idx = None
                for j in range(i + 1, j_end + 1):
                    if lu_arr[j]:
                        if j >= j_start:
                            second_idx = j
                        break  # 不管是不是有效, 遇到涨停就停 — 否则中间已有涨停, 失败

                if second_idx is None:
                    continue

                second_close = float(g["close"].iloc[second_idx])
                if second_close <= first_close:
                    continue   # 必须创新高

                # 震荡区间 (i+1, second_idx-1) 的高低点 — 不含首板与第二涨停本身
                wash_slice = g.iloc[i + 1: second_idx]
                if wash_slice.empty:
                    wash_max = float("nan")
                    wash_min = float("nan")
                else:
                    wash_max = float(wash_slice["high"].max())
                    wash_min = float(wash_slice["low"].min())

                out_rows.append({
                    "code": str(code),
                    "first_date": g["date"].iloc[i],
                    "second_date": g["date"].iloc[second_idx],
                    "gap_days": int(second_idx - i),
                    "first_open": first_open,
                    "first_close": first_close,
                    "second_open": float(g["open"].iloc[second_idx]),
                    "second_close": second_close,
                    "wash_max_high": wash_max,
                    "wash_min_low": wash_min,
                    "stop_loss_price": first_open,  # 跌破首板 open 即出
                })

        if not out_rows:
            return self._empty()
        return pd.DataFrame(out_rows)[OUTPUT_COLS].sort_values(
            ["code", "first_date"]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=OUTPUT_COLS)
