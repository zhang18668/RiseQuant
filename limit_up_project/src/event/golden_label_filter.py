"""E-006 金标准过滤器 (Pattern-Cluster v2)

对 ``WashSecondDetector`` 输出的 (first, second) 三元组叠加一层
"未来 N 个交易日最高 close 涨幅 ∈ [lo, hi]" 的过滤.

只有满足这一条件的事件才会被视为 "形态聚类的正样本来源"; 不满足的事件 (中性 / 失败)
仍然在 events 表中保留, 但 ``is_golden = False``, 在下游会作为负样本池.

业务定义 (v2)
-------------
对每一个 event 行 ``(code, second_date, second_close)``:

1. 取该 code 在 ``second_date`` *之后* (不含当日) ``horizon`` 个交易日的 close;
2. ``future_max_close`` = 这段窗口内的最大 close;
3. ``future_max_pct``   = ``future_max_close / second_close - 1``;
4. ``future_max_date``  = 取到最大 close 的那一天;
5. ``is_golden = lo <= future_max_pct <= hi``.

如果 second_date 之后日线不足 ``horizon`` 天, 用现有可用窗口计算
(不补外推, 但 ``is_golden`` 不强制为 False — 哪怕只剩 10 天就涨到 25% 也算金标准).
但 ``has_full_horizon`` 列会标 False, 便于事后做敏感性分析.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


OUTPUT_EXTRA_COLS = [
    "future_max_close", "future_max_pct", "future_max_date",
    "has_full_horizon", "is_golden",
]


@dataclass
class GoldenLabelFilter:
    """金标准过滤器 — 看 second_date 之后 N 日最高 close 涨幅."""

    lo: float = 0.15
    hi: float = 0.35
    horizon: int = 22

    # ------------------------------------------------------------------
    def filter(
        self,
        events: pd.DataFrame,
        daily_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """给 events 增加 5 个新列, 不丢行.

        Parameters
        ----------
        events : DataFrame
            ``WashSecondDetector.detect()`` 的输出, 至少含 ``code, second_date, second_close``.
        daily_data : DataFrame
            原始日线, 至少含 ``date, code, close``.

        Returns
        -------
        DataFrame
            原 events 的全部列 + 5 个新列.
        """
        if events is None or events.empty:
            return self._empty_with_cols(events)

        required = {"code", "second_date", "second_close"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"events missing columns: {missing}")
        if daily_data is None or daily_data.empty:
            raise ValueError("daily_data is empty")
        if not {"date", "code", "close"}.issubset(daily_data.columns):
            raise ValueError("daily_data must contain date/code/close")

        df = daily_data[["date", "code", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 索引: code -> (sorted_dates, close_array)
        groups: Dict[str, tuple] = {}
        for code, g in df.groupby("code", sort=False):
            groups[str(code)] = (
                g["date"].tolist(),
                g["close"].to_numpy(dtype=float),
            )

        out_rows: List[Dict] = []
        for _, ev in events.iterrows():
            code = str(ev["code"])
            second_date = pd.Timestamp(ev["second_date"])
            second_close = float(ev["second_close"])
            row = ev.to_dict()
            row.update(self._compute_future_window(groups.get(code), second_date, second_close))
            out_rows.append(row)

        out = pd.DataFrame(out_rows)
        # 保证列顺序: 原列在前, 新列在后
        ordered_cols = list(events.columns) + [c for c in OUTPUT_EXTRA_COLS if c not in events.columns]
        for c in OUTPUT_EXTRA_COLS:
            if c not in out.columns:
                out[c] = np.nan if c != "is_golden" and c != "has_full_horizon" else False
        return out[ordered_cols]

    # ------------------------------------------------------------------
    def _compute_future_window(
        self,
        code_info: tuple,
        second_date: pd.Timestamp,
        second_close: float,
    ) -> Dict:
        """计算单个 event 的 future_max_* 与 is_golden."""
        nan_result = {
            "future_max_close": float("nan"),
            "future_max_pct":   float("nan"),
            "future_max_date":  pd.NaT,
            "has_full_horizon": False,
            "is_golden":        False,
        }
        if code_info is None:
            return nan_result
        dates, closes = code_info
        if not dates or second_close <= 0:
            return nan_result

        # 找 second_date 在 dates 中的位置
        try:
            idx = dates.index(second_date)
        except ValueError:
            # second_date 不在该 code 的交易日序列中 — 数据缺口, 跳过
            return nan_result

        start = idx + 1
        end = min(len(dates), start + self.horizon)
        if start >= end:
            return nan_result

        window = closes[start:end]
        if len(window) == 0:
            return nan_result

        # 找窗口内最大 close 及其日期
        max_pos = int(np.argmax(window))
        future_max_close = float(window[max_pos])
        future_max_date = dates[start + max_pos]
        future_max_pct = future_max_close / second_close - 1.0
        has_full = (end - start) >= self.horizon
        is_golden = bool(self.lo <= future_max_pct <= self.hi)

        return {
            "future_max_close": future_max_close,
            "future_max_pct":   float(future_max_pct),
            "future_max_date":  future_max_date,
            "has_full_horizon": has_full,
            "is_golden":        is_golden,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_with_cols(events) -> pd.DataFrame:
        cols = list(events.columns) if events is not None else []
        for c in OUTPUT_EXTRA_COLS:
            if c not in cols:
                cols.append(c)
        return pd.DataFrame(columns=cols)
