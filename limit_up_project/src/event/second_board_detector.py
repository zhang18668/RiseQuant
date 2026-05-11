"""E-002 二板事件检测器

输入：涨停事件表 ``[date, code, change_pct]``。
对每只股票按时间排序，若两次相邻的涨停事件 *日历日* 间隔 <= ``n_days``
个交易日，则后者视为前者的"二板"。注意这里的 ``n_days`` 度量的是
"在涨停事件序列上的位置间隔"，即首板 -> 后续 n_days 个交易日内若有新的
涨停则成立。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.validator import DataValidator


@dataclass
class SecondBoardDetector:
    """二板事件检测器。"""

    n_days: int = 5  # 首板后多少个交易日内出现下一次涨停

    # ------------------------------------------------------------------
    def detect(
        self,
        limit_up_events: pd.DataFrame,
        daily_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """检测二板事件。

        Parameters
        ----------
        limit_up_events : DataFrame
            列 ``date, code``。
        daily_data : DataFrame
            列至少包含 ``date, code``，用来推导每只股票的交易日序列。

        Returns
        -------
        DataFrame
            列 ``code, first_date, second_date, gap_days``，``gap_days``
            表示二板与首板间隔的交易日数(>=1)。
        """
        DataValidator.check_required_columns(limit_up_events, ["date", "code"], raise_error=True)
        DataValidator.check_required_columns(daily_data, ["date", "code"], raise_error=True)

        if limit_up_events.empty:
            return self._empty()

        # 构建 (code -> ordered trading dates)
        daily_sorted = daily_data[["code", "date"]].drop_duplicates().sort_values(["code", "date"])
        date_rank = (
            daily_sorted.groupby("code").cumcount().rename("rank")
        )
        daily_indexed = daily_sorted.assign(rank=date_rank.values)
        rank_lookup = daily_indexed.set_index(["code", "date"])["rank"]

        records = []
        for code, group in limit_up_events.sort_values(["code", "date"]).groupby("code", sort=False):
            dates = group["date"].tolist()
            for first, second in zip(dates, dates[1:]):
                try:
                    r1 = int(rank_lookup.loc[(code, first)])
                    r2 = int(rank_lookup.loc[(code, second)])
                except KeyError:
                    continue
                gap = r2 - r1
                if 1 <= gap <= self.n_days:
                    records.append({
                        "code": code,
                        "first_date": first,
                        "second_date": second,
                        "gap_days": gap,
                    })

        if not records:
            return self._empty()

        df = pd.DataFrame(records)
        df = df.sort_values(["code", "first_date"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def is_second_board(
        self,
        code: str,
        first_date,
        limit_up_events: pd.DataFrame,
        daily_data: pd.DataFrame,
    ) -> Optional[dict]:
        """对单个 first_date 进行检测，返回 ``{second_date, gap_days}`` 或 ``None``。"""
        events_code = limit_up_events[limit_up_events["code"] == code].sort_values("date")
        daily_code = daily_data[daily_data["code"] == code].sort_values("date")
        if events_code.empty or daily_code.empty:
            return None

        dates = events_code["date"].tolist()
        first_date = pd.Timestamp(first_date)
        if first_date not in dates:
            return None
        idx = dates.index(first_date)
        if idx + 1 >= len(dates):
            return None
        next_date = dates[idx + 1]

        daily_sorted = daily_code.reset_index(drop=True)
        try:
            r1 = daily_sorted.index[daily_sorted["date"] == first_date][0]
            r2 = daily_sorted.index[daily_sorted["date"] == next_date][0]
        except IndexError:
            return None
        gap = int(r2 - r1)
        if 1 <= gap <= self.n_days:
            return {"second_date": next_date, "gap_days": gap}
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "first_date", "second_date", "gap_days"])
