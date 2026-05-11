"""E-003 主升浪事件检测器

输入：二板事件表 ``[code, first_date, second_date]``。
对每条二板事件，计算 ``second_date`` 后 ``n_days`` 个交易日内的
最高收盘价 / 二板收盘价 - 1 是否达到 ``return_threshold``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.validator import DataValidator


@dataclass
class MainWaveDetector:
    """主升浪事件检测器。"""

    n_days: int = 10
    return_threshold: float = 0.15

    # ------------------------------------------------------------------
    def detect(
        self,
        second_board_events: pd.DataFrame,
        daily_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """检测主升浪事件。

        Returns
        -------
        DataFrame
            列 ``code, second_date, main_wave_date, period_return, is_main_wave``。
        """
        DataValidator.check_required_columns(
            second_board_events, ["code", "second_date"], raise_error=True
        )
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "close"], raise_error=True
        )

        if second_board_events.empty:
            return self._empty()

        records = []
        daily_sorted = daily_data.sort_values(["code", "date"])
        grouped = {code: g.reset_index(drop=True) for code, g in daily_sorted.groupby("code")}

        for _, ev in second_board_events.iterrows():
            code = ev["code"]
            sec_date = ev["second_date"]
            day_df = grouped.get(code)
            if day_df is None or day_df.empty:
                continue

            try:
                idx = int(day_df.index[day_df["date"] == sec_date][0])
            except IndexError:
                continue

            close_at_second = float(day_df.loc[idx, "close"])
            window = day_df.iloc[idx + 1 : idx + 1 + self.n_days]
            if window.empty:
                continue

            max_close = float(window["close"].max())
            period_return = (max_close - close_at_second) / close_at_second
            is_main = period_return >= self.return_threshold
            if is_main:
                main_wave_date = window.loc[window["close"].idxmax(), "date"]
            else:
                main_wave_date = pd.NaT

            records.append({
                "code": code,
                "second_date": sec_date,
                "main_wave_date": main_wave_date,
                "period_return": period_return,
                "is_main_wave": is_main,
            })

        if not records:
            return self._empty()
        return pd.DataFrame(records).sort_values(["code", "second_date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def calc_period_return(
        self,
        code: str,
        start_date,
        n_days: int,
        daily_data: pd.DataFrame,
    ) -> float:
        """计算 ``start_date`` 后 ``n_days`` 交易日的最大收益率。"""
        sub = daily_data[daily_data["code"] == code].sort_values("date").reset_index(drop=True)
        try:
            idx = int(sub.index[sub["date"] == start_date][0])
        except IndexError:
            return float("nan")
        base = float(sub.loc[idx, "close"])
        window = sub.iloc[idx + 1 : idx + 1 + n_days]
        if window.empty:
            return float("nan")
        return (float(window["close"].max()) - base) / base

    # ------------------------------------------------------------------
    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "code", "second_date", "main_wave_date",
                "period_return", "is_main_wave",
            ]
        )
