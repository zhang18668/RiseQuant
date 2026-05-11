"""U-001 交易日历工具

设计原则
--------
默认行为仅基于"周一至周五"判定交易日。节假日采用注入式，
调用方可以传入特定年份的法定节假日列表，避免内置数据陈旧带来的隐性 bug。

接口
----
- ``TradingCalendar.is_trading_day(date)``
- ``TradingCalendar.next_trading_day(date, n=1)``
- ``TradingCalendar.prev_trading_day(date, n=1)``
- ``TradingCalendar.count_trading_days(start, end, inclusive=True)``
- ``TradingCalendar.get_trading_days(start, end)``

日期入参既可以是 ``str``、``datetime``、也可以是 ``pandas.Timestamp``。
"""

from __future__ import annotations

from datetime import date as _date, datetime, timedelta
from typing import Iterable, List, Optional, Union

import pandas as pd

DateLike = Union[str, _date, datetime, pd.Timestamp]


def _to_date(d: DateLike) -> _date:
    """将多种日期类型统一转换为 ``datetime.date``。"""
    if isinstance(d, _date) and not isinstance(d, datetime):
        return d
    return pd.Timestamp(d).date()


def _fmt(d: _date) -> str:
    return d.strftime("%Y-%m-%d")


class TradingCalendar:
    """A 股交易日历。

    Parameters
    ----------
    holidays : Iterable[DateLike], optional
        额外的节假日（不计为交易日）。默认空，意味着只过滤周末。
    """

    def __init__(self, holidays: Optional[Iterable[DateLike]] = None) -> None:
        self._holidays: set[_date] = set()
        if holidays:
            self._holidays = {_to_date(d) for d in holidays}

    # ------------------------------------------------------------------
    # 节假日管理
    # ------------------------------------------------------------------
    def add_holidays(self, holidays: Iterable[DateLike]) -> None:
        """新增节假日。"""
        self._holidays.update(_to_date(d) for d in holidays)

    @property
    def holidays(self) -> set[_date]:
        return set(self._holidays)

    # ------------------------------------------------------------------
    # 核心判定
    # ------------------------------------------------------------------
    def is_trading_day(self, date: DateLike) -> bool:
        """判断给定日期是否为交易日。"""
        d = _to_date(date)
        if d.weekday() >= 5:  # 周六/周日
            return False
        if d in self._holidays:
            return False
        return True

    # ------------------------------------------------------------------
    # 偏移
    # ------------------------------------------------------------------
    def next_trading_day(self, date: DateLike, n: int = 1) -> str:
        """向后偏移 ``n`` 个交易日，返回 ``YYYY-MM-DD``。"""
        if n <= 0:
            raise ValueError("n must be positive")
        d = _to_date(date)
        count = 0
        while count < n:
            d = d + timedelta(days=1)
            if self.is_trading_day(d):
                count += 1
        return _fmt(d)

    def prev_trading_day(self, date: DateLike, n: int = 1) -> str:
        """向前偏移 ``n`` 个交易日。"""
        if n <= 0:
            raise ValueError("n must be positive")
        d = _to_date(date)
        count = 0
        while count < n:
            d = d - timedelta(days=1)
            if self.is_trading_day(d):
                count += 1
        return _fmt(d)

    # 兼容别名 (方案文档中提及)
    get_next_trading_day = next_trading_day
    get_prev_trading_day = prev_trading_day
    get_n_trading_days_before = prev_trading_day
    get_n_trading_days_after = next_trading_day

    # ------------------------------------------------------------------
    # 区间
    # ------------------------------------------------------------------
    def get_trading_days(self, start_date: DateLike, end_date: DateLike) -> List[str]:
        """返回 [start, end] 闭区间内的交易日列表。"""
        start = _to_date(start_date)
        end = _to_date(end_date)
        if end < start:
            return []

        days: List[str] = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                days.append(_fmt(d))
            d = d + timedelta(days=1)
        return days

    def count_trading_days(
        self,
        start_date: DateLike,
        end_date: DateLike,
        inclusive: bool = True,
    ) -> int:
        """统计交易日数量。

        Parameters
        ----------
        inclusive : bool, default True
            是否将 ``start_date`` 和 ``end_date`` 计入。
            若为 ``False``，则仅计算 (start, end) 开区间内的交易日。
        """
        start = _to_date(start_date)
        end = _to_date(end_date)
        if end < start:
            return 0
        if not inclusive:
            start = start + timedelta(days=1)
            end = end - timedelta(days=1)
            if end < start:
                return 0
        return len(self.get_trading_days(start, end))


# ----------------------------------------------------------------------
# 模块级便捷函数
# ----------------------------------------------------------------------
_default_calendar: Optional[TradingCalendar] = None


def get_calendar() -> TradingCalendar:
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = TradingCalendar()
    return _default_calendar


def is_trading_day(date: DateLike) -> bool:
    return get_calendar().is_trading_day(date)


def next_trading_day(date: DateLike, n: int = 1) -> str:
    return get_calendar().next_trading_day(date, n)


def prev_trading_day(date: DateLike, n: int = 1) -> str:
    return get_calendar().prev_trading_day(date, n)
