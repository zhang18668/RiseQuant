"""U-001 TradingCalendar 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.calendar import TradingCalendar


class TestTradingCalendar:
    def setup_method(self):
        self.cal = TradingCalendar()

    def test_weekday_is_trading(self):
        # 2023-01-02 是周一
        assert self.cal.is_trading_day("2023-01-02") is True

    def test_weekend_not_trading(self):
        assert self.cal.is_trading_day("2023-01-07") is False
        assert self.cal.is_trading_day("2023-01-08") is False

    def test_holiday_not_trading(self):
        cal = TradingCalendar(holidays=["2023-01-02"])
        assert cal.is_trading_day("2023-01-02") is False

    def test_next_trading_day(self):
        # 2023-01-06 周五 -> 下个交易日 2023-01-09 周一
        assert self.cal.next_trading_day("2023-01-06") == "2023-01-09"

    def test_prev_trading_day(self):
        # 2023-01-09 周一 -> 上一个交易日 2023-01-06 周五
        assert self.cal.prev_trading_day("2023-01-09") == "2023-01-06"

    def test_next_trading_day_n_steps(self):
        # 2023-01-02 周一 -> +4 = 2023-01-06 周五
        assert self.cal.next_trading_day("2023-01-02", n=4) == "2023-01-06"

    def test_count_trading_days(self):
        # 2023-01-02 ~ 2023-01-06 周一到周五 = 5 个交易日
        assert self.cal.count_trading_days("2023-01-02", "2023-01-06") == 5

    def test_get_trading_days(self):
        days = self.cal.get_trading_days("2023-01-02", "2023-01-06")
        assert len(days) == 5
        assert days[0] == "2023-01-02"
        assert days[-1] == "2023-01-06"

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            self.cal.next_trading_day("2023-01-02", n=0)

    def test_count_with_holidays(self):
        cal = TradingCalendar(holidays=["2023-01-03"])
        # 周一~周五减去周二 = 4
        assert cal.count_trading_days("2023-01-02", "2023-01-06") == 4
