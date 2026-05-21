"""回测引擎测试"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import Backtester, Trade, Position


class TestBacktester:
    """回测引擎测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.backtester = Backtester(
            initial_cash=10000000,
            topk=10,
            sell_n=5,
            slippage=0.003,
        )

    def test_initialization(self):
        """初始化"""
        assert self.backtester.initial_cash == 10000000
        assert self.backtester.cash == 10000000
        assert len(self.backtester.positions) == 0
        assert len(self.backtester.trades) == 0

    def test_run_with_empty_signals(self):
        """空信号"""
        signals = pd.DataFrame(columns=["date", "code", "score"])
        daily_data = pd.DataFrame()
        result = self.backtester.run(signals, daily_data)
        assert result.empty

    def test_run_with_signals(self):
        """运行回测"""
        # 创建信号
        dates = pd.date_range("2023-01-01", periods=10, freq="B").strftime("%Y-%m-%d")
        signals = pd.DataFrame({
            "date": list(dates) * 3,
            "code": ["000001", "000002", "000003"] * 10,
            "score": np.random.rand(30),
        })

        # 创建日线数据
        records = []
        for code in ["000001", "000002", "000003"]:
            for i, date in enumerate(dates):
                records.append({
                    "date": date,
                    "code": code,
                    "open": 10.0 + i * 0.1,
                    "high": 10.1 + i * 0.1,
                    "low": 9.9 + i * 0.1,
                    "close": 10.0 + i * 0.1,
                    "volume": 1000000,
                })
        daily_data = pd.DataFrame(records)

        result = self.backtester.run(signals, daily_data)

        # 检查结果
        assert not result.empty
        assert "date" in result.columns
        assert "total_value" in result.columns

    def test_same_day_close_entry_price(self):
        """Signal-day close execution for tail-buy simulations."""
        bt = Backtester(
            initial_cash=100000,
            topk=1,
            sell_n=1,
            slippage=0.0,
            entry_delay=0,
            entry_price="close",
            skip_zhangting_open=False,
        )
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        signals = pd.DataFrame({
            "date": [dates[0]],
            "code": ["600000"],
            "score": [1.0],
        })
        daily = pd.DataFrame({
            "date": list(dates),
            "code": ["600000"] * 3,
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.5, 10.5, 11.5],
            "close": [10.8, 11.5, 12.5],
            "volume": [1000, 1000, 1000],
        })

        bt.run(signals, daily)
        trades = bt.get_trades()

        buy = trades[trades["action"] == "BUY"].iloc[0]
        assert buy["date"] == "2024-01-02"
        assert buy["price"] == 10.8


class TestTrade:
    """交易记录测试"""

    def test_trade_creation(self):
        """创建交易"""
        trade = Trade(
            date="2023-01-01",
            code="000001",
            action="BUY",
            price=10.0,
            quantity=1000,
            amount=10000.0,
            commission=3.0,
            slippage=10.0,
        )

        assert trade.date == "2023-01-01"
        assert trade.code == "000001"
        assert trade.action == "BUY"
        assert trade.price == 10.0


class TestPosition:
    """持仓测试"""

    def test_position_creation(self):
        """创建持仓"""
        position = Position(
            code="000001",
            quantity=1000,
            avg_cost=10.0,
            current_price=11.0,
        )

        assert position.code == "000001"
        assert position.quantity == 1000
        assert position.avg_cost == 10.0

    def test_position_market_value(self):
        """市值计算"""
        position = Position(
            code="000001",
            quantity=1000,
            avg_cost=10.0,
            current_price=11.0,
        )

        assert position.market_value == 11000.0

    def test_position_profit(self):
        """盈亏计算"""
        position = Position(
            code="000001",
            quantity=1000,
            avg_cost=10.0,
            current_price=11.0,
        )

        assert position.profit == 1000.0
