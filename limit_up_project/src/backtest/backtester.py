"""B-001 简易回测引擎

实现一个最小但闭合的事件驱动回测器：
- 每个交易日，按当日 ``signals`` 选 top-K 候选；
- 持仓上限 ``topk`` 只，等权买入；
- 持仓达 ``sell_n`` 个交易日后强制卖出 (按收盘价)；
- 滑点 ``slippage`` 影响买入价。

输入约定
--------
``signals`` : DataFrame
    列 ``date, code, score``。``score`` 越高优先级越高。
``daily_data`` : DataFrame
    列 ``date, code, open, high, low, close, volume``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.validator import DataValidator


@dataclass
class _Position:
    code: str
    buy_date: pd.Timestamp
    buy_price: float
    shares: float


@dataclass
class Backtester:
    """事件驱动的简易回测器。"""

    initial_cash: float = 10_000_000.0
    topk: int = 10
    sell_n: int = 5
    slippage: float = 0.003

    # internal state
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "equity"]))

    # ------------------------------------------------------------------
    def run(self, signals: pd.DataFrame, daily_data: pd.DataFrame) -> pd.DataFrame:
        DataValidator.check_required_columns(
            signals, ["date", "code", "score"], raise_error=True
        )
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "open", "close"], raise_error=True
        )

        if signals.empty or daily_data.empty:
            self.equity_curve = pd.DataFrame({"date": [], "equity": []})
            return self.equity_curve

        signals = signals.copy()
        daily_data = daily_data.copy()
        signals["date"] = pd.to_datetime(signals["date"])
        daily_data["date"] = pd.to_datetime(daily_data["date"])

        # 索引化：日线按 (date, code) 查
        daily_idx = daily_data.set_index(["date", "code"])
        all_dates = sorted(daily_data["date"].unique())

        cash = float(self.initial_cash)
        positions: List[_Position] = []
        trades: List[Dict] = []
        equity_records: List[Dict] = []

        for current_date in all_dates:
            # 1. 处理到期卖出
            keep: List[_Position] = []
            for pos in positions:
                holding = self._holding_days(pos.buy_date, current_date, all_dates)
                row = daily_idx.loc[(current_date, pos.code)] if (current_date, pos.code) in daily_idx.index else None
                if row is None:
                    keep.append(pos)
                    continue
                sell_price = float(row["close"])
                if holding >= self.sell_n:
                    pnl = (sell_price - pos.buy_price) * pos.shares
                    cash += sell_price * pos.shares
                    trades.append({
                        "code": pos.code,
                        "buy_date": pos.buy_date,
                        "sell_date": current_date,
                        "buy_price": pos.buy_price,
                        "sell_price": sell_price,
                        "shares": pos.shares,
                        "pnl": pnl,
                        "return": (sell_price / pos.buy_price - 1.0) if pos.buy_price > 0 else 0.0,
                    })
                else:
                    keep.append(pos)
            positions = keep

            # 2. 买入：按 signals 当日 top-K
            day_signals = signals[signals["date"] == current_date].sort_values("score", ascending=False)
            held_codes = {p.code for p in positions}
            slots = self.topk - len(positions)
            if slots > 0 and not day_signals.empty:
                budget_per = cash / max(slots, 1)
                for _, sig in day_signals.iterrows():
                    if slots <= 0:
                        break
                    code = sig["code"]
                    if code in held_codes:
                        continue
                    if (current_date, code) not in daily_idx.index:
                        continue
                    open_price = float(daily_idx.loc[(current_date, code), "open"])
                    if open_price <= 0:
                        continue
                    buy_price = open_price * (1 + self.slippage)
                    shares = budget_per / buy_price
                    if shares <= 0:
                        continue
                    cost = shares * buy_price
                    if cost > cash:
                        continue
                    cash -= cost
                    positions.append(_Position(code, current_date, buy_price, shares))
                    held_codes.add(code)
                    slots -= 1

            # 3. 估算当日权益
            mv = 0.0
            for pos in positions:
                row = daily_idx.loc[(current_date, pos.code)] if (current_date, pos.code) in daily_idx.index else None
                if row is None:
                    mv += pos.shares * pos.buy_price
                else:
                    mv += pos.shares * float(row["close"])
            equity_records.append({"date": current_date, "equity": cash + mv})

        self.trades = trades
        self.equity_curve = pd.DataFrame(equity_records)
        return self.equity_curve

    # ------------------------------------------------------------------
    @staticmethod
    def _holding_days(buy_date: pd.Timestamp, current_date: pd.Timestamp, all_dates) -> int:
        """计算从 ``buy_date`` (不含) 到 ``current_date`` (含) 的交易日数。"""
        try:
            i_buy = all_dates.index(buy_date)
            i_now = all_dates.index(current_date)
        except ValueError:
            return 0
        return max(0, i_now - i_buy)

    # ------------------------------------------------------------------
    def get_trades(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "code", "buy_date", "sell_date", "buy_price", "sell_price",
                "shares", "pnl", "return",
            ])
        return pd.DataFrame(self.trades)
