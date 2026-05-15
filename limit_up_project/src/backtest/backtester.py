"""B-001 简易回测引擎

实现一个最小但闭合的事件驱动回测器:
- 每个交易日 T, 取 ``T - entry_delay`` 日 (默认 T-1) 产生的 ``signals`` 选 top-K 候选;
- 持仓上限 ``topk`` 只, 等权买入;
- 持仓达 ``sell_n`` 个交易日后强制卖出 (按收盘价);
- 滑点 ``slippage`` 影响买入价;
- 佣金 ``commission`` 影响交易成本.

杜绝未来函数说明
------------------
- 信号日 (signal_date) 必须使用 *仅含 signal_date 及之前* 的数据生成;
- 实际撮合发生在 signal_date 之后的第 ``entry_delay`` 个交易日 (默认 1, 即 T+1 开盘);
- ``entry_delay`` 设为 0 等同于"信号日当日开盘成交", 仅在你 *确认信号特征 *
  完全不依赖信号日数据* (例如使用 ``signal_date - 1`` 的特征) 时才可用,
  否则会引入 look-ahead bias.

公开类
------
- :class:`Trade` 单笔交易记录;
- :class:`Position` 单只股票持仓状态;
- :class:`Backtester` 回测引擎.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validator import DataValidator

logger = get_logger(__name__)


@dataclass
class Trade:
    """单笔交易记录."""

    date: str
    code: str
    action: str  # BUY / SELL
    price: float
    quantity: float
    amount: float
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "date": self.date,
            "code": self.code,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "amount": self.amount,
            "commission": self.commission,
            "slippage": self.slippage,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
        }


@dataclass
class Position:
    """单只股票持仓状态."""

    code: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    buy_date: Optional[pd.Timestamp] = None
    holding_days: int = 0
    high_watermark: float = 0.0           # 持仓期间到达过的最高价 (移动止盈用)
    dynamic_stop_price: float = 0.0       # 动态止损价 (跌破即出, 由信号附带)

    @property
    def market_value(self) -> float:
        return float(self.quantity) * float(self.current_price)

    @property
    def cost_value(self) -> float:
        return float(self.quantity) * float(self.avg_cost)

    @property
    def profit(self) -> float:
        return (float(self.current_price) - float(self.avg_cost)) * float(self.quantity)

    @property
    def return_pct(self) -> float:
        if self.avg_cost <= 0:
            return 0.0
        return float(self.current_price) / float(self.avg_cost) - 1.0


TRADING_DAYS_PER_YEAR = 252


class Backtester:
    """事件驱动的简易回测器."""

    def __init__(
        self,
        initial_cash: float = 10_000_000.0,
        topk: int = 10,
        sell_n: int = 5,
        slippage: float = 0.003,
        commission: float = 0.0,
        entry_delay: int = 1,
        # ---- 风控扩展 ----
        stop_loss: float = 0.0,        # 硬止损, 例如 0.03 表示 -3% 立即出场
        take_profit: float = 0.0,      # 硬止盈, 例如 0.10 表示 +10% 兑现
        trailing_stop: float = 0.0,    # 移动止盈, 例如 0.05 表示从最高点回撤 5% 出场
        min_score: float = 0.0,        # 概率阈值: 信号 score < min_score 不买
        skip_zhangting_open: bool = True,  # T+1 一字涨停 (open>=9.5%) 买不进
        cooldown_after_stop: int = 5,  # 止损后该 code 在 N 个交易日内禁买 (0 关闭)
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.topk = int(topk)
        self.sell_n = int(sell_n)
        self.slippage = float(slippage)
        self.commission = float(commission)
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)
        self.trailing_stop = float(trailing_stop)
        self.min_score = float(min_score)
        self.skip_zhangting_open = bool(skip_zhangting_open)
        self.cooldown_after_stop = int(cooldown_after_stop)
        # 信号 -> 撮合的交易日延迟 (>=1 杜绝未来函数, 信号日 T 在 T+entry_delay 开盘成交)
        if int(entry_delay) < 0:
            raise ValueError("entry_delay must be >= 0")
        if int(entry_delay) == 0:
            msg = (
                "Backtester: entry_delay=0 — 信号日当日开盘成交。除非信号特征完全"
                "不使用信号日的数据，否则会引入未来函数。建议使用 entry_delay>=1。"
            )
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
        self.entry_delay = int(entry_delay)

        # 运行时状态
        self.cash: float = self.initial_cash
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: pd.DataFrame = pd.DataFrame(
            columns=["date", "cash", "market_value", "total_value"]
        )

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.cash = self.initial_cash
        self.positions = []
        self.trades = []
        self.equity_curve = pd.DataFrame(
            columns=["date", "cash", "market_value", "total_value"]
        )
        # 冷静期: code -> 解禁日 (Timestamp), 该日及之前禁买
        self.cooldown_until: Dict[str, pd.Timestamp] = {}

    @staticmethod
    def _fmt_date(d) -> str:
        return pd.Timestamp(d).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    def run(
        self,
        signals: pd.DataFrame,
        daily_data: pd.DataFrame,
    ) -> pd.DataFrame:
        self._reset_state()

        # 早退: 空输入直接返回空 curve, 不强校验列
        if signals is None or daily_data is None or signals.empty or daily_data.empty:
            return self.equity_curve.copy()

        DataValidator.check_required_columns(
            signals, ["date", "code", "score"], raise_error=True
        )
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "open", "close"], raise_error=True
        )

        signals = signals.copy()
        daily_data = daily_data.copy()
        signals["date"] = pd.to_datetime(signals["date"])
        daily_data["date"] = pd.to_datetime(daily_data["date"])

        # 去重 + 排序: 防止 (date, code) 重复导致 .loc 返回 Series,
        # 同时消除 "indexing past lexsort depth" 的 PerformanceWarning
        daily_data = daily_data.drop_duplicates(subset=["date", "code"], keep="last")
        daily_idx = daily_data.set_index(["date", "code"]).sort_index()
        all_dates = sorted(daily_data["date"].unique())

        equity_records: List[Dict[str, object]] = []

        for current_date in all_dates:
            self._process_sells(current_date, daily_idx, all_dates)
            self._process_buys(current_date, signals, daily_idx, all_dates)
            market_value = self._mark_to_market(current_date, daily_idx)
            total_value = self.cash + market_value
            equity_records.append({
                "date": current_date,
                "cash": self.cash,
                "market_value": market_value,
                "total_value": total_value,
            })

        self.equity_curve = pd.DataFrame(equity_records)
        return self.equity_curve.copy()

    # ------------------------------------------------------------------
    def _process_sells(self, current_date, daily_idx, all_dates) -> None:
        keep: List[Position] = []
        for pos in self.positions:
            row = (
                daily_idx.loc[(current_date, pos.code)]
                if (current_date, pos.code) in daily_idx.index
                else None
            )
            if row is None:
                keep.append(pos)
                continue

            holding = self._holding_days(pos.buy_date, current_date, all_dates)
            pos.holding_days = holding
            # 用当日 low/high 判定盘中触发止损/止盈, 用 close 更新当前价
            day_high = float(row["high"]) if "high" in row.index else float(row["close"])
            day_low  = float(row["low"])  if "low"  in row.index else float(row["close"])
            day_close = float(row["close"])
            pos.current_price = day_close
            # 维护持仓期间最高价 (用每日 high)
            pos.high_watermark = max(pos.high_watermark, day_high) if pos.high_watermark else day_high

            # ---- 判定卖出条件 ----
            sell_reason: Optional[str] = None
            sell_price: float = day_close

            ret_to_close = (day_close / pos.avg_cost - 1.0) if pos.avg_cost > 0 else 0.0
            ret_to_low   = (day_low   / pos.avg_cost - 1.0) if pos.avg_cost > 0 else 0.0
            ret_to_high  = (day_high  / pos.avg_cost - 1.0) if pos.avg_cost > 0 else 0.0

            # 0) 动态止损: 当日 low 跌破信号附带的 stop_loss_price 即出
            if pos.dynamic_stop_price > 0 and day_low <= pos.dynamic_stop_price:
                sell_reason = "STOP_DYNAMIC"
                sell_price = pos.dynamic_stop_price
            # 1) 硬止损 (按当日 low 触发, 成交价 = 触发价)
            elif self.stop_loss > 0 and ret_to_low <= -self.stop_loss:
                sell_reason = "STOP_LOSS"
                sell_price = pos.avg_cost * (1.0 - self.stop_loss)
            # 2) 硬止盈 (按当日 high 触发)
            elif self.take_profit > 0 and ret_to_high >= self.take_profit:
                sell_reason = "TAKE_PROFIT"
                sell_price = pos.avg_cost * (1.0 + self.take_profit)
            # 3) 移动止盈: 从 high_watermark 回撤超过阈值
            elif self.trailing_stop > 0 and pos.high_watermark > 0:
                drawdown = day_close / pos.high_watermark - 1.0
                if drawdown <= -self.trailing_stop:
                    sell_reason = "TRAILING_STOP"
                    sell_price = day_close
            # 4) 时间止损 (默认逻辑保留)
            if sell_reason is None and holding >= self.sell_n:
                sell_reason = "TIME_STOP"
                sell_price = day_close

            if sell_reason is not None:
                amount = sell_price * pos.quantity
                commission_cost = amount * self.commission
                proceeds = amount - commission_cost
                pnl = (sell_price - pos.avg_cost) * pos.quantity - commission_cost
                return_pct = (sell_price / pos.avg_cost - 1.0) if pos.avg_cost > 0 else 0.0
                self.cash += proceeds
                self.trades.append(
                    Trade(
                        date=self._fmt_date(current_date),
                        code=pos.code,
                        action=f"SELL_{sell_reason}",
                        price=sell_price,
                        quantity=pos.quantity,
                        amount=amount,
                        commission=commission_cost,
                        slippage=0.0,
                        pnl=pnl,
                        return_pct=return_pct,
                    )
                )
                # 止损类触发冷静期, 设定 cooldown 截止日 (含当日)
                if self.cooldown_after_stop > 0 and sell_reason in (
                    "STOP_DYNAMIC", "STOP_LOSS", "TRAILING_STOP"
                ):
                    try:
                        cur_pos = all_dates.index(current_date)
                        end_pos = min(len(all_dates) - 1, cur_pos + self.cooldown_after_stop)
                        self.cooldown_until[pos.code] = all_dates[end_pos]
                    except ValueError:
                        # 找不到 current_date, 退化为日历日加 N 天
                        self.cooldown_until[pos.code] = (
                            pd.Timestamp(current_date) + pd.Timedelta(days=self.cooldown_after_stop)
                        )
            else:
                keep.append(pos)
        self.positions = keep

    # ------------------------------------------------------------------
    def _process_buys(
        self,
        current_date,
        signals: pd.DataFrame,
        daily_idx,
        all_dates,
    ) -> None:
        # ------------------------------------------------------------------
        # 杜绝未来函数: 信号日 (signal_date) 必须严格早于撮合日 (current_date),
        # 由 entry_delay 控制. 默认 entry_delay=1, 即 T 日信号在 T+1 开盘成交.
        # ------------------------------------------------------------------
        if self.entry_delay <= 0:
            signal_date = current_date
        else:
            try:
                cur_pos = all_dates.index(current_date)
            except ValueError:
                return
            sig_pos = cur_pos - self.entry_delay
            if sig_pos < 0:
                return  # 暖机期: 没有足够历史日产生信号
            signal_date = all_dates[sig_pos]

        day_signals = signals[signals["date"] == signal_date].sort_values(
            "score", ascending=False
        )
        # 概率阈值过滤
        if self.min_score > 0:
            day_signals = day_signals[day_signals["score"] >= self.min_score]
        if day_signals.empty:
            return
        held_codes = {p.code for p in self.positions}
        slots = self.topk - len(self.positions)
        if slots <= 0:
            return

        budget_per = self.cash / max(slots, 1)
        for _, sig in day_signals.iterrows():
            if slots <= 0:
                break
            code = str(sig["code"])
            if code in held_codes:
                continue
            # 冷静期: 该 code 在 cooldown 截止日及之前禁买
            cd = self.cooldown_until.get(code)
            if cd is not None and pd.Timestamp(current_date) <= cd:
                continue
            if (current_date, code) not in daily_idx.index:
                continue
            row = daily_idx.loc[(current_date, code)]
            open_price = float(row["open"])
            if open_price <= 0:
                continue

            # 一字/秒板涨停: 实盘买不进
            if self.skip_zhangting_open:
                # 找到 signal_date 的 close 作为参考
                if (signal_date, code) in daily_idx.index:
                    ref_close = float(daily_idx.loc[(signal_date, code), "close"])
                    if ref_close > 0:
                        gap_pct = open_price / ref_close - 1.0
                        if gap_pct >= 0.095:
                            continue

            slip_amount = open_price * self.slippage
            buy_price = open_price + slip_amount
            if buy_price <= 0:
                continue
            shares = budget_per / buy_price
            if shares <= 0:
                continue
            amount = shares * buy_price
            commission_cost = amount * self.commission
            cost = amount + commission_cost
            if cost > self.cash + 1e-9:
                continue
            self.cash -= cost
            # 信号附带的动态止损价 (如首板 open)
            dynamic_stop = 0.0
            if "stop_loss_price" in sig.index:
                v = sig["stop_loss_price"]
                if pd.notna(v) and float(v) > 0:
                    dynamic_stop = float(v)
            self.positions.append(
                Position(
                    code=code,
                    quantity=shares,
                    avg_cost=buy_price,
                    current_price=buy_price,
                    buy_date=pd.Timestamp(current_date),
                    holding_days=0,
                    high_watermark=buy_price,
                    dynamic_stop_price=dynamic_stop,
                )
            )
            self.trades.append(
                Trade(
                    date=self._fmt_date(current_date),
                    code=code,
                    action="BUY",
                    price=buy_price,
                    quantity=shares,
                    amount=amount,
                    commission=commission_cost,
                    slippage=slip_amount * shares,
                )
            )
            held_codes.add(code)
            slots -= 1

    # ------------------------------------------------------------------
    def _mark_to_market(self, current_date, daily_idx) -> float:
        mv = 0.0
        for pos in self.positions:
            if (current_date, pos.code) in daily_idx.index:
                pos.current_price = float(
                    daily_idx.loc[(current_date, pos.code), "close"]
                )
            mv += pos.market_value
        return float(mv)

    @staticmethod
    def _holding_days(buy_date, current_date, all_dates) -> int:
        if buy_date is None:
            return 0
        try:
            i_buy = all_dates.index(pd.Timestamp(buy_date))
            i_now = all_dates.index(pd.Timestamp(current_date))
        except ValueError:
            return 0
        return max(0, i_now - i_buy)

    # ------------------------------------------------------------------
    def get_trades(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "date", "code", "action", "price", "quantity", "amount",
                "commission", "slippage", "pnl", "return_pct",
            ])
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def get_metrics(self) -> Dict[str, float]:
        if self.equity_curve is None or self.equity_curve.empty:
            return {
                "initial_cash": self.initial_cash,
                "final_value": self.initial_cash,
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": float("nan"),
                "max_drawdown": 0.0,
                "num_trades": 0,
                "win_rate": float("nan"),
            }

        ec = self.equity_curve.sort_values("date").reset_index(drop=True)
        equity = ec["total_value"].astype(float)
        initial = float(self.initial_cash)
        final = float(equity.iloc[-1])
        total_return = final / initial - 1.0 if initial > 0 else 0.0
        days = max(len(equity), 1)
        annual_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1)

        rets = equity.pct_change().dropna()
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if rets.std() > 0
            else float("nan")
        )

        cummax = equity.cummax()
        max_drawdown = float((equity / cummax - 1.0).min()) if not cummax.empty else 0.0

        sell_trades = [t for t in self.trades if str(t.action).startswith("SELL")]
        if sell_trades:
            wins = sum(1 for t in sell_trades if t.return_pct > 0)
            win_rate = wins / len(sell_trades)
        else:
            win_rate = float("nan")

        return {
            "initial_cash": initial,
            "final_value": final,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": int(len(self.trades)),
            "win_rate": win_rate,
        }
