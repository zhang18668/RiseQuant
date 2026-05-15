"""Backtester variant for staged pullback entries."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.backtest.backtester import Backtester, Position, Trade


class AmbushBacktester(Backtester):
    def __init__(
        self,
        *args,
        scale_in_steps: int = 2,
        max_code_weight: float = 0.12,
        max_entry_over_first_close: float = 0.08,
        max_entry_gap: float = 0.06,
        max_entry_day_change_pct: float = 7.0,
        skip_broken_first_open: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scale_in_steps = max(1, int(scale_in_steps))
        self.max_code_weight = float(max_code_weight)
        self.max_entry_over_first_close = float(max_entry_over_first_close)
        self.max_entry_gap = float(max_entry_gap)
        self.max_entry_day_change_pct = float(max_entry_day_change_pct)
        self.skip_broken_first_open = bool(skip_broken_first_open)

    def _reset_state(self) -> None:
        super()._reset_state()
        self.tranches_by_code: Dict[str, int] = {}

    def _position_by_code(self, code: str) -> Position | None:
        for pos in self.positions:
            if pos.code == code:
                return pos
        return None

    def _passes_entry_filters(self, sig, row, signal_date, code, daily_idx) -> bool:
        open_price = float(row["open"])
        if open_price <= 0:
            return False

        if self.max_entry_day_change_pct > 0 and "change_pct" in row.index:
            if float(row["change_pct"]) >= self.max_entry_day_change_pct:
                return False

        if self.skip_broken_first_open and "f_w_broke_stop" in sig.index:
            v = sig["f_w_broke_stop"]
            if pd.notna(v) and float(v) >= 0.5:
                return False

        if self.max_entry_over_first_close > 0 and "first_close" in sig.index:
            first_close = sig["first_close"]
            if pd.notna(first_close) and float(first_close) > 0:
                if open_price / float(first_close) - 1.0 > self.max_entry_over_first_close:
                    return False

        if self.max_entry_gap > 0 and (signal_date, code) in daily_idx.index:
            ref_close = float(daily_idx.loc[(signal_date, code), "close"])
            if ref_close > 0 and open_price / ref_close - 1.0 > self.max_entry_gap:
                return False

        return True

    def _process_buys(self, current_date, signals: pd.DataFrame, daily_idx, all_dates) -> None:
        if self.entry_delay <= 0:
            signal_date = current_date
        else:
            try:
                cur_pos = all_dates.index(current_date)
            except ValueError:
                return
            sig_pos = cur_pos - self.entry_delay
            if sig_pos < 0:
                return
            signal_date = all_dates[sig_pos]

        day_signals = signals[signals["date"] == signal_date].sort_values("score", ascending=False)
        if self.min_score > 0:
            day_signals = day_signals[day_signals["score"] >= self.min_score]
        if day_signals.empty:
            return

        slots = self.topk - len(self.positions)
        for _, sig in day_signals.iterrows():
            code = str(sig["code"])
            pos = self._position_by_code(code)
            is_add = pos is not None
            if not is_add and slots <= 0:
                break
            if is_add and self.tranches_by_code.get(code, 1) >= self.scale_in_steps:
                continue

            cd = self.cooldown_until.get(code)
            if cd is not None and pd.Timestamp(current_date) <= cd:
                continue
            if (current_date, code) not in daily_idx.index:
                continue
            row = daily_idx.loc[(current_date, code)]
            open_price = float(row["open"])
            if not self._passes_entry_filters(sig, row, signal_date, code, daily_idx):
                continue

            if self.skip_zhangting_open and (signal_date, code) in daily_idx.index:
                ref_close = float(daily_idx.loc[(signal_date, code), "close"])
                if ref_close > 0 and open_price / ref_close - 1.0 >= 0.095:
                    continue

            slip_amount = open_price * self.slippage
            buy_price = open_price + slip_amount
            if buy_price <= 0:
                continue

            position_cap = self.initial_cash * self.max_code_weight
            tranche_budget = position_cap / self.scale_in_steps
            if not is_add and slots > 0:
                tranche_budget = min(tranche_budget, self.cash / max(slots, 1))
            tranche_budget = min(tranche_budget, self.cash)
            shares = tranche_budget / buy_price
            if shares <= 0:
                continue

            amount = shares * buy_price
            commission_cost = amount * self.commission
            cost = amount + commission_cost
            if cost > self.cash + 1e-9:
                continue
            self.cash -= cost

            dynamic_stop = 0.0
            if "stop_loss_price" in sig.index:
                v = sig["stop_loss_price"]
                if pd.notna(v) and float(v) > 0:
                    dynamic_stop = float(v)

            if pos is None:
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
                self.tranches_by_code[code] = 1
                slots -= 1
                action = "BUY"
            else:
                old_qty = pos.quantity
                new_qty = old_qty + shares
                pos.avg_cost = (pos.avg_cost * old_qty + buy_price * shares) / new_qty
                pos.quantity = new_qty
                pos.current_price = buy_price
                pos.high_watermark = max(pos.high_watermark, buy_price)
                if dynamic_stop > 0:
                    pos.dynamic_stop_price = max(pos.dynamic_stop_price, dynamic_stop)
                self.tranches_by_code[code] = self.tranches_by_code.get(code, 1) + 1
                action = "BUY_ADD"

            self.trades.append(
                Trade(
                    date=self._fmt_date(current_date),
                    code=code,
                    action=action,
                    price=buy_price,
                    quantity=shares,
                    amount=amount,
                    commission=commission_cost,
                    slippage=slip_amount * shares,
                )
            )
