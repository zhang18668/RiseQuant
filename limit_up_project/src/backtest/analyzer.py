"""B-002 业绩归因/绩效分析

输入回测引擎产生的 ``equity_curve``，计算常用指标:
- 累计收益、年化收益
- 夏普比率
- 最大回撤
- 胜率、平均盈亏比（基于交易记录）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


class PerformanceAnalyzer:
    """绩效分析器。"""

    def analyze(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        if equity_curve is None or equity_curve.empty:
            return self._empty_metrics()

        df = equity_curve.sort_values("date").reset_index(drop=True)
        equity = df["equity"].astype(float)
        if (equity <= 0).any() or len(equity) < 2:
            return self._empty_metrics()

        returns = equity.pct_change().dropna()
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        days = max(len(equity), 1)
        annual_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1)

        sharpe = float(
            returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        ) if returns.std() > 0 else float("nan")

        cummax = equity.cummax()
        drawdown = equity / cummax - 1.0
        max_drawdown = float(drawdown.min())

        win_rate = float("nan")
        avg_pnl = float("nan")
        if trades is not None and not trades.empty and "return" in trades.columns:
            wins = (trades["return"] > 0).sum()
            win_rate = float(wins / len(trades))
            avg_pnl = float(trades["pnl"].mean()) if "pnl" in trades.columns else float("nan")

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "n_days": int(days),
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "total_return": float("nan"),
            "annual_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
            "avg_pnl": float("nan"),
            "n_days": 0,
        }
