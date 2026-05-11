"""B-003 回测报告生成

输出一个简短的 markdown / 字典报告，把 :class:`PerformanceAnalyzer` 的
指标以及交易明细整理在一起。
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


class ReportGenerator:
    """回测报告生成器。"""

    def to_markdown(
        self,
        metrics: Dict[str, float],
        trades: Optional[pd.DataFrame] = None,
    ) -> str:
        lines = ["# 回测报告", ""]
        lines.append("## 业绩指标")
        for key in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown",
                    "win_rate", "avg_pnl", "n_days"):
            v = metrics.get(key)
            if v is None:
                continue
            if isinstance(v, float) and not pd.isna(v):
                if key in ("total_return", "annual_return", "max_drawdown", "win_rate"):
                    lines.append(f"- **{key}**: {v:.2%}")
                else:
                    lines.append(f"- **{key}**: {v:.4f}")
            else:
                lines.append(f"- **{key}**: {v}")

        if trades is not None and not trades.empty:
            lines.append("")
            lines.append(f"## 交易明细 ({len(trades)} 笔)")
            lines.append(trades.head(10).to_markdown(index=False))
        return "\n".join(lines)

    def to_dict(
        self,
        metrics: Dict[str, float],
        trades: Optional[pd.DataFrame] = None,
    ) -> Dict[str, object]:
        return {
            "metrics": metrics,
            "n_trades": int(len(trades)) if trades is not None else 0,
            "trades": trades.to_dict("records") if (trades is not None and not trades.empty) else [],
        }
