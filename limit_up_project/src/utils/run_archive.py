"""U-005 运行档案

把一次"训练 + 回测"的全部产物存进同一个目录, 便于复盘 / 对比 / 部署:

    models/<strategy>/run_<YYYYmmdd_HHMMSS>/
        model.pkl                  — LightGBMClassifier 序列化
        config.json                — 全部超参与数据范围 (用于完全复现)
        train_metrics.json         — train/valid/test 集分类指标
        backtest_metrics.json      — 总收益/夏普/回撤/胜率/卖出原因分布
        feature_importance.csv     — 全部特征重要性
        equity_curve.csv           — 回测每日净值
        trades.csv                 — 回测交易明细 (含卖出原因)
        signals.csv (可选)         — 模型对回测期间所有样本的预测分

同时维护:
    models/<strategy>/latest/      — 软拷贝最近一次 run, 包含 model.pkl, 用于 predict 脚本

用法
----
::

    arch = RunArchive("models", "wash_second")
    arch.save_config(full_cfg)
    arch.save_model(trainer)
    arch.save_train_metrics(metrics)
    arch.save_feature_importance(trainer)
    arch.save_backtest(backtester, signals_df=signals)
    print(arch.summary())
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _json_default(o: Any):
    """让 json.dump 能吃 datetime / Timestamp / numpy 标量."""
    if isinstance(o, (datetime, pd.Timestamp)):
        return pd.Timestamp(o).isoformat()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


@dataclass
class RunArchive:
    """一次训练 + 回测的产物归档器."""

    base_dir: str
    strategy: str
    run_id: Optional[str] = None   # 默认用当前时间戳

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.base_dir) / self.strategy / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir = Path(self.base_dir) / self.strategy / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self._summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "run_dir": str(self.run_dir),
            "artifacts": [],
        }

    # ------------------------------------------------------------------
    def save_config(self, config: Dict[str, Any]) -> None:
        p = self.run_dir / "config.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)
        self._summary["artifacts"].append("config.json")

    # ------------------------------------------------------------------
    def save_model(self, trainer) -> None:
        p_run = self.run_dir / "model.pkl"
        p_latest = self.latest_dir / "model.pkl"
        trainer.save(str(p_run))
        trainer.save(str(p_latest))
        self._summary["artifacts"].append("model.pkl")
        # 同时把 config 拷一份到 latest, 方便 predict 脚本直接拿超参
        cfg_src = self.run_dir / "config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, self.latest_dir / "config.json")

    # ------------------------------------------------------------------
    def save_train_metrics(
        self,
        metrics: Dict[str, Any],
        split_sizes: Optional[Dict[str, int]] = None,
        split_ranges: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "metrics": metrics,
            "split_sizes": split_sizes or {},
            "split_ranges": split_ranges or {},
        }
        p = self.run_dir / "train_metrics.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        self._summary["artifacts"].append("train_metrics.json")
        self._summary["train_metrics"] = metrics

    # ------------------------------------------------------------------
    def save_feature_importance(self, trainer, top_n: Optional[int] = None) -> None:
        imp = trainer.get_feature_importance(top_n=top_n)
        p = self.run_dir / "feature_importance.csv"
        imp.to_csv(p, index=False, encoding="utf-8-sig")
        self._summary["artifacts"].append("feature_importance.csv")

    # ------------------------------------------------------------------
    def save_backtest(
        self,
        backtester,
        signals_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """保存回测的全部产物, 返回 backtest_metrics dict."""
        metrics = dict(backtester.get_metrics())
        trades = backtester.get_trades()
        equity = backtester.equity_curve.copy() if backtester.equity_curve is not None else pd.DataFrame()

        # 卖出原因分布
        sell_reasons: Dict[str, int] = {}
        if len(trades):
            r = trades[trades["action"].astype(str).str.startswith("SELL")]
            sell_reasons = r["action"].value_counts().to_dict()
        metrics["sell_reasons"] = sell_reasons

        # 盈亏分布 (按卖出交易)
        if len(trades):
            sell_trades = trades[trades["action"].astype(str).str.startswith("SELL")]
            if len(sell_trades):
                metrics["sell_avg_return"] = float(sell_trades["return_pct"].mean())
                metrics["sell_median_return"] = float(sell_trades["return_pct"].median())
                metrics["sell_max_loss"] = float(sell_trades["return_pct"].min())
                metrics["sell_max_gain"] = float(sell_trades["return_pct"].max())

        with open(self.run_dir / "backtest_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)
        self._summary["artifacts"].append("backtest_metrics.json")
        self._summary["backtest_metrics"] = metrics

        if len(equity):
            equity.to_csv(self.run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
            self._summary["artifacts"].append("equity_curve.csv")
        if len(trades):
            trades.to_csv(self.run_dir / "trades.csv", index=False, encoding="utf-8-sig")
            self._summary["artifacts"].append("trades.csv")
        if signals_df is not None and len(signals_df):
            signals_df.to_csv(self.run_dir / "signals.csv", index=False, encoding="utf-8-sig")
            self._summary["artifacts"].append("signals.csv")
        return metrics

    # ------------------------------------------------------------------
    def write_summary(self) -> Path:
        """最后一步: 写出 run summary, 方便 ls 一眼看出每个 run 的结果."""
        p = self.run_dir / "summary.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._summary, f, ensure_ascii=False, indent=2, default=_json_default)
        # 同步到 latest/summary.json
        try:
            shutil.copy2(p, self.latest_dir / "summary.json")
        except Exception:
            pass
        return p

    def summary(self) -> Dict[str, Any]:
        return dict(self._summary)
