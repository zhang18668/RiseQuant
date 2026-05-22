"""Optimize the limit-up pullback rule strategy.

Example:
    python scripts/optimize_limit_up_pullback.py --source cache --start 2022-01-01 --end 2025-12-31 --bt-start 2024-01-01 --bt-end 2025-12-31 --ignore-turnover-rate
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest.backtester import Backtester
from src.data.pipeline_loader import load_daily_data
from src.strategy.limit_up_pullback import LimitUpPullbackParams, select_limit_up_pullback
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger

setup_logger(log_level="INFO")
logger = get_logger(__name__)


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Optimize limit-up pullback selector")
    p.add_argument("--source", choices=["cache", "tdx", "auto"], default="cache")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bt-start", required=True)
    p.add_argument("--bt-end", required=True)
    p.add_argument("--out", default="./models/rule_limit_up_pullback")
    p.add_argument("--ignore-turnover-rate", action="store_true")
    p.add_argument("--limit-return-grid", default="0.095,0.098")
    p.add_argument("--confirm-return-max-grid", default="0.05,0.07")
    p.add_argument("--confirm-high-return-max-grid", default="0.085,0.095")
    p.add_argument("--turnover-min-grid", default="3")
    p.add_argument("--turnover-max-grid", default="10")
    p.add_argument("--volume-ratio-min-grid", default="1.0,1.2,1.5")
    p.add_argument("--topk-grid", default="1,2,3,5")
    p.add_argument("--sell-n-grid", default="2,3,5,8")
    p.add_argument("--stop-loss-grid", default="0.0,0.03,0.05")
    p.add_argument("--take-profit-grid", default="0.0,0.08,0.12")
    p.add_argument("--trailing-stop-grid", default="0.0,0.04,0.06")
    p.add_argument("--slippage", type=float, default=0.003)
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--entry-delay", type=int, default=0, help="0 means signal-day execution")
    p.add_argument("--entry-price", choices=["open", "close"], default="close",
                   help="default close = T tail buy for this rule strategy")
    p.add_argument("--min-sells", type=int, default=20)
    p.add_argument("--max-combos", type=int, default=0, help="0 means full grid")
    return p.parse_args()


def patch_config_dates(config, start: str, end: str):
    data = config.get_section("data")
    data["start_date"] = start
    data["end_date"] = end
    return config


def iter_grid(args) -> Iterable[Dict[str, Any]]:
    grids = {
        "limit_return_min": parse_float_list(args.limit_return_grid),
        "confirm_return_max": parse_float_list(args.confirm_return_max_grid),
        "confirm_high_return_max": parse_float_list(args.confirm_high_return_max_grid),
        "turnover_min": parse_float_list(args.turnover_min_grid),
        "turnover_max": parse_float_list(args.turnover_max_grid),
        "volume_ratio_min": parse_float_list(args.volume_ratio_min_grid),
        "topk": parse_int_list(args.topk_grid),
        "sell_n": parse_int_list(args.sell_n_grid),
        "stop_loss": parse_float_list(args.stop_loss_grid),
        "take_profit": parse_float_list(args.take_profit_grid),
        "trailing_stop": parse_float_list(args.trailing_stop_grid),
    }
    keys = list(grids)
    values = [grids[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_one(daily: pd.DataFrame, args, combo: Dict[str, Any]) -> Dict[str, Any]:
    rule_params = LimitUpPullbackParams(
        limit_return_min=combo["limit_return_min"],
        confirm_return_max=combo["confirm_return_max"],
        confirm_high_return_max=combo["confirm_high_return_max"],
        turnover_min=combo["turnover_min"],
        turnover_max=combo["turnover_max"],
        require_turnover_rate=not args.ignore_turnover_rate,
        require_market_filter=False,
        volume_ratio_min=combo["volume_ratio_min"],
    )
    signals = select_limit_up_pullback(daily, rule_params)
    signals_bt = signals[
        (signals["date"] >= pd.Timestamp(args.bt_start))
        & (signals["date"] <= pd.Timestamp(args.bt_end))
    ].copy()

    daily_bt = daily[
        (daily["date"] >= pd.Timestamp(args.bt_start))
        & (daily["date"] <= pd.Timestamp(args.bt_end))
    ].copy()
    bt = Backtester(
        initial_cash=args.initial_cash,
        topk=combo["topk"],
        sell_n=combo["sell_n"],
        slippage=args.slippage,
        entry_delay=args.entry_delay,
        entry_price=args.entry_price,
        stop_loss=combo["stop_loss"],
        take_profit=combo["take_profit"],
        trailing_stop=combo["trailing_stop"],
        min_score=0.0,
        skip_zhangting_open=True,
    )
    bt.run(signals_bt[["date", "code", "score"]], daily_bt)
    metrics = bt.get_metrics()
    trades = bt.get_trades()
    num_sells = int(trades["action"].astype(str).str.startswith("SELL").sum()) if len(trades) else 0
    return {
        **combo,
        **rule_params.to_dict(),
        "n_signals": int(len(signals_bt)),
        "num_sells": num_sells,
        "total_return": metrics.get("total_return"),
        "annual_return": metrics.get("annual_return"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate"),
    }


def rank_grid(grid: pd.DataFrame, min_sells: int) -> pd.DataFrame:
    if grid.empty:
        return grid
    ranked = grid.copy()
    ranked["sharpe_rank_value"] = ranked["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    ranked["drawdown_abs"] = ranked["max_drawdown"].abs().fillna(999.0)
    eligible = ranked[ranked["num_sells"] >= min_sells].copy()
    if eligible.empty:
        eligible = ranked
    eligible = eligible.sort_values(
        ["sharpe_rank_value", "annual_return", "total_return", "drawdown_abs"],
        ascending=[False, False, False, True],
    )
    return eligible


def save_best_run(daily: pd.DataFrame, args, best: Dict[str, Any], out_dir: Path) -> None:
    rule_params = LimitUpPullbackParams(
        limit_return_min=best["limit_return_min"],
        confirm_return_max=best["confirm_return_max"],
        confirm_high_return_max=best["confirm_high_return_max"],
        turnover_min=best["turnover_min"],
        turnover_max=best["turnover_max"],
        require_turnover_rate=not args.ignore_turnover_rate,
        require_market_filter=False,
        volume_ratio_min=best["volume_ratio_min"],
    )
    signals = select_limit_up_pullback(daily, rule_params)
    signals_bt = signals[
        (signals["date"] >= pd.Timestamp(args.bt_start))
        & (signals["date"] <= pd.Timestamp(args.bt_end))
    ].copy()
    daily_bt = daily[
        (daily["date"] >= pd.Timestamp(args.bt_start))
        & (daily["date"] <= pd.Timestamp(args.bt_end))
    ].copy()
    bt = Backtester(
        initial_cash=args.initial_cash,
        topk=int(best["topk"]),
        sell_n=int(best["sell_n"]),
        slippage=args.slippage,
        entry_delay=args.entry_delay,
        entry_price=args.entry_price,
        stop_loss=float(best["stop_loss"]),
        take_profit=float(best["take_profit"]),
        trailing_stop=float(best["trailing_stop"]),
        skip_zhangting_open=True,
    )
    bt.run(signals_bt[["date", "code", "score"]], daily_bt)
    signals_bt.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    bt.get_trades().to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    bt.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "backtest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(bt.get_metrics(), f, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    args = parse_args()
    config = patch_config_dates(get_config(), args.start, args.end)
    daily = load_daily_data(config, source=args.source)
    daily["date"] = pd.to_datetime(daily["date"])

    out_dir = Path(args.out) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, combo in enumerate(iter_grid(args), 1):
        if args.max_combos and i > args.max_combos:
            break
        rows.append(run_one(daily, args, combo))
        if i % 50 == 0:
            logger.info("optimized %s combos", i)

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "backtest_grid.csv", index=False, encoding="utf-8-sig")
    ranked = rank_grid(grid, args.min_sells)
    if ranked.empty:
        logger.warning("No grid result.")
        return 2
    best = ranked.iloc[0].to_dict()
    with (out_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2, default=str)
    save_best_run(daily, args, best, out_dir)
    logger.info("best params -> %s", out_dir / "best_params.json")
    logger.info(
        "best sharpe=%.3f annual=%.2f%% return=%.2f%% drawdown=%.2f%% sells=%s signals=%s",
        float(best.get("sharpe_ratio") or 0),
        float(best.get("annual_return") or 0) * 100,
        float(best.get("total_return") or 0) * 100,
        float(best.get("max_drawdown") or 0) * 100,
        int(best.get("num_sells") or 0),
        int(best.get("n_signals") or 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
