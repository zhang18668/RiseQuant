r"""Backtest a saved wash_second model on a custom date range.

Usage:
    cd D:\project\RiseQuant\limit_up_project
    python scripts\backtest_wash_second_model.py --run-dir models\wash_second\run_20260515_003557 --start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.backtester import Backtester
from src.data.tdx_loader import TDXDataLoader
from src.event.wash_sample_builder import WashSampleBuilder
from src.event.wash_second_detector import WashSecondDetector
from src.feature.market_features import MarketFeatures, market_proxy_from_daily
from src.feature.wash_features import WashFeatures
from src.model.model_trainer import LimitUpModelTrainer
from src.utils.logger import get_logger, setup_logger


setup_logger(log_level="INFO")
logger = get_logger(__name__)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_daily(tdx_dir: str, start: str, end: str) -> pd.DataFrame:
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    tdx = TDXDataLoader(tdx_dir)
    if tdx.data_path is None:
        raise FileNotFoundError(f"TDX vipdoc not found: {tdx_dir}")
    sh = tdx.load_stock_list("sh")
    sz = tdx.load_stock_list("sz")
    codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    logger.info(f"Loading {len(codes)} main-board stocks: {load_start} ~ {end}")
    daily = tdx.load_batch(codes=codes, start_date=load_start, end_date=end)
    if daily.empty:
        raise RuntimeError("daily data is empty")
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def build_features(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    logger.info("Detecting wash-second events and rolling samples")
    detector = WashSecondDetector(
        threshold=cfg.get("limit_up_threshold", 9.9),
        cooldown_days=cfg.get("cooldown_days", 3),
        min_gap=cfg.get("min_gap", 3),
        max_gap=cfg.get("max_gap", 30),
        exclude_st=cfg.get("exclude_st", True),
    )
    events = detector.detect(daily)

    builder = WashSampleBuilder(
        positive_window=cfg.get("positive_window", 5),
        min_gap_for_neg=cfg.get("min_gap", 3),
        max_gap_for_neg=cfg.get("max_gap", 30),
        threshold=cfg.get("limit_up_threshold", 9.9),
        cooldown_days=cfg.get("cooldown_days", 3),
    )
    samples = builder.build(daily, events=events, include_negatives=True)
    logger.info(f"Samples: {len(samples)}")
    if samples.empty:
        return pd.DataFrame()

    logger.info("Calculating wash and market features")
    wf = WashFeatures()
    daily_by_code = {
        c: g.sort_values("date").reset_index(drop=True)
        for c, g in daily.groupby("code")
    }
    rows = []
    for i, r in samples.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            continue
        feats = wf.calculate(sub, r["first_date"], r["potential_date"]).to_dict()
        feats["sample_id"] = r["sample_id"]
        feats["stop_loss_price"] = float(r["stop_loss_price"])
        rows.append(feats)
        if (i + 1) % 10000 == 0:
            logger.info(f"  feature progress: {i + 1}/{len(samples)}")

    features = pd.DataFrame(rows)
    if features.empty:
        return features

    market_daily = market_proxy_from_daily(daily)
    mf = MarketFeatures()
    features["potential_date"] = pd.to_datetime(features["potential_date"])
    unique_dates = features["potential_date"].drop_duplicates().sort_values()
    market_rows = [mf.calculate_at_event(d, market_daily) for d in unique_dates]
    market_features = pd.DataFrame(market_rows)
    if not market_features.empty:
        market_features["event_date"] = pd.to_datetime(market_features["event_date"])
        market_features = market_features.rename(columns={"event_date": "potential_date"})
        features = features.merge(market_features, on="potential_date", how="left")
    return features


def make_signals(features: pd.DataFrame, trainer: LimitUpModelTrainer, start: str, end: str) -> pd.DataFrame:
    feature_cols = trainer.feature_names_ or []
    x = features.reindex(columns=feature_cols, fill_value=0).fillna(0)
    proba = trainer.predict_proba(x)
    p_pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()

    signals = features[["potential_date", "stop_loss_price"]].copy()
    signals["code"] = features["sample_id"].astype(str).str.split("_").str[0]
    signals = signals.rename(columns={"potential_date": "date"})
    signals["date"] = pd.to_datetime(signals["date"])
    signals["score"] = p_pos
    signals = signals[
        (signals["date"] >= pd.Timestamp(start)) &
        (signals["date"] <= pd.Timestamp(end))
    ].reset_index(drop=True)
    return signals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Saved run dir, e.g. models/wash_second/run_20260515_003557")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--tdx", default=r"C:\new_tdx\vipdoc")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--sell-n", type=int, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--trailing-stop", type=float, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    cfg = load_json(run_dir / "config.json")
    best = load_json(run_dir / "best_backtest_params.json")
    params = {**cfg, **best}
    if args.min_score is not None:
        params["min_score"] = args.min_score
    if args.topk is not None:
        params["topk"] = args.topk
    if args.sell_n is not None:
        params["sell_n"] = args.sell_n
    if args.take_profit is not None:
        params["take_profit"] = args.take_profit
    if args.trailing_stop is not None:
        params["trailing_stop"] = args.trailing_stop

    trainer = LimitUpModelTrainer.load(str(run_dir / "model.pkl"))
    daily = load_daily(args.tdx, args.start, args.end)
    features = build_features(daily, cfg)
    signals = make_signals(features, trainer, args.start, args.end)
    daily_bt = daily[
        (daily["date"] >= pd.Timestamp(args.start)) &
        (daily["date"] <= pd.Timestamp(args.end))
    ].copy()

    logger.info(f"Backtest {args.start} ~ {args.end}, signals={len(signals)}")
    logger.info(
        "Params: min_score=%s topk=%s sell_n=%s take_profit=%s trailing_stop=%s",
        params.get("min_score"), params.get("topk"), params.get("sell_n"),
        params.get("take_profit"), params.get("trailing_stop"),
    )

    bt = Backtester(
        initial_cash=params.get("initial_cash", 10_000_000),
        topk=params.get("topk", 3),
        sell_n=params.get("sell_n", 2),
        slippage=params.get("slippage", 0.003),
        entry_delay=1,
        min_score=params.get("min_score", 0.68),
        stop_loss=0.0,
        take_profit=params.get("take_profit", 0.0),
        trailing_stop=params.get("trailing_stop", 0.04),
        skip_zhangting_open=True,
        cooldown_after_stop=params.get("cooldown_after_stop", 5),
    )
    bt.run(signals, daily_bt)
    metrics = bt.get_metrics()
    trades = bt.get_trades()

    out_dir = run_dir / f"backtest_{args.start}_{args.end}_{datetime.now().strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    bt.get_equity_curve().to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info(f"Total return   : {metrics.get('total_return', 0):.2%}")
    logger.info(f"Annual return  : {metrics.get('annual_return', 0):.2%}")
    logger.info(f"Sharpe         : {metrics.get('sharpe_ratio', float('nan')):.2f}")
    logger.info(f"Max drawdown   : {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"Trades         : {metrics.get('num_trades', 0)}")
    win_rate = metrics.get("win_rate")
    logger.info(f"Win rate       : {win_rate:.2%}" if win_rate == win_rate else "Win rate       : nan")
    if len(trades):
        sells = trades[trades["action"].astype(str).str.startswith("SELL")]
        logger.info(f"Sell reasons:\n{sells['action'].value_counts().to_string()}")
    logger.info(f"Output written : {out_dir}")


if __name__ == "__main__":
    main()
