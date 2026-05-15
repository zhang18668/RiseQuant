r"""Pattern-Cluster v2 backtest pipeline (decoupled from training).

Usage (Windows, single-line commands):

    set PYTHONPATH=%CD%

    # plan A, full year
    python scripts\wash_pattern_backtest.py --bundle models\pattern_cluster\latest --bundle-type per_cluster --bt-start 2024-01-01 --bt-end 2024-12-31

    # plan B
    python scripts\wash_pattern_backtest.py --bundle models\pattern_cluster\latest --bundle-type single_with_cf --bt-start 2024-01-01 --bt-end 2024-12-31

    # restrict clusters + raise threshold
    python scripts\wash_pattern_backtest.py --bundle models\pattern_cluster\latest --bundle-type per_cluster --clusters 0,2,4 --min-score 0.7

    # walk-forward bundle (auto-route by date)
    python scripts\wash_pattern_backtest.py --bundle-dir models\pattern_cluster\walkforward_2026Q1 --bundle-type per_cluster --bt-start 2024-01-01 --bt-end 2025-04-30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest.backtester import Backtester
from src.data.tdx_loader import TDXDataLoader
from src.event.wash_sample_builder import WashSampleBuilder
from src.event.wash_second_detector import WashSecondDetector
from src.feature.market_features import MarketFeatures, market_proxy_from_daily
from src.feature.wash_features import WashFeatures
from src.model.model_bundle import ModelBundle
from src.pattern.pattern_router import PatternRouter
from src.pattern.sequence_extractor import SequenceExtractor
from src.utils.logger import get_logger, setup_logger

setup_logger(log_level="INFO")
logger = get_logger(__name__)


# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pattern-Cluster v2 回测")
    g_bundle = p.add_mutually_exclusive_group(required=True)
    g_bundle.add_argument("--bundle", help="单 bundle 目录")
    g_bundle.add_argument("--bundle-dir", help="walk-forward 目录 (含 schedule.json)")
    p.add_argument("--bundle-type", choices=["per_cluster", "single_with_cf"],
                   default="per_cluster")
    p.add_argument("--bt-start", required=True)
    p.add_argument("--bt-end", required=True)
    p.add_argument("--tdx-path", default=r"C:\new_tdx\vipdoc")
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--sell-n", type=int, default=8)
    p.add_argument("--slippage", type=float, default=0.003)
    p.add_argument("--min-score", type=float, default=0.6)
    p.add_argument("--min-score-per-cluster", default="",
                   help="JSON 文件路径, e.g. {\"0\": 0.65, \"1\": 0.55}")
    p.add_argument("--cooldown", type=int, default=5)
    p.add_argument("--clusters", default="",
                   help="逗号分隔 cluster_id, 仅在此列表中的 cluster 进入回测")
    p.add_argument("--code-list", default="",
                   help="逗号分隔股票代码; 不指定则用全主板")
    p.add_argument("--max-gap", type=int, default=30)
    p.add_argument("--positive-window", type=int, default=5)
    p.add_argument("--out", default="./backtests",
                   help="回测产物根目录")
    return p.parse_args()


# ============================================================
def load_min_score_per_cluster(path: str) -> Optional[Dict[int, float]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"min-score-per-cluster 文件不存在: {p}")
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): float(v) for k, v in data.items()}


# ============================================================
def load_daily(args, code_list: Optional[List[str]] = None) -> pd.DataFrame:
    tdx = TDXDataLoader(args.tdx_path)
    if tdx.data_path is None:
        logger.error(f"找不到通达信目录: {args.tdx_path}")
        sys.exit(2)
    if code_list:
        codes = code_list
    else:
        sh = tdx.load_stock_list("sh")
        sz = tdx.load_stock_list("sz")
        codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    # 回测窗口前多取 60 天给特征用
    bt_start_with_buffer = (pd.Timestamp(args.bt_start) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    daily = tdx.load_batch(codes=codes, start_date=bt_start_with_buffer, end_date=args.bt_end)
    if daily.empty:
        sys.exit("日线为空")
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ============================================================
def build_features_and_route(
    daily: pd.DataFrame, args, router: PatternRouter,
) -> pd.DataFrame:
    """重做 detector + sample_builder + features + router."""
    logger.info("回测前置: detector + sample_builder + features + router")
    det = WashSecondDetector(max_gap=args.max_gap)
    events = det.detect(daily)
    if events.empty:
        return pd.DataFrame()
    builder = WashSampleBuilder(
        positive_window=args.positive_window,
        min_gap_for_neg=3, max_gap_for_neg=args.max_gap,
    )
    samples = builder.build(daily, events=events, include_negatives=True)
    if samples.empty:
        return pd.DataFrame()

    # 算特征
    wf = WashFeatures()
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in daily.groupby("code")}
    rows = []
    for _, r in samples.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            continue
        feats = wf.calculate(sub, r["first_date"], r["potential_date"]).to_dict()
        feats["sample_id"] = r["sample_id"]
        feats["potential_date"] = r["potential_date"]
        feats["code"] = r["code"]
        feats["stop_loss_price"] = float(r["stop_loss_price"])
        rows.append(feats)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    market_data = market_proxy_from_daily(daily)
    mf = MarketFeatures()
    df["potential_date"] = pd.to_datetime(df["potential_date"])
    unique_dates = df["potential_date"].drop_duplicates().sort_values()
    mkt_df = pd.DataFrame([mf.calculate_at_event(d, market_data) for d in unique_dates])
    if not mkt_df.empty:
        mkt_df["event_date"] = pd.to_datetime(mkt_df["event_date"])
        mkt_df = mkt_df.rename(columns={"event_date": "potential_date"})
        df = df.merge(mkt_df, on="potential_date", how="left")

    # 路由
    ex = SequenceExtractor()
    cluster_ids, dtw_dists = [], []
    for _, r in df.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            cluster_ids.append(-1); dtw_dists.append(float("inf"))
            continue
        s = ex.extract(sub, samples.loc[samples["sample_id"] == r["sample_id"], "first_date"].iloc[0],
                       r["potential_date"])
        if s is None:
            cluster_ids.append(-1); dtw_dists.append(float("inf"))
            continue
        cid, d = router.route(s)
        cluster_ids.append(cid); dtw_dists.append(d)
    df["cluster_id"] = cluster_ids
    df["dtw_dist"] = dtw_dists
    return df


# ============================================================
def make_signals(
    features_df: pd.DataFrame, bundle: ModelBundle, args,
    min_score_per_cluster: Optional[Dict[int, float]],
    cluster_filter: Optional[List[int]],
) -> pd.DataFrame:
    """打分 + 阈值过滤 + 去重."""
    if features_df.empty:
        return pd.DataFrame()
    cluster_ids = features_df["cluster_id"].astype(int)
    scores = bundle.predict(features_df, cluster_ids=cluster_ids)
    df = features_df[["potential_date", "code", "cluster_id", "stop_loss_price"]].copy()
    df["date"] = df["potential_date"]
    df["score"] = scores

    # cluster 过滤
    df = df[df["cluster_id"] >= 0]
    if cluster_filter:
        df = df[df["cluster_id"].isin(cluster_filter)]

    # 阈值过滤 (per-cluster 优先, fallback 全局)
    keep = []
    for _, r in df.iterrows():
        threshold = args.min_score
        if min_score_per_cluster and int(r["cluster_id"]) in min_score_per_cluster:
            threshold = float(min_score_per_cluster[int(r["cluster_id"])])
        keep.append(r["score"] >= threshold)
    df = df[pd.Series(keep, index=df.index)]

    # 同日同 code 取 score 最高
    df = df.sort_values("score", ascending=False).drop_duplicates(["date", "code"], keep="first")
    return df[["date", "code", "cluster_id", "score", "stop_loss_price"]].sort_values(["date", "code"])


# ============================================================
def run_backtest(signals: pd.DataFrame, daily: pd.DataFrame, args) -> Backtester:
    bt = Backtester(
        initial_cash=args.initial_cash,
        topk=args.topk,
        sell_n=args.sell_n,
        slippage=args.slippage,
        entry_delay=1,
        min_score=args.min_score,
        stop_loss=0.0, take_profit=0.0, trailing_stop=0.0,
        skip_zhangting_open=True,
        cooldown_after_stop=args.cooldown,
    )
    bt.run(signals, daily)
    return bt


# ============================================================
def write_backtest_artifacts(
    bt: Backtester, signals: pd.DataFrame, out_dir: Path, run_cfg: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(bt.get_metrics())
    trades = bt.get_trades()
    if len(trades):
        sell = trades[trades["action"].astype(str).str.startswith("SELL")]
        metrics["sell_reasons"] = sell["action"].value_counts().to_dict()
        if len(sell):
            metrics["sell_avg_return"] = float(sell["return_pct"].mean())
            metrics["sell_median_return"] = float(sell["return_pct"].median())

    # per-cluster 拆分
    per_cluster = {}
    if len(trades) and len(signals):
        # 合并 cluster_id
        buy = trades[trades["action"] == "BUY"][["date", "code"]].copy()
        sig = signals[["date", "code", "cluster_id"]].copy()
        sig["date"] = pd.to_datetime(sig["date"])
        buy["date"] = pd.to_datetime(buy["date"])
        # 信号日期 + 1 (T+1 撮合)
        sig["match_date"] = sig["date"] + pd.Timedelta(days=1)
        # 用 code + 临近日期对齐 (允许 +/- 5 天)
        merged = pd.merge_asof(
            buy.sort_values("date"),
            sig.sort_values("match_date").rename(columns={"match_date": "_md"}),
            left_on="date", right_on="_md", by="code", direction="backward",
            tolerance=pd.Timedelta(days=5),
        )
        trades_with_cid = trades.merge(
            merged[["date", "code", "cluster_id"]], on=["date", "code"], how="left",
        )
        for cid, g in trades_with_cid.dropna(subset=["cluster_id"]).groupby("cluster_id"):
            sell = g[g["action"].astype(str).str.startswith("SELL")]
            per_cluster[int(cid)] = {
                "n_trades": int(len(g[g["action"] == "BUY"])),
                "win_rate": float((sell["return_pct"] > 0).mean()) if len(sell) else None,
                "avg_return": float(sell["return_pct"].mean()) if len(sell) else None,
                "sell_reasons": sell["action"].value_counts().to_dict() if len(sell) else {},
            }
    metrics["per_cluster"] = per_cluster

    with open(out_dir / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2, default=str)
    if len(signals):
        signals.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    if len(trades):
        trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    if bt.equity_curve is not None and len(bt.equity_curve):
        bt.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_dir": str(out_dir),
        "total_return": metrics.get("total_return"),
        "annual_return": metrics.get("annual_return"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate"),
        "num_trades": metrics.get("num_trades"),
        "per_cluster": per_cluster,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("=" * 50)
    logger.info(f"总收益率 : {metrics.get('total_return', 0):.2%}")
    logger.info(f"年化收益 : {metrics.get('annual_return', 0):.2%}")
    logger.info(f"夏普     : {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"最大回撤 : {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"交易数   : {metrics.get('num_trades', 0)}")
    logger.info(f"产物目录 : {out_dir}")


# ============================================================
def main():
    args = parse_args()
    code_list = [c.strip() for c in args.code_list.split(",") if c.strip()] or None
    cluster_filter = ([int(c.strip()) for c in args.clusters.split(",") if c.strip()]
                      if args.clusters else None)
    min_score_per_cluster = load_min_score_per_cluster(args.min_score_per_cluster)

    # 单 bundle 或多 bundle (walk-forward)
    bundle_paths: List[Path]
    if args.bundle:
        bundle_paths = [Path(args.bundle)]
        walkforward = False
    else:
        bd = Path(args.bundle_dir)
        schedule_p = bd / "schedule.json"
        if not schedule_p.exists():
            logger.error(f"{schedule_p} 不存在")
            sys.exit(2)
        with open(schedule_p, encoding="utf-8") as f:
            schedule = json.load(f)
        bundle_paths = [Path(s["bundle_dir"]) for s in schedule]
        walkforward = True

    # 加载所有 bundle (walk-forward 时按日期切片各跑)
    daily = load_daily(args, code_list=code_list)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_cfg = vars(args).copy()
    run_cfg["run_id"] = run_id

    if not walkforward:
        bundle = ModelBundle.load(bundle_paths[0], bundle_type=args.bundle_type)
        # 裁回测窗口
        d = daily[(daily["date"] >= pd.Timestamp(args.bt_start)) &
                  (daily["date"] <= pd.Timestamp(args.bt_end))].copy()
        # 但 detector 需要看 first 在 bt 窗口之前的事件 — 用全量 daily 跑 detector,
        # 然后只对落在 bt 窗口的 potential_date 出信号
        feats = build_features_and_route(daily, args, bundle.router)
        feats = feats[(feats["potential_date"] >= pd.Timestamp(args.bt_start)) &
                       (feats["potential_date"] <= pd.Timestamp(args.bt_end))]
        signals = make_signals(feats, bundle, args, min_score_per_cluster, cluster_filter)
        bt = run_backtest(signals, d, args)
        out_dir = (Path(args.out) / bundle_paths[0].name /
                   args.bundle_type / f"run_{run_id}")
        write_backtest_artifacts(bt, signals, out_dir, run_cfg)
    else:
        all_signals = []
        for sch_item in schedule:
            bundle_dir = Path(sch_item["bundle_dir"])
            t_start = pd.Timestamp(sch_item["test_start"])
            t_end = pd.Timestamp(sch_item["test_end"])
            t_lo = max(t_start, pd.Timestamp(args.bt_start))
            t_hi = min(t_end, pd.Timestamp(args.bt_end))
            if t_lo > t_hi:
                continue
            logger.info(f"--- 窗口 {sch_item['window_id']}: {t_lo.date()} ~ {t_hi.date()} ---")
            bundle = ModelBundle.load(bundle_dir, bundle_type=args.bundle_type)
            feats = build_features_and_route(daily, args, bundle.router)
            feats = feats[(feats["potential_date"] >= t_lo) & (feats["potential_date"] <= t_hi)]
            sig = make_signals(feats, bundle, args, min_score_per_cluster, cluster_filter)
            sig["bundle_window"] = sch_item["window_id"]
            all_signals.append(sig)
        signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
        d = daily[(daily["date"] >= pd.Timestamp(args.bt_start)) &
                  (daily["date"] <= pd.Timestamp(args.bt_end))].copy()
        bt = run_backtest(signals, d, args)
        out_dir = (Path(args.out) / Path(args.bundle_dir).name /
                   args.bundle_type / f"run_{run_id}")
        write_backtest_artifacts(bt, signals, out_dir, run_cfg)


if __name__ == "__main__":
    main()
