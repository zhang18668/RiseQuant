r"""Pattern-Cluster v2 training pipeline.

Usage (Windows, single-line commands):

    set PYTHONPATH=%CD%

    # single run
    python scripts\wash_pattern_train.py --start 2022-01-01 --end 2024-12-31 --train-end 2023-12-31 --valid-end 2024-06-30 --test-end 2024-12-31 --golden-lo 0.15 --golden-hi 0.35 --k-range 3 8 --bundle-type both --out models\pattern_cluster

    # walk-forward
    python scripts\wash_pattern_train.py --start 2022-01-01 --end 2025-04-30 --walk-forward-months 6 --min-train-months 12 --golden-lo 0.15 --golden-hi 0.35 --bundle-type both --out models\pattern_cluster\walkforward_2026Q1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.tdx_loader import TDXDataLoader
from src.event.data_dryrun import DataDryRun
from src.event.golden_label_filter import GoldenLabelFilter
from src.event.wash_sample_builder import WashSampleBuilder
from src.event.wash_second_detector import WashSecondDetector
from src.feature.market_features import MarketFeatures, market_proxy_from_daily
from src.feature.wash_features import WashFeatures
from src.model.cluster_model_trainer import ClusterModelTrainer
from src.model.model_bundle import BundleArchive
from src.model.single_with_cf_trainer import SingleWithCFTrainer
from src.pattern.pattern_clusterer import PatternClusterer
from src.pattern.pattern_router import PatternRouter
from src.pattern.sequence_extractor import SequenceExtractor
from src.pattern.visualizer import (
    plot_calibration,
    plot_cluster_year_distribution,
    plot_feature_importance_per_cluster,
    plot_prototypes,
)
from src.utils.logger import get_logger, setup_logger
from src.utils.walkforward import WalkForwardScheduler

setup_logger(log_level="INFO")
logger = get_logger(__name__)


# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pattern-Cluster v2 训练 pipeline")
    p.add_argument("--start", required=True, help="数据起始 YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="数据结束 YYYY-MM-DD")
    p.add_argument("--tdx-path", default=r"C:\new_tdx\vipdoc")
    p.add_argument("--train-end", default=None)
    p.add_argument("--valid-end", default=None)
    p.add_argument("--test-end",  default=None)
    p.add_argument("--golden-lo", type=float, default=0.15)
    p.add_argument("--golden-hi", type=float, default=0.35)
    p.add_argument("--golden-horizon", type=int, default=22)
    p.add_argument("--min-golden-events", type=int, default=300)
    p.add_argument("--k-range", type=int, nargs=2, default=[3, 8])
    p.add_argument("--k-fixed", type=int, default=None)
    p.add_argument("--bundle-type", choices=["per_cluster", "single_with_cf", "both"],
                   default="both")
    p.add_argument("--max-gap", type=int, default=30)
    p.add_argument("--positive-window", type=int, default=5)
    p.add_argument("--min-train-samples-per-cluster", type=int, default=80)
    p.add_argument("--min-auc-test", type=float, default=0.55)
    p.add_argument("--walk-forward-months", type=int, default=0,
                   help="walk-forward 步长 (月), 0=禁用")
    p.add_argument("--min-train-months", type=int, default=12)
    p.add_argument("--out", default="./models/pattern_cluster",
                   help="bundle 输出根目录")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--save-training-data", action="store_true",
                   help="把训练用 samples + features 也存进 shared/ (可追溯, 但占空间)")
    p.add_argument("--skip-dryrun-abort", action="store_true",
                   help="即使 dry-run 不达标也继续训练")
    return p.parse_args()


# ============================================================
def load_daily(args) -> pd.DataFrame:
    logger.info("加载日线数据 …")
    tdx = TDXDataLoader(args.tdx_path)
    if tdx.data_path is None:
        logger.error(f"找不到通达信目录: {args.tdx_path}")
        sys.exit(2)
    sh = tdx.load_stock_list("sh")
    sz = tdx.load_stock_list("sz")
    codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    logger.info(f"加载 {len(codes)} 只主板股票")
    daily = tdx.load_batch(codes=codes, start_date=args.start, end_date=args.end)
    if daily.empty:
        logger.error("日线数据为空")
        sys.exit(2)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ============================================================
def detect_and_label(
    daily: pd.DataFrame, args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1+2: WashSecondDetector + GoldenLabelFilter."""
    logger.info("Step 1: WashSecondDetector")
    det = WashSecondDetector(max_gap=args.max_gap)
    events = det.detect(daily)
    logger.info(f"  配对事件 {len(events)} 条")
    if events.empty:
        sys.exit("无配对事件, 退出")

    logger.info("Step 2: GoldenLabelFilter")
    flt = GoldenLabelFilter(lo=args.golden_lo, hi=args.golden_hi,
                            horizon=args.golden_horizon)
    events = flt.filter(events, daily)
    n_gold = int(events["is_golden"].fillna(False).astype(bool).sum())
    logger.info(f"  金标准事件 {n_gold} 条 (区间 [{args.golden_lo}, {args.golden_hi}])")
    return events, daily


# ============================================================
def build_samples(events: pd.DataFrame, daily: pd.DataFrame, args) -> pd.DataFrame:
    """Step 3: WashSampleBuilder + 标 is_golden_event."""
    logger.info("Step 3: WashSampleBuilder")
    builder = WashSampleBuilder(
        positive_window=args.positive_window,
        min_gap_for_neg=3,
        max_gap_for_neg=args.max_gap,
    )
    samples = builder.build(daily, events=events, include_negatives=True)

    # is_golden_event 透传
    golden_keys = set()
    for _, ev in events.iterrows():
        if bool(ev.get("is_golden", False)):
            golden_keys.add((str(ev["code"]), pd.Timestamp(ev["first_date"])))
    samples["is_golden_event"] = samples.apply(
        lambda r: (str(r["code"]), pd.Timestamp(r["first_date"])) in golden_keys, axis=1,
    )
    logger.info(f"  全量样本 {len(samples)} 条, 来自金标准事件的 {samples['is_golden_event'].sum()} 条")
    return samples


# ============================================================
def compute_features(samples: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Step 8: 算 wash + market features (复用现有模块)."""
    logger.info("Step 8: 特征工程 (WashFeatures + MarketFeatures)")
    wf = WashFeatures()
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in daily.groupby("code")}
    rows = []
    for i, r in samples.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            continue
        feats = wf.calculate(sub, r["first_date"], r["potential_date"]).to_dict()
        feats["sample_id"]      = r["sample_id"]
        feats["label_pre"]      = int(r["label_pre"])
        feats["sample_weight"]  = float(r["sample_weight"])
        feats["stop_loss_price"]= float(r["stop_loss_price"])
        feats["first_open"]     = float(r["first_open"])
        rows.append(feats)
        if (i + 1) % 2000 == 0:
            logger.info(f"  wash feature progress: {i+1}/{len(samples)}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    market_data = market_proxy_from_daily(daily)
    mf = MarketFeatures()
    df["potential_date"] = pd.to_datetime(df["potential_date"])
    unique_dates = df["potential_date"].drop_duplicates().sort_values()
    mkt_rows = [mf.calculate_at_event(d, market_data) for d in unique_dates]
    mkt_df = pd.DataFrame(mkt_rows)
    if not mkt_df.empty:
        mkt_df["event_date"] = pd.to_datetime(mkt_df["event_date"])
        mkt_df = mkt_df.rename(columns={"event_date": "potential_date"})
        df = df.merge(mkt_df, on="potential_date", how="left")

    n_wash = sum(1 for c in df.columns if c.startswith("f_w_"))
    n_mkt  = sum(1 for c in df.columns if c.startswith("f_mkt_"))
    logger.info(f"  特征表 {len(df)} 行 × ({n_wash} wash + {n_mkt} market) = {n_wash + n_mkt} 维")
    return df


# ============================================================
def fit_clusterer(
    samples: pd.DataFrame, daily: pd.DataFrame, args,
    train_end: pd.Timestamp,
) -> Tuple[PatternClusterer, "ClusterFitResult", List[np.ndarray]]:
    """Step 4-6: 抽 train 切片的金标准事件代表序列 + 聚类."""
    logger.info("Step 4-6: 形态聚类")
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in daily.groupby("code")}

    ex = SequenceExtractor()
    # 每个金标准事件取距 second 最近的正样本日作为代表
    # 这一步在 train_end 之前的样本中找
    train_samples = samples[
        (samples["is_golden_event"]) &
        (samples["label_pre"] == 1) &
        (pd.to_datetime(samples["potential_date"]) <= train_end)
    ].copy()
    # 每个 (code, first_date) 取 days_to_second 最小的 (距 second 最近) 那条
    train_samples = train_samples.sort_values(["code", "first_date", "days_to_second"])
    rep = train_samples.drop_duplicates(["code", "first_date"], keep="first")
    logger.info(f"  train 切片代表序列: {len(rep)} 条")

    seqs: List[np.ndarray] = []
    keys: List[str] = []
    for _, r in rep.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            continue
        s = ex.extract(sub, r["first_date"], r["potential_date"])
        if s is None:
            continue
        seqs.append(s)
        keys.append(r["sample_id"])

    if len(seqs) < 10:
        sys.exit(f"代表序列数 {len(seqs)} 过少, 无法聚类")

    k_fixed = args.k_fixed
    k_range = tuple(args.k_range)
    pc = PatternClusterer(
        k_range=k_range, k_fixed=k_fixed,
        kmedoids_init_n=10, route_dist_quantile=0.95,
    )
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    logger.info(f"  best_k={res.best_k}, silhouette={res.silhouette:.4f}")
    for c, n in res.cluster_sizes.items():
        logger.info(f"    cluster {c}: {n} 个")
    return pc, res, seqs


# ============================================================
def route_all_samples(
    samples: pd.DataFrame, daily: pd.DataFrame, router: PatternRouter,
) -> pd.DataFrame:
    """Step 7: 对全量样本打 cluster_id."""
    logger.info("Step 7: PatternRouter.route_samples (全量)")
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in daily.groupby("code")}
    ex = SequenceExtractor()
    cluster_ids = []
    dtw_dists = []
    for i, r in samples.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            cluster_ids.append(-1)
            dtw_dists.append(float("inf"))
            continue
        s = ex.extract(sub, r["first_date"], r["potential_date"])
        if s is None:
            cluster_ids.append(-1)
            dtw_dists.append(float("inf"))
            continue
        cid, d = router.route(s)
        cluster_ids.append(cid)
        dtw_dists.append(d)
        if (i + 1) % 2000 == 0:
            logger.info(f"  route progress: {i+1}/{len(samples)}")
    samples = samples.copy()
    samples["cluster_id"] = cluster_ids
    samples["dtw_dist"] = dtw_dists
    n_known = int((samples["cluster_id"] >= 0).sum())
    logger.info(f"  cluster_id ≥ 0 的样本: {n_known} / {len(samples)}")
    return samples


# ============================================================
def train_one_window(
    daily: pd.DataFrame, args,
    train_end: pd.Timestamp, valid_end: pd.Timestamp, test_end: pd.Timestamp,
    out_root: Path, window_id: Optional[str] = None,
) -> Path:
    """对单个时间窗口跑完整 pipeline, 输出一个 bundle 目录."""
    run_id = (datetime.now().strftime("%Y%m%d_%H%M%S")
              if not window_id else f"window_{window_id}")
    arch = BundleArchive(base_dir=str(out_root), bundle_name="", run_id=run_id)
    # 把 base_dir/""/ 调整成 base_dir/  — BundleArchive 默认 base/bundle_name/run_X
    # 这里为 walk-forward 我们想要 out_root/window_<id>/ , 简单做法: 直接重置
    arch.run_dir = out_root / (f"window_{window_id}" if window_id else f"run_{run_id}")
    arch.run_dir.mkdir(parents=True, exist_ok=True)
    arch.latest_dir = out_root / "latest"

    logger.info(f"=== 训练窗口: train_end={train_end}, valid_end={valid_end}, test_end={test_end} ===")

    events, daily = detect_and_label(daily, args)
    samples = build_samples(events, daily, args)
    features = compute_features(samples, daily)
    if features.empty:
        logger.error("特征为空, 退出")
        return arch.run_dir

    pc, fit_res, _ = fit_clusterer(samples, daily, args, train_end=train_end)
    router = PatternRouter.from_fit_result(fit_res, dtw_band=pc.dtw_band)
    samples = route_all_samples(samples, daily, router)

    # cluster_stats
    stats_rows = []
    for cid in sorted([c for c in samples["cluster_id"].unique() if c >= 0]):
        sub = samples[samples["cluster_id"] == cid]
        stats_rows.append({
            "cluster_id": int(cid),
            "n_samples": int(len(sub)),
            "n_positive": int(sub["label_pre"].sum()),
            "n_golden_event_samples": int(sub["is_golden_event"].sum()),
        })
    cluster_stats = pd.DataFrame(stats_rows)

    feature_cols = [c for c in features.columns
                    if c.startswith("f_w_") or c.startswith("f_mkt_")]

    # 训练
    model_params = {"n_estimators": args.n_estimators, "verbose": -1}
    cluster_artifacts: Dict[int, Any] = {}
    single_artifact = None

    if args.bundle_type in ("per_cluster", "both"):
        logger.info("Step 9-A: ClusterModelTrainer (方案 A)")
        cmt = ClusterModelTrainer(
            min_train_samples=args.min_train_samples_per_cluster,
            min_auc_test=args.min_auc_test,
            model_params=model_params,
            feature_cols=feature_cols,
        )
        cluster_artifacts = cmt.train(
            samples, features,
            train_end=train_end, valid_end=valid_end, test_end=test_end,
        )

    if args.bundle_type in ("single_with_cf", "both"):
        logger.info("Step 9-B: SingleWithCFTrainer (方案 B)")
        swcf = SingleWithCFTrainer(
            model_params=model_params,
            feature_cols=feature_cols,
        )
        single_artifact = swcf.train(
            samples, features,
            train_end=train_end, valid_end=valid_end, test_end=test_end,
        )

    # 保存
    logger.info("Step 10: 写入 bundle 产物")
    arch.save_shared(
        feature_cols=feature_cols,
        router=router,
        cluster_stats=cluster_stats,
        training_data_range=[str(pd.Timestamp(args.start).date()),
                              str(pd.Timestamp(train_end).date())],
        training_config={
            "max_gap": args.max_gap,
            "positive_window": args.positive_window,
            "golden_lo": args.golden_lo,
            "golden_hi": args.golden_hi,
            "golden_horizon": args.golden_horizon,
            "k_fixed": args.k_fixed,
            "k_range": list(args.k_range),
            "silhouette": float(fit_res.silhouette),
        },
        cluster_sizes=fit_res.cluster_sizes,
        dist_thresholds=fit_res.dist_thresholds,
    )

    if cluster_artifacts:
        trainers = {cid: a.trainer for cid, a in cluster_artifacts.items() if a.trainer is not None}
        metrics_per_cluster = {cid: a.metrics for cid, a in cluster_artifacts.items()}
        importance_per_cluster = {cid: a.feature_importance for cid, a in cluster_artifacts.items()
                                   if a.feature_importance is not None}
        usability = {cid: a.usable for cid, a in cluster_artifacts.items()}
        arch.save_per_cluster(
            trainers=trainers,
            metrics_per_cluster=metrics_per_cluster,
            feature_importance_per_cluster=importance_per_cluster,
            usability_flags=usability,
        )

    if single_artifact and single_artifact.trainer:
        arch.save_single_with_cf(
            trainer=single_artifact.trainer,
            metrics=single_artifact.metrics,
            feature_importance=single_artifact.feature_importance,
            categorical_feature=["cluster_id"],
        )

    if args.save_training_data:
        arch.save_training_data(
            samples_with_cluster=samples,
            features=features,
        )

    # 可视化
    logger.info("Step 11: 出可视化图")
    viz_dir = arch.run_dir / "visualization"
    plot_prototypes(fit_res.medoid_sequences, viz_dir)
    plot_cluster_year_distribution(samples, viz_dir)
    if cluster_artifacts:
        plot_feature_importance_per_cluster(
            {cid: a.feature_importance for cid, a in cluster_artifacts.items()
             if a.feature_importance is not None},
            viz_dir,
        )

    # 汇总 training_summary.json
    summary = {
        "window_id": window_id,
        "best_k": fit_res.best_k,
        "silhouette": float(fit_res.silhouette),
        "cluster_sizes": {int(k): int(v) for k, v in fit_res.cluster_sizes.items()},
        "per_cluster_summary": [a.to_summary_row() for a in cluster_artifacts.values()] if cluster_artifacts else [],
        "single_with_cf_summary": single_artifact.to_summary() if single_artifact else None,
    }
    arch.save_training_summary(summary)
    arch.update_latest()

    logger.info(f"=== bundle 写入: {arch.run_dir} ===")
    return arch.run_dir


# ============================================================
def main():
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    daily = load_daily(args)

    # dry-run
    logger.info("=== Dry-run 数据诊断 ===")
    dry = DataDryRun(
        lo_hi_bins=[(args.golden_lo, args.golden_hi)],
        target_bin=(args.golden_lo, args.golden_hi),
        horizon=args.golden_horizon,
        min_events_required=args.min_golden_events,
        detector_max_gap=args.max_gap,
    )
    rep = dry.summarize(daily)
    print(rep.pretty_print())
    if not rep.ok and not args.skip_dryrun_abort:
        logger.error("Dry-run 不达标, 退出 (用 --skip-dryrun-abort 强行继续)")
        sys.exit(1)

    # walk-forward or single
    if args.walk_forward_months > 0:
        logger.info("=== Walk-Forward 模式 ===")
        sch = WalkForwardScheduler(
            start=args.start, end=args.end,
            step_months=args.walk_forward_months,
            min_train_months=args.min_train_months,
        )
        windows = sch.generate_windows()
        if not windows:
            logger.error("无可用 walk-forward 窗口")
            sys.exit(1)
        schedule_rows = []
        for w in windows:
            bundle_path = train_one_window(
                daily, args,
                train_end=w.train_end,
                valid_end=(w.train_end + pd.Timedelta(days=15)),   # valid 取 train_end 之后 15 天
                test_end=w.test_end,
                out_root=out_root,
                window_id=w.window_id,
            )
            schedule_rows.append({**w.to_dict(), "bundle_dir": str(bundle_path)})
        with open(out_root / "schedule.json", "w", encoding="utf-8") as f:
            json.dump(schedule_rows, f, ensure_ascii=False, indent=2)
        logger.info(f"walk-forward schedule 写入: {out_root / 'schedule.json'}")
    else:
        if args.train_end is None or args.test_end is None:
            logger.error("单次模式需要 --train-end / --test-end")
            sys.exit(1)
        train_end = pd.Timestamp(args.train_end)
        valid_end = pd.Timestamp(args.valid_end) if args.valid_end else train_end + pd.Timedelta(days=15)
        test_end = pd.Timestamp(args.test_end)
        train_one_window(
            daily, args,
            train_end=train_end, valid_end=valid_end, test_end=test_end,
            out_root=out_root,
        )


if __name__ == "__main__":
    main()
