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
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    p.add_argument("--model-n-jobs", type=int, default=-1,
                   help="LightGBM 单模型线程数; -1 使用全部核心, 1 适合配合 --cluster-train-jobs")
    p.add_argument("--cluster-train-jobs", type=int, default=1,
                   help="per_cluster 方案同时训练多少个 cluster 模型")
    p.add_argument("--feature-jobs", type=int, default=1,
                   help="Step 8 feature engineering worker process count")
    p.add_argument("--feature-cache", action="store_true",
                   help="cache Step 8 features and reuse when inputs/params/code match")
    p.add_argument("--feature-cache-dir", default=None,
                   help="feature cache directory, default: <out>/feature_cache")
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
def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_frame(df: pd.DataFrame, cols: List[str]) -> str:
    present = [c for c in cols if c in df.columns]
    if not present:
        return "empty"
    part = df[present].copy()
    for c in present:
        if pd.api.types.is_datetime64_any_dtype(part[c]):
            part[c] = part[c].astype("datetime64[ns]").astype("int64")
    values = pd.util.hash_pandas_object(part, index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def _feature_cache_key(samples: pd.DataFrame, daily: pd.DataFrame, args) -> str:
    payload = {
        "version": 1,
        "args": {
            "start": args.start,
            "end": args.end,
            "tdx_path": str(args.tdx_path),
            "golden_lo": args.golden_lo,
            "golden_hi": args.golden_hi,
            "golden_horizon": args.golden_horizon,
            "max_gap": args.max_gap,
            "positive_window": args.positive_window,
        },
        "code_hashes": {
            "wash_features.py": _hash_file(ROOT / "src" / "feature" / "wash_features.py"),
            "market_features.py": _hash_file(ROOT / "src" / "feature" / "market_features.py"),
            "technical_indicators.py": _hash_file(ROOT / "src" / "feature" / "technical_indicators.py"),
            "wash_sample_builder.py": _hash_file(ROOT / "src" / "event" / "wash_sample_builder.py"),
            "wash_second_detector.py": _hash_file(ROOT / "src" / "event" / "wash_second_detector.py"),
            "golden_label_filter.py": _hash_file(ROOT / "src" / "event" / "golden_label_filter.py"),
        },
        "samples": {
            "rows": int(len(samples)),
            "hash": _hash_frame(
                samples,
                [
                    "sample_id", "code", "first_date", "potential_date",
                    "label_pre", "sample_weight", "stop_loss_price",
                    "first_open", "is_golden_event",
                ],
            ),
        },
        "daily": {
            "rows": int(len(daily)),
            "hash": _hash_frame(
                daily,
                [
                    "date", "code", "open", "high", "low", "close",
                    "volume", "turnover", "change_pct",
                ],
            ),
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _feature_cache_paths(args, out_root: Path, key: str) -> Tuple[Path, Path]:
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else out_root / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"features_{key}.pkl.gz", cache_dir / f"features_{key}.json"


def _compute_feature_rows(samples_chunk: pd.DataFrame, daily_by_code: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    wf = WashFeatures()
    rows = []
    for r in samples_chunk.itertuples(index=False):
        code = str(getattr(r, "code")).zfill(6)
        sub = daily_by_code.get(code)
        if sub is None:
            continue
        feats = wf.calculate(sub, getattr(r, "first_date"), getattr(r, "potential_date")).to_dict()
        feats["sample_id"] = getattr(r, "sample_id")
        feats["label_pre"] = int(getattr(r, "label_pre"))
        feats["sample_weight"] = float(getattr(r, "sample_weight"))
        feats["stop_loss_price"] = float(getattr(r, "stop_loss_price"))
        feats["first_open"] = float(getattr(r, "first_open"))
        rows.append(feats)
    return pd.DataFrame(rows)


def _compute_feature_partition(payload: Tuple[int, pd.DataFrame, Dict[str, pd.DataFrame]]) -> Tuple[int, int, pd.DataFrame]:
    worker_id, samples_chunk, daily_chunk = payload
    return worker_id, len(samples_chunk), _compute_feature_rows(samples_chunk, daily_chunk)


def _feature_partitions(
    samples: pd.DataFrame,
    daily_by_code: Dict[str, pd.DataFrame],
    n_jobs: int,
) -> List[Tuple[int, pd.DataFrame, Dict[str, pd.DataFrame]]]:
    normalized_codes = samples["code"].astype(str).str.zfill(6)
    code_counts = normalized_codes.groupby(normalized_codes).size().sort_values(ascending=False)
    buckets: List[List[str]] = [[] for _ in range(n_jobs)]
    bucket_sizes = [0 for _ in range(n_jobs)]
    for code, count in code_counts.items():
        idx = int(np.argmin(bucket_sizes))
        buckets[idx].append(str(code))
        bucket_sizes[idx] += int(count)

    parts = []
    for i, codes in enumerate(buckets):
        if not codes:
            continue
        code_set = set(codes)
        chunk = samples[normalized_codes.isin(code_set)].copy()
        daily_chunk = {c: daily_by_code[c] for c in codes if c in daily_by_code}
        parts.append((i, chunk, daily_chunk))
    return parts


def _append_market_features(features: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    market_data = market_proxy_from_daily(daily)
    mf = MarketFeatures()
    features = features.copy()
    features["potential_date"] = pd.to_datetime(features["potential_date"])
    unique_dates = features["potential_date"].drop_duplicates().sort_values()
    mkt_rows = [mf.calculate_at_event(d, market_data) for d in unique_dates]
    mkt_df = pd.DataFrame(mkt_rows)
    if not mkt_df.empty:
        mkt_df["event_date"] = pd.to_datetime(mkt_df["event_date"])
        mkt_df = mkt_df.rename(columns={"event_date": "potential_date"})
        features = features.merge(mkt_df, on="potential_date", how="left")
    return features


def compute_features(
    samples: pd.DataFrame,
    daily: pd.DataFrame,
    args=None,
    out_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Step 8: 算 wash + market features (复用现有模块)."""
    logger.info("Step 8: 特征工程 (WashFeatures + MarketFeatures)")

    cache_file = None
    cache_meta = None
    if args is not None and out_root is not None and getattr(args, "feature_cache", False):
        logger.info("  feature cache: checking inputs/params/code fingerprint...")
        key = _feature_cache_key(samples, daily, args)
        cache_file, cache_meta = _feature_cache_paths(args, out_root, key)
        if cache_file.exists():
            logger.info(f"  feature cache HIT: {cache_file}")
            return pd.read_pickle(cache_file, compression="gzip")
        logger.info(f"  feature cache MISS: {cache_file}")

    jobs = max(1, int(getattr(args, "feature_jobs", 1) if args is not None else 1))
    daily_by_code = {str(c).zfill(6): g.sort_values("date").reset_index(drop=True)
                     for c, g in daily.groupby("code")}

    if jobs == 1:
        rows = []
        wf = WashFeatures()
        for i, r in samples.iterrows():
            sub = daily_by_code.get(str(r["code"]).zfill(6))
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
    else:
        parts = _feature_partitions(samples, daily_by_code, jobs)
        logger.info(f"  parallel wash features: {len(parts)} partitions, {jobs} workers")
        done = 0
        frames = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_compute_feature_partition, part) for part in parts]
            for future in as_completed(futures):
                worker_id, count, frame = future.result()
                done += count
                frames.append(frame)
                logger.info(f"  wash feature chunk {worker_id} done: {done}/{len(samples)}")
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        return df

    df = _append_market_features(df, daily)

    n_wash = sum(1 for c in df.columns if c.startswith("f_w_"))
    n_mkt  = sum(1 for c in df.columns if c.startswith("f_mkt_"))
    logger.info(f"  特征表 {len(df)} 行 × ({n_wash} wash + {n_mkt} market) = {n_wash + n_mkt} 维")
    if cache_file is not None and cache_meta is not None:
        df.to_pickle(cache_file, compression="gzip")
        with cache_meta.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "cache_file": str(cache_file),
                    "rows": int(len(df)),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "feature_jobs": jobs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"  feature cache saved: {cache_file}")
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
    features = compute_features(samples, daily, args=args, out_root=out_root)
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
    model_params = {
        "n_estimators": args.n_estimators,
        "n_jobs": args.model_n_jobs,
        "verbose": -1,
    }
    cluster_artifacts: Dict[int, Any] = {}
    single_artifact = None

    if args.bundle_type in ("per_cluster", "both"):
        logger.info("Step 9-A: ClusterModelTrainer (方案 A)")
        cmt = ClusterModelTrainer(
            min_train_samples=args.min_train_samples_per_cluster,
            min_auc_test=args.min_auc_test,
            model_params=model_params,
            feature_cols=feature_cols,
            n_jobs=args.cluster_train_jobs,
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
