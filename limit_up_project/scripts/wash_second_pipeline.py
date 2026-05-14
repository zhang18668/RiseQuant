r"""新策略 pipeline: 首板 → 震荡洗盘 → 第二涨停 (前几天潜伏买入)

业务定义:
- 首板: 当日涨停, 前 3 个交易日内无涨停 (严格)
- 第二涨停: 首板后 [3, 20] 交易日内出现的下一次涨停, 中间不能有其他涨停,
  且第二涨停 close > 首板 close (创新高)
- 训练目标: 在震荡区间的每一天, 给"未来 K 天 (默认 3) 内会出第二涨停"打分
- 止损: 跌破首板 open 即出 (动态止损)

用法:
    set PYTHONPATH=%CD%
    python scripts\wash_second_pipeline.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import setup_logger, get_logger
from src.utils.validator import LookAheadValidator
from src.utils.run_archive import RunArchive
from src.data.tdx_loader import TDXDataLoader
from src.event.wash_second_detector import WashSecondDetector
from src.event.wash_sample_builder import WashSampleBuilder
from src.feature.wash_features import WashFeatures
from src.feature.market_features import MarketFeatures, market_proxy_from_daily
from src.dataset.sector_split import SectorStockSplitter
from src.model.model_trainer import LimitUpModelTrainer
from src.model.model_evaluator import ModelEvaluator
from src.backtest.backtester import Backtester

setup_logger(log_level="INFO")
logger = get_logger(__name__)


def detect_events_and_samples(daily: pd.DataFrame, cfg: dict):
    logger.info("Step 1: 检测 first->wash->second 事件")
    det = WashSecondDetector(
        threshold=cfg.get("limit_up_threshold", 9.9),
        cooldown_days=cfg.get("cooldown_days", 3),
        min_gap=cfg.get("min_gap", 3),
        max_gap=cfg.get("max_gap", 20),
        exclude_st=cfg.get("exclude_st", True),
    )
    events = det.detect(daily)
    logger.info(f"  捕获事件三元组: {len(events)} 条")
    if events.empty:
        return events, pd.DataFrame()

    logger.info("Step 2: 构造滚动潜伏样本 (正负样本)")
    builder = WashSampleBuilder(
        positive_window=cfg.get("positive_window", 3),
        min_gap_for_neg=cfg.get("min_gap", 3),
        max_gap_for_neg=cfg.get("max_gap", 20),
        threshold=cfg.get("limit_up_threshold", 9.9),
        cooldown_days=cfg.get("cooldown_days", 3),
    )
    samples = builder.build(daily, events=events, include_negatives=True)
    n_pos = int(samples["label_pre"].sum())
    n_neg = int((samples["label_pre"] == 0).sum())
    logger.info(f"  样本: {len(samples)} 条 (positive={n_pos}, negative={n_neg})")
    return events, samples


def compute_features(samples: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 3: 计算震荡洗盘特征 (WashFeatures) + 大盘特征 (MarketFeatures)")
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
        feats["days_to_second"] = int(r["days_to_second"]) if r["days_to_second"] != -1 else -1
        rows.append(feats)
        if (i + 1) % 2000 == 0:
            logger.info(f"  wash feature progress: {i+1}/{len(samples)}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # B2: 大盘特征 — 用全市场代理 (close=median, volume=sum), 按 potential_date 横切
    logger.info("  合成大盘代理 + 计算大盘特征 (按日期去重批量算)")
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
    logger.info(f"  特征表: {len(df)} 行 x {n_wash} wash + {n_mkt} market = {n_wash+n_mkt} 维")
    return df


def train(features_df: pd.DataFrame, full_cfg: dict, arch: RunArchive):
    logger.info("Step 4: 切分 + 训练")
    # B1+B2: wash 特征 + 大盘特征 都纳入
    feature_cols = [c for c in features_df.columns
                    if c.startswith("f_w_") or c.startswith("f_mkt_")]
    LookAheadValidator.scan_feature_names(feature_cols, raise_error=True)

    sp_method = full_cfg.get("split_method", "date")
    splitter = SectorStockSplitter(
        test_ratio=full_cfg.get("test_ratio", 0.2),
        valid_ratio=full_cfg.get("valid_ratio", 0.1),
    )
    if sp_method == "date":
        splits = splitter.split_by_date_ranges(
            features_df, date_col="potential_date",
            train_end=full_cfg.get("train_end"),
            valid_end=full_cfg.get("valid_end"),
            test_end=full_cfg.get("test_end"),
        )
    else:
        splits = splitter.split_by_time(features_df, date_col="potential_date")
    LookAheadValidator.assert_no_time_leakage(splits, date_col="potential_date")

    logger.info(f"  train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

    X_train = splits["train"][feature_cols].fillna(0)
    y_train = splits["train"]["label_pre"]
    X_valid = splits["valid"][feature_cols].fillna(0)
    y_valid = splits["valid"]["label_pre"]
    X_test  = splits["test"][feature_cols].fillna(0)
    y_test  = splits["test"]["label_pre"]
    # B3: 样本权重
    w_train = splits["train"]["sample_weight"] if "sample_weight" in splits["train"].columns else None
    w_valid = splits["valid"]["sample_weight"] if "sample_weight" in splits["valid"].columns else None
    if w_train is not None:
        logger.info(f"  样本权重: train mean={w_train.mean():.3f}, std={w_train.std():.3f}")

    trainer = LimitUpModelTrainer({"params": full_cfg.get("model_params", {})})
    trainer.train(
        X_train, y_train, X_valid, y_valid,
        feature_names=feature_cols,
        sample_weight=w_train,
        eval_sample_weight=w_valid,
    )

    # 评估
    ev = ModelEvaluator()
    y_pred = trainer.predict(X_test)
    y_proba = trainer.predict_proba(X_test)
    metrics = ev.evaluate(y_test.values, y_pred, y_proba)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")

    # ---- 写入档案 ----
    split_sizes = {
        "train": int(len(X_train)), "valid": int(len(X_valid)), "test": int(len(X_test)),
    }
    def _rng(name):
        seg = splits.get(name)
        if seg is None or seg.empty or "potential_date" not in seg.columns:
            return None
        d = pd.to_datetime(seg["potential_date"])
        return [str(d.min()), str(d.max())]
    split_ranges = {
        "train": _rng("train"), "valid": _rng("valid"), "test": _rng("test"),
    }
    arch.save_config({
        **full_cfg,
        "feature_cols": feature_cols,
    })
    arch.save_model(trainer)
    arch.save_train_metrics(metrics, split_sizes=split_sizes, split_ranges=split_ranges)
    arch.save_feature_importance(trainer)
    logger.info(f"  训练产物已写入: {arch.run_dir}")

    # 特征重要性 top 20 也打到日志
    imp = trainer.get_feature_importance(top_n=20)
    logger.info("Top 20 特征:")
    for _, r in imp.iterrows():
        logger.info(f"  {r['feature']}: {r['importance']:.0f}")

    return trainer, splits, feature_cols


def backtest(features_df: pd.DataFrame, trainer, daily: pd.DataFrame, full_cfg: dict, arch: RunArchive):
    logger.info("Step 5: 回测 (动态止损 = 首板 open)")
    feature_cols = trainer.feature_names_
    X = features_df.reindex(columns=feature_cols, fill_value=0).fillna(0)
    proba = trainer.predict_proba(X)
    # 二分类: 取 P(label=1) (即"未来 K 日内出第二涨停"的概率)
    p_pos = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]

    signals = features_df[["potential_date", "stop_loss_price"]].copy()
    signals["code"] = features_df["sample_id"].str.split("_").str[0]
    signals = signals.rename(columns={"potential_date": "date"})
    signals["score"] = p_pos

    # 回测窗口
    bt_start = full_cfg.get("bt_start")
    bt_end   = full_cfg.get("bt_end")
    if bt_start:
        signals = signals[signals["date"] >= pd.Timestamp(bt_start)]
        daily   = daily[daily["date"]   >= pd.Timestamp(bt_start)]
    if bt_end:
        signals = signals[signals["date"] <= pd.Timestamp(bt_end)]
        daily   = daily[daily["date"]   <= pd.Timestamp(bt_end)]
    logger.info(f"  回测 {bt_start or 'begin'} ~ {bt_end or 'end'}, 信号 {len(signals)} 条")

    bt = Backtester(
        initial_cash=full_cfg.get("initial_cash", 10_000_000),
        topk=full_cfg.get("topk", 5),
        sell_n=full_cfg.get("sell_n", 8),
        slippage=full_cfg.get("slippage", 0.003),
        entry_delay=1,
        min_score=full_cfg.get("min_score", 0.6),
        stop_loss=0.0,            # 关闭百分比硬止损, 用动态止损
        take_profit=full_cfg.get("take_profit", 0.0),
        trailing_stop=full_cfg.get("trailing_stop", 0.0),
        skip_zhangting_open=True,
        cooldown_after_stop=full_cfg.get("cooldown_after_stop", 5),
    )
    bt.run(signals, daily)
    m = bt.get_metrics()
    logger.info("=" * 50)
    logger.info(f"  总收益率   : {m['total_return']:.2%}")
    logger.info(f"  年化收益率 : {m['annual_return']:.2%}")
    logger.info(f"  夏普比率   : {m['sharpe_ratio']:.2f}")
    logger.info(f"  最大回撤   : {m['max_drawdown']:.2%}")
    logger.info(f"  交易次数   : {m['num_trades']}")
    logger.info(f"  胜率       : {m['win_rate']:.2%}" if m['win_rate'] == m['win_rate'] else "胜率: nan")
    trades = bt.get_trades()
    if len(trades):
        reasons = trades[trades["action"].str.startswith("SELL")]["action"].value_counts()
        logger.info(f"  卖出原因分布:\n{reasons.to_string()}")

    # ---- 写入档案 (含 trades.csv / equity_curve.csv / backtest_metrics.json) ----
    bt_metrics = arch.save_backtest(bt, signals_df=signals)
    logger.info(f"  回测产物已写入: {arch.run_dir}")
    return bt


def main():
    cfg = get_config()
    event_cfg = cfg.get_section("event")
    bt_cfg = cfg.get_section("backtest")

    full_cfg = {
        "limit_up_threshold": event_cfg.get("limit_up_threshold", 9.9),
        "cooldown_days":      3,
        "min_gap":            3,
        "max_gap":            30,
        "positive_window":    5,
        "exclude_st":         event_cfg.get("exclude_st", True),
        "split_method":       cfg.get("dataset.split_method", "date"),
        "test_ratio":         cfg.get("dataset.test_ratio", 0.2),
        "valid_ratio":        cfg.get("dataset.valid_ratio", 0.1),
        "train_end":          cfg.get("dataset.train_end"),
        "valid_end":          cfg.get("dataset.valid_end"),
        "test_end":           cfg.get("dataset.test_end"),
        "model_dir":          cfg.get("dataset.model_dir", "./models"),
        "model_params":       cfg.get_section("model").get("params", {}),
        "initial_cash":       bt_cfg.get("initial_cash", 10_000_000),
        "topk":               5,
        "sell_n":             bt_cfg.get("sell_n", 8),
        "slippage":           bt_cfg.get("slippage", 0.003),
        "min_score":          bt_cfg.get("min_score", 0.6),
        "trailing_stop":      bt_cfg.get("trailing_stop", 0.0),
        "cooldown_after_stop": bt_cfg.get("cooldown_after_stop", 5),
        "bt_start":           bt_cfg.get("start_date"),
        "bt_end":             bt_cfg.get("end_date"),
    }

    # 加载数据
    tdx = TDXDataLoader(r"C:\new_tdx\vipdoc")
    if tdx.data_path is None:
        logger.error("找不到通达信目录")
        return
    sh = tdx.load_stock_list("sh")
    sz = tdx.load_stock_list("sz")
    codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    logger.info(f"加载 {len(codes)} 只主板股票")
    daily = tdx.load_batch(codes=codes, start_date="2022-01-01", end_date="2024-12-31")
    if daily.empty:
        logger.error("数据为空")
        return
    daily["date"] = pd.to_datetime(daily["date"])

    # 运行档案: 每次训练独立目录
    arch = RunArchive(
        base_dir=full_cfg.get("model_dir", "./models"),
        strategy="wash_second",
    )
    logger.info(f"运行档案: {arch.run_dir}")

    # Pipeline
    events, samples = detect_events_and_samples(daily, full_cfg)
    if samples.empty:
        logger.error("无样本, 退出")
        return
    feats = compute_features(samples, daily)
    if feats.empty:
        logger.error("特征为空, 退出")
        return
    trainer, splits, fcols = train(feats, full_cfg, arch)
    backtest(feats, trainer, daily, full_cfg, arch)
    summary_path = arch.write_summary()
    logger.info(f"summary.json -> {summary_path}")
    logger.info(f"latest -> {arch.latest_dir}")


if __name__ == "__main__":
    main()
