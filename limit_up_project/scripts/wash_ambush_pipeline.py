"""Train and backtest the wash_ambush strategy.

Goal:
- Buy during the pullback after the first limit-up.
- Prefer entries 2-6 trading days before the second limit-up.
- Avoid buying after the move has already expanded too much.
- Use staged entries through AmbushBacktester.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest.ambush_backtester import AmbushBacktester
from src.data.tdx_loader import TDXDataLoader
from src.dataset.sector_split import SectorStockSplitter
from src.event.wash_ambush_sample_builder import WashAmbushSampleBuilder
from src.event.wash_second_detector import WashSecondDetector
from src.feature.market_features import MarketFeatures, market_proxy_from_daily
from src.feature.wash_features import WashFeatures
from src.model.model_evaluator import ModelEvaluator
from src.model.model_trainer import LimitUpModelTrainer
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger
from src.utils.run_archive import RunArchive
from src.utils.validator import LookAheadValidator

setup_logger(log_level="INFO")
logger = get_logger(__name__)


def detect_events_and_samples(daily: pd.DataFrame, cfg: dict):
    logger.info("Step 1: detect first->pullback->second events")
    det = WashSecondDetector(
        threshold=cfg.get("limit_up_threshold", 9.9),
        cooldown_days=cfg.get("cooldown_days", 3),
        min_gap=cfg.get("min_gap", 3),
        max_gap=cfg.get("max_gap", 30),
        exclude_st=cfg.get("exclude_st", True),
    )
    events = det.detect(daily)
    logger.info(f"  events: {len(events)}")
    if events.empty:
        return events, pd.DataFrame()

    logger.info("Step 2: build ambush samples")
    builder = WashAmbushSampleBuilder(
        positive_min_lead=cfg.get("positive_min_lead", 2),
        positive_max_lead=cfg.get("positive_max_lead", 6),
        min_gap_for_neg=cfg.get("min_gap", 3),
        max_gap_for_neg=cfg.get("max_gap", 30),
        threshold=cfg.get("limit_up_threshold", 9.9),
        cooldown_days=cfg.get("cooldown_days", 3),
    )
    samples = builder.build(daily, events=events, include_negatives=True)
    n_pos = int(samples["label_pre"].sum()) if len(samples) else 0
    n_neg = int((samples["label_pre"] == 0).sum()) if len(samples) else 0
    logger.info(f"  samples: {len(samples)} positive={n_pos} negative={n_neg}")
    return events, samples


def balance_samples(samples: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if samples.empty or "label_pre" not in samples.columns:
        return samples
    neg_ratio = float(cfg.get("negative_sample_ratio", 3.0))
    max_samples = int(cfg.get("max_feature_samples", 90000))
    positives = samples[samples["label_pre"] == 1]
    negatives = samples[samples["label_pre"] == 0]
    if positives.empty or negatives.empty:
        return samples

    n_neg = min(len(negatives), int(len(positives) * neg_ratio))
    sampled_neg = negatives.sample(n=n_neg, random_state=42) if n_neg < len(negatives) else negatives
    balanced = pd.concat([positives, sampled_neg], ignore_index=True)
    if len(balanced) > max_samples:
        keep_pos = positives
        n_neg = max(0, max_samples - len(keep_pos))
        sampled_neg = sampled_neg.sample(n=min(n_neg, len(sampled_neg)), random_state=43)
        balanced = pd.concat([keep_pos, sampled_neg], ignore_index=True)
    balanced = balanced.sort_values(["code", "first_date", "potential_date"]).reset_index(drop=True)
    logger.info(
        "  balanced samples for features: %d positive=%d negative=%d"
        % (len(balanced), int((balanced["label_pre"] == 1).sum()), int((balanced["label_pre"] == 0).sum()))
    )
    return balanced


def compute_features(samples: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 3: compute wash + market features")
    wf = WashFeatures()
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in daily.groupby("code")}
    rows = []
    for i, r in samples.iterrows():
        sub = daily_by_code.get(r["code"])
        if sub is None:
            continue
        feats = wf.calculate(sub, r["first_date"], r["potential_date"]).to_dict()
        feats["sample_id"] = r["sample_id"]
        feats["label_pre"] = int(r["label_pre"])
        feats["sample_weight"] = float(r["sample_weight"])
        feats["stop_loss_price"] = float(r["stop_loss_price"])
        feats["first_open"] = float(r["first_open"])
        feats["first_close"] = float(r["first_close"]) if pd.notna(r["first_close"]) else float("nan")
        feats["first_date"] = pd.Timestamp(r["first_date"])
        feats["second_date"] = pd.Timestamp(r["second_date"]) if pd.notna(r["second_date"]) else pd.NaT
        feats["days_to_second"] = int(r["days_to_second"]) if r["days_to_second"] != -1 else -1
        feats["gap_so_far"] = int(r["gap_so_far"])
        rows.append(feats)
        if (i + 1) % 5000 == 0:
            logger.info(f"  feature progress: {i + 1}/{len(samples)}")

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

    n_wash = sum(1 for c in df.columns if c.startswith("f_w_"))
    n_mkt = sum(1 for c in df.columns if c.startswith("f_mkt_"))
    logger.info(f"  features: {len(df)} rows x {n_wash + n_mkt} dims")
    return df


def train(features_df: pd.DataFrame, full_cfg: dict, arch: RunArchive):
    logger.info("Step 4: split + train")
    feature_cols = [c for c in features_df.columns if c.startswith("f_w_") or c.startswith("f_mkt_")]
    LookAheadValidator.scan_feature_names(feature_cols, raise_error=True)

    splitter = SectorStockSplitter(
        test_ratio=full_cfg.get("test_ratio", 0.2),
        valid_ratio=full_cfg.get("valid_ratio", 0.1),
    )
    if full_cfg.get("split_method", "date") == "date":
        splits = splitter.split_by_date_ranges(
            features_df,
            date_col="potential_date",
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
    X_test = splits["test"][feature_cols].fillna(0)
    y_test = splits["test"]["label_pre"]
    w_train = splits["train"]["sample_weight"] if "sample_weight" in splits["train"].columns else None
    w_valid = splits["valid"]["sample_weight"] if "sample_weight" in splits["valid"].columns else None

    trainer = LimitUpModelTrainer({"params": full_cfg.get("model_params", {})})
    trainer.train(
        X_train,
        y_train,
        X_valid,
        y_valid,
        feature_names=feature_cols,
        sample_weight=w_train,
        eval_sample_weight=w_valid,
    )

    ev = ModelEvaluator()
    metrics = ev.evaluate(y_test.values, trainer.predict(X_test), trainer.predict_proba(X_test))
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")

    def _rng(name):
        seg = splits.get(name)
        if seg is None or seg.empty or "potential_date" not in seg.columns:
            return None
        d = pd.to_datetime(seg["potential_date"])
        return [str(d.min()), str(d.max())]

    arch.save_config({**full_cfg, "feature_cols": feature_cols})
    arch.save_model(trainer)
    arch.save_train_metrics(
        metrics,
        split_sizes={k: int(len(v)) for k, v in splits.items()},
        split_ranges={k: _rng(k) for k in splits},
    )
    arch.save_feature_importance(trainer)
    logger.info(f"  train artifacts: {arch.run_dir}")
    return trainer, splits, feature_cols


def make_signals(features_df: pd.DataFrame, trainer) -> pd.DataFrame:
    feature_cols = trainer.feature_names_
    X = features_df.reindex(columns=feature_cols, fill_value=0).fillna(0)
    proba = trainer.predict_proba(X)
    p_pos = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]

    keep = [
        "potential_date",
        "stop_loss_price",
        "first_open",
        "first_close",
        "first_date",
        "second_date",
        "days_to_second",
        "gap_so_far",
        "f_w_broke_stop",
        "f_w_close_to_fc",
        "f_w_close_to_fh",
        "f_w_gap_so_far",
    ]
    cols = [c for c in keep if c in features_df.columns]
    signals = features_df[cols].copy()
    signals["code"] = features_df["sample_id"].str.split("_").str[0]
    signals = signals.rename(columns={"potential_date": "date"})
    signals["score"] = p_pos
    return signals


def _as_list(v, default):
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def run_bt(signals: pd.DataFrame, daily: pd.DataFrame, full_cfg: dict, params: dict | None = None):
    params = params or {}
    bt = AmbushBacktester(
        initial_cash=full_cfg.get("initial_cash", 10_000_000),
        topk=params.get("topk", full_cfg.get("topk", 5)),
        sell_n=params.get("sell_n", full_cfg.get("sell_n", 8)),
        slippage=full_cfg.get("slippage", 0.003),
        entry_delay=1,
        min_score=params.get("min_score", full_cfg.get("min_score", 0.6)),
        stop_loss=0.0,
        take_profit=params.get("take_profit", full_cfg.get("take_profit", 0.0)),
        trailing_stop=params.get("trailing_stop", full_cfg.get("trailing_stop", 0.0)),
        skip_zhangting_open=True,
        cooldown_after_stop=params.get("cooldown_after_stop", full_cfg.get("cooldown_after_stop", 5)),
        scale_in_steps=params.get("scale_in_steps", full_cfg.get("scale_in_steps", 2)),
        max_code_weight=params.get("max_code_weight", full_cfg.get("max_code_weight", 0.12)),
        max_entry_over_first_close=params.get(
            "max_entry_over_first_close",
            full_cfg.get("max_entry_over_first_close", 0.08),
        ),
        max_entry_gap=params.get("max_entry_gap", full_cfg.get("max_entry_gap", 0.06)),
        max_entry_day_change_pct=params.get(
            "max_entry_day_change_pct",
            full_cfg.get("max_entry_day_change_pct", 7.0),
        ),
    )
    bt.run(signals, daily)
    return bt


def optimize_backtest(signals: pd.DataFrame, daily: pd.DataFrame, full_cfg: dict, arch: RunArchive):
    cfg = full_cfg.get("backtest_optimize", {}) or {}
    grids = {
        "min_score": _as_list(cfg.get("min_score_grid"), [0.58, 0.64]),
        "topk": _as_list(cfg.get("topk_grid"), [2, 3]),
        "sell_n": _as_list(cfg.get("sell_n_grid"), [4, 6]),
        "trailing_stop": _as_list(cfg.get("trailing_stop_grid"), [0.05]),
        "take_profit": _as_list(cfg.get("take_profit_grid"), [0.0, 0.12]),
        "max_entry_over_first_close": _as_list(cfg.get("max_entry_over_first_close_grid"), [0.04, 0.08]),
        "scale_in_steps": _as_list(cfg.get("scale_in_steps_grid"), [2]),
    }
    min_sells = int(cfg.get("min_sells", 20))
    rows = []
    total = int(np.prod([len(v) for v in grids.values()]))
    logger.info(f"  optimize ambush grid: {total} combos")

    for min_score in grids["min_score"]:
        for topk in grids["topk"]:
            for sell_n in grids["sell_n"]:
                for trailing_stop in grids["trailing_stop"]:
                    for take_profit in grids["take_profit"]:
                        for max_entry_over_first_close in grids["max_entry_over_first_close"]:
                            for scale_in_steps in grids["scale_in_steps"]:
                                params = {
                                    "min_score": float(min_score),
                                    "topk": int(topk),
                                    "sell_n": int(sell_n),
                                    "trailing_stop": float(trailing_stop),
                                    "take_profit": float(take_profit),
                                    "max_entry_over_first_close": float(max_entry_over_first_close),
                                    "scale_in_steps": int(scale_in_steps),
                                }
                                bt = run_bt(signals, daily, full_cfg, params)
                                metrics = dict(bt.get_metrics())
                                trades = bt.get_trades()
                                sells = trades[trades["action"].astype(str).str.startswith("SELL")] if len(trades) else pd.DataFrame()
                                row = {**params, **metrics, "num_sells": int(len(sells))}
                                if len(sells):
                                    row["sell_avg_return"] = float(sells["return_pct"].mean())
                                    row["sell_median_return"] = float(sells["return_pct"].median())
                                rows.append(row)

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return {}, grid_df
    grid_df["sharpe_rank_value"] = grid_df["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    eligible = grid_df[grid_df["num_sells"] >= min_sells].copy()
    if eligible.empty:
        eligible = grid_df.copy()
        logger.warning(f"  no combo reached min_sells={min_sells}; selecting from all combos")
    eligible = eligible.sort_values(
        ["sharpe_rank_value", "total_return", "max_drawdown", "num_sells"],
        ascending=[False, False, False, False],
    )
    best = eligible.iloc[0].to_dict()
    best_params = {
        "min_score": float(best["min_score"]),
        "topk": int(best["topk"]),
        "sell_n": int(best["sell_n"]),
        "trailing_stop": float(best["trailing_stop"]),
        "take_profit": float(best["take_profit"]),
        "max_entry_over_first_close": float(best["max_entry_over_first_close"]),
        "scale_in_steps": int(best["scale_in_steps"]),
    }
    grid_df = grid_df.sort_values(
        ["sharpe_rank_value", "total_return", "max_drawdown"],
        ascending=[False, False, False],
    ).drop(columns=["sharpe_rank_value"])
    grid_df.to_csv(arch.run_dir / "backtest_grid.csv", index=False, encoding="utf-8-sig")
    with open(arch.run_dir / "best_backtest_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    arch._summary["artifacts"].extend(["backtest_grid.csv", "best_backtest_params.json"])
    logger.info(
        "  best grid sharpe=%.2f return=%.2f%% drawdown=%.2f%% sells=%d"
        % (
            float(best.get("sharpe_ratio", np.nan)),
            float(best.get("total_return", 0.0)) * 100,
            float(best.get("max_drawdown", 0.0)) * 100,
            int(best.get("num_sells", 0)),
        )
    )
    return best_params, grid_df


def backtest(features_df: pd.DataFrame, trainer, daily: pd.DataFrame, full_cfg: dict, arch: RunArchive):
    logger.info("Step 5: ambush backtest")
    signals = make_signals(features_df, trainer)
    bt_start = full_cfg.get("bt_start")
    bt_end = full_cfg.get("bt_end")
    if bt_start:
        signals = signals[signals["date"] >= pd.Timestamp(bt_start)]
        daily = daily[daily["date"] >= pd.Timestamp(bt_start)]
    if bt_end:
        signals = signals[signals["date"] <= pd.Timestamp(bt_end)]
        daily = daily[daily["date"] <= pd.Timestamp(bt_end)]
    logger.info(f"  backtest {bt_start or 'begin'} ~ {bt_end or 'end'}, signals={len(signals)}")

    if full_cfg.get("optimize_backtest", True):
        best_params, _ = optimize_backtest(signals, daily, full_cfg, arch)
        if best_params:
            full_cfg = {**full_cfg, **best_params}
            logger.info(f"  best params: {best_params}")

    bt = run_bt(signals, daily, full_cfg)
    metrics = bt.get_metrics()
    logger.info("=" * 50)
    logger.info(f"  total return : {metrics['total_return']:.2%}")
    logger.info(f"  annual return: {metrics['annual_return']:.2%}")
    logger.info(f"  sharpe       : {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  max drawdown : {metrics['max_drawdown']:.2%}")
    logger.info(f"  trades       : {metrics['num_trades']}")
    logger.info(f"  win rate     : {metrics['win_rate']:.2%}" if metrics["win_rate"] == metrics["win_rate"] else "  win rate     : nan")
    arch.save_backtest(bt, signals_df=signals)
    return bt


def main():
    cfg = get_config()
    event_cfg = cfg.get_section("event")
    bt_cfg = cfg.get_section("backtest")
    model_params = dict(cfg.get_section("model").get("params", {}))
    model_params.update({
        "num_leaves": 15,
        "learning_rate": 0.01,
        "n_estimators": 320,
        "min_child_samples": 60,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.65,
        "bagging_freq": 3,
        "lambda_l1": 0.2,
        "lambda_l2": 0.4,
        "max_depth": 5,
    })

    full_cfg = {
        "strategy_name": "wash_ambush",
        "limit_up_threshold": event_cfg.get("limit_up_threshold", 9.9),
        "cooldown_days": 3,
        "min_gap": 3,
        "max_gap": 30,
        "positive_min_lead": 2,
        "positive_max_lead": 6,
        "negative_sample_ratio": 3.0,
        "max_feature_samples": 90000,
        "exclude_st": event_cfg.get("exclude_st", True),
        "split_method": cfg.get("dataset.split_method", "date"),
        "test_ratio": cfg.get("dataset.test_ratio", 0.2),
        "valid_ratio": cfg.get("dataset.valid_ratio", 0.1),
        "train_end": cfg.get("dataset.train_end"),
        "valid_end": cfg.get("dataset.valid_end"),
        "test_end": cfg.get("dataset.test_end"),
        "model_dir": cfg.get("dataset.model_dir", "./models"),
        "model_params": model_params,
        "initial_cash": bt_cfg.get("initial_cash", 10_000_000),
        "topk": 3,
        "sell_n": 6,
        "slippage": bt_cfg.get("slippage", 0.003),
        "min_score": 0.60,
        "take_profit": 0.12,
        "trailing_stop": 0.05,
        "cooldown_after_stop": bt_cfg.get("cooldown_after_stop", 5),
        "scale_in_steps": 2,
        "max_code_weight": 0.12,
        "max_entry_over_first_close": 0.06,
        "max_entry_gap": 0.06,
        "max_entry_day_change_pct": 7.0,
        "optimize_backtest": True,
        "backtest_optimize": {
            "min_score_grid": [0.58, 0.64],
            "topk_grid": [2, 3],
            "sell_n_grid": [4, 6],
            "trailing_stop_grid": [0.05],
            "take_profit_grid": [0.0, 0.12],
            "max_entry_over_first_close_grid": [0.04, 0.08],
            "scale_in_steps_grid": [2],
            "min_sells": 20,
        },
        "bt_start": bt_cfg.get("start_date"),
        "bt_end": bt_cfg.get("end_date"),
    }

    tdx = TDXDataLoader(r"C:\new_tdx\vipdoc")
    if tdx.data_path is None:
        logger.error("TDX data path not found")
        return
    sh = tdx.load_stock_list("sh")
    sz = tdx.load_stock_list("sz")
    codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    logger.info(f"loading {len(codes)} main-board stocks")
    daily = tdx.load_batch(codes=codes, start_date="2022-01-01", end_date="2024-12-31")
    if daily.empty:
        logger.error("empty daily data")
        return
    daily["date"] = pd.to_datetime(daily["date"])

    arch = RunArchive(base_dir=full_cfg["model_dir"], strategy="wash_ambush")
    logger.info(f"run archive: {arch.run_dir}")

    _, samples = detect_events_and_samples(daily, full_cfg)
    if samples.empty:
        logger.error("no samples")
        return
    samples = balance_samples(samples, full_cfg)
    feats = compute_features(samples, daily)
    if feats.empty:
        logger.error("no features")
        return
    trainer, _, _ = train(feats, full_cfg, arch)
    backtest(feats, trainer, daily, full_cfg, arch)
    summary_path = arch.write_summary()
    logger.info(f"summary.json -> {summary_path}")
    logger.info(f"latest -> {arch.latest_dir}")


if __name__ == "__main__":
    main()
