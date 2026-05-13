r"""加载已训练模型 → 重新预测 → 回测.

用法:
    set PYTHONPATH=%CD%
    python scripts\predict_and_backtest.py
    python scripts\predict_and_backtest.py --model models\limit_up_lgbm_20251201_080000.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import setup_logger, get_logger
from src.model.model_trainer import LimitUpModelTrainer
from src.backtest.backtester import Backtester
from src.data.tdx_loader import TDXDataLoader
from src.event.event_sequence_builder import EventSequenceBuilder
from src.feature.feature_calculator import FeatureCalculator

setup_logger(log_level="INFO")
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/limit_up_lgbm_latest.pkl",
                        help="已训练模型路径")
    parser.add_argument("--tdx",   default=r"C:\new_tdx\vipdoc",
                        help="通达信 vipdoc 路径")
    parser.add_argument("--start", default=None, help="覆盖回测起始日 (YYYY-MM-DD)")
    parser.add_argument("--end",   default=None, help="覆盖回测结束日 (YYYY-MM-DD)")
    parser.add_argument("--min_score",    type=float, default=None, help="概率阈值, 高于阈值才买")
    parser.add_argument("--stop_loss",    type=float, default=None, help="硬止损, 例如 0.03")
    parser.add_argument("--take_profit",  type=float, default=None, help="硬止盈, 例如 0.10")
    parser.add_argument("--trailing_stop", type=float, default=None, help="移动止盈, 例如 0.05")
    parser.add_argument("--topk", type=int, default=None)
    args = parser.parse_args()

    config = get_config()
    bt_cfg = config.get_section("backtest")
    feat_cfg = config.get_section("feature")
    ev_cfg   = config.get_section("event")

    # 1) 加载模型
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}. 请先 run_pipeline.py 训练一次.")
        return
    trainer = LimitUpModelTrainer.load(str(model_path))
    logger.info(f"已加载模型: {model_path} (特征数={len(trainer.feature_names_)})")

    # 2) 加载数据
    bt_start = args.start or bt_cfg.get("start_date") or "2023-01-01"
    bt_end   = args.end   or bt_cfg.get("end_date")   or "2024-12-31"
    # 多取一段历史以便算特征 (PreTrend 需要 40 天)
    load_start = (pd.Timestamp(bt_start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    tdx = TDXDataLoader(args.tdx)
    if tdx.data_path is None:
        logger.error(f"未找到通达信目录: {args.tdx}")
        return
    sh = tdx.load_stock_list("sh")
    sz = tdx.load_stock_list("sz")
    main_codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    logger.info(f"加载 {len(main_codes)} 只主板股票, 区间 {load_start} ~ {bt_end}")
    daily = tdx.load_batch(codes=main_codes, start_date=load_start, end_date=bt_end)
    if daily.empty:
        logger.error("数据为空")
        return

    # 3) 事件 + 特征
    builder = EventSequenceBuilder.from_config({"event": ev_cfg})
    samples = builder.build(daily)
    samples = samples[
        (samples["first_date"] >= pd.Timestamp(bt_start)) &
        (samples["first_date"] <= pd.Timestamp(bt_end))
    ]
    logger.info(f"事件样本: {len(samples)} 条")

    fc = FeatureCalculator(config={
        "feature": {
            "pre_trend_window_1m": feat_cfg.get("pre_trend_window_1m", 20),
            "pre_trend_window_2m": feat_cfg.get("pre_trend_window_2m", 40),
            "use_technical": feat_cfg.get("use_technical", True),
            "use_market":    feat_cfg.get("use_market", True),
        },
        "event": {"limit_up_threshold": ev_cfg.get("limit_up_threshold", 9.9)},
    })
    events = samples[["sample_id", "code", "first_date"]].copy()
    events = events.rename(columns={"first_date": "event_date"})
    features_df = fc.calculate(daily, events)
    if features_df.empty:
        logger.error("特征表为空")
        return
    features_df = features_df.rename(columns={"event_date": "first_date"})

    # 4) 预测
    X = features_df.reindex(columns=trainer.feature_names_, fill_value=0).fillna(0)
    proba = trainer.predict_proba(X)
    signals = features_df[["first_date", "code"]].rename(columns={"first_date": "date"}).copy()
    signals["score"] = proba[:, -1]  # 最大类别概率作为 score

    # 5) 回测 — 命令行可覆盖风控参数
    bt = Backtester(
        initial_cash=bt_cfg.get("initial_cash", 10_000_000),
        topk=args.topk if args.topk is not None else bt_cfg.get("topk", 10),
        sell_n=bt_cfg.get("sell_n", 5),
        slippage=bt_cfg.get("slippage", 0.003),
        entry_delay=bt_cfg.get("entry_delay", 1),
        stop_loss=args.stop_loss if args.stop_loss is not None else bt_cfg.get("stop_loss", 0.03),
        take_profit=args.take_profit if args.take_profit is not None else bt_cfg.get("take_profit", 0.0),
        trailing_stop=args.trailing_stop if args.trailing_stop is not None else bt_cfg.get("trailing_stop", 0.0),
        min_score=args.min_score if args.min_score is not None else bt_cfg.get("min_score", 0.0),
        skip_zhangting_open=bt_cfg.get("skip_zhangting_open", True),
    )
    logger.info(
        f"风控: stop_loss={bt.stop_loss}, take_profit={bt.take_profit}, "
        f"trailing_stop={bt.trailing_stop}, min_score={bt.min_score}, "
        f"skip_zhangting_open={bt.skip_zhangting_open}"
    )
    logger.info(f"回测窗口: {bt_start} ~ {bt_end}, entry_delay={bt.entry_delay}")
    bt.run(signals, daily)
    m = bt.get_metrics()
    logger.info("=" * 50)
    logger.info(f"总收益率   : {m['total_return']:.2%}")
    logger.info(f"年化收益率 : {m['annual_return']:.2%}")
    logger.info(f"夏普比率   : {m['sharpe_ratio']:.2f}")
    logger.info(f"最大回撤   : {m['max_drawdown']:.2%}")
    logger.info(f"交易次数   : {m['num_trades']}")
    logger.info(f"胜率       : {m['win_rate']:.2%}" if m['win_rate'] == m['win_rate'] else "胜率: nan")


if __name__ == "__main__":
    main()
