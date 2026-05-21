"""
涨停二板主升浪因子挖掘 - 完整流程脚本

该脚本演示了从数据加载、事件检测、特征计算、模型训练到回测的完整流程。

用法:
    python scripts/run_pipeline.py
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import setup_logger, get_logger
from src.utils.calendar import TradingCalendar

# 事件检测
from src.event.limit_up_detector import LimitUpEventDetector
from src.event.second_board_detector import SecondBoardDetector
from src.event.main_wave_detector import MainWaveDetector
from src.event.event_sequence_builder import EventSequenceBuilder

# 特征
from src.feature.pre_trend_features import PreTrendFeatures

# 标签
from src.label.label_builder import LabelBuilder

# 数据集
from src.dataset.sector_split import SectorStockSplitter

# 模型
from src.model.model_trainer import LimitUpModelTrainer
from src.model.model_evaluator import ModelEvaluator
from src.model.model_registry import ModelRegistry

# 回测
from src.backtest.backtester import Backtester

# 数据加载
from src.data.pipeline_loader import load_daily_data

# 防未来函数校验器 + 运行档案
from src.utils.validator import LookAheadValidator
from src.utils.run_archive import RunArchive

# 设置日志
setup_logger(log_level="INFO")
logger = get_logger(__name__)


def detect_events(daily_data: pd.DataFrame, config: dict) -> dict:
    """
    检测事件（涨停、二板、主升浪）

    Args:
        daily_data: 日线数据
        config: 配置

    Returns:
        dict: 事件字典
    """
    logger.info("=" * 50)
    logger.info("Step 1: 检测涨停事件")
    logger.info("=" * 50)

    # 涨停检测
    limit_up_detector = LimitUpEventDetector(
        exclude_st=config.get("exclude_st", True),
        threshold=config.get("limit_up_threshold", 9.9),
    )
    limit_up_events = limit_up_detector.detect(daily_data)
    logger.info(f"检测到 {len(limit_up_events)} 个涨停事件")

    # 过滤首板
    first_board_events = limit_up_events[
        limit_up_events["limit_up_type"] == "first_board"
    ].copy()
    logger.info(f"其中 {len(first_board_events)} 个首板事件")

    logger.info("=" * 50)
    logger.info("Step 2: 检测二板事件")
    logger.info("=" * 50)

    # 二板检测
    second_board_detector = SecondBoardDetector(
        n_days=config.get("second_board_days", 5),
    )
    second_board_events = second_board_detector.detect(
        limit_up_events, daily_data
    )
    logger.info(f"检测到 {len(second_board_events)} 个二板事件")

    logger.info("=" * 50)
    logger.info("Step 3: 检测主升浪事件")
    logger.info("=" * 50)

    # 主升浪检测
    main_wave_detector = MainWaveDetector(
        n_days=config.get("main_wave_days", 10),
        return_threshold=config.get("main_wave_return", 0.15),
    )
    main_wave_events = main_wave_detector.detect(
        second_board_events, daily_data
    )
    logger.info(f"检测到 {len(main_wave_events)} 个主升浪事件")

    logger.info("=" * 50)
    logger.info("Step 4: 构建事件序列")
    logger.info("=" * 50)

    # 事件序列构建
    sequence_builder = EventSequenceBuilder()
    samples = sequence_builder.build(
        first_board_events,
        second_board_events,
        main_wave_events,
    )
    logger.info(f"构建了 {len(samples)} 个样本")
    logger.info(f"样本分布:\n{samples['label_combined'].value_counts().sort_index()}")

    return {
        "limit_up_events": limit_up_events,
        "first_board_events": first_board_events,
        "second_board_events": second_board_events,
        "main_wave_events": main_wave_events,
        "samples": samples,
    }


def calculate_features(
    samples: pd.DataFrame,
    daily_data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    计算特征

    Args:
        samples: 样本表
        daily_data: 日线数据
        config: 配置

    Returns:
        pd.DataFrame: 带特征的样本
    """
    logger.info("=" * 50)
    logger.info("Step 5: 计算特征 (PreTrend + Technical + Market)")
    logger.info("=" * 50)

    from src.feature.feature_calculator import FeatureCalculator
    fc = FeatureCalculator(config={
        "feature": {
            "pre_trend_window_1m": config.get("pre_trend_window_1m", 20),
            "pre_trend_window_2m": config.get("pre_trend_window_2m", 40),
            "use_technical": config.get("use_technical", True),
            "use_market":    config.get("use_market", True),
        },
        "event": {"limit_up_threshold": config.get("limit_up_threshold", 9.9)},
    })

    # 准备 events 表 (FeatureCalculator 需要 event_date 列)
    events = samples[["sample_id", "code", "first_date", "label_short", "label_combined"]].copy()
    events = events.rename(columns={"first_date": "event_date"})

    features_df = fc.calculate(daily_data, events)
    if features_df.empty:
        logger.warning("特征表为空")
        return features_df

    # 还原列名: event_date -> first_date, 与下游一致
    features_df = features_df.rename(columns={"event_date": "first_date"})
    n_feat = sum(1 for c in features_df.columns if c.startswith("f_"))
    logger.info(f"计算完成: {len(features_df)} 个样本 x {n_feat} 维特征")
    return features_df


def train_model(
    features_df: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    训练模型

    Args:
        features_df: 特征表
        config: 配置

    Returns:
        tuple: (model, trainer, X_test, y_test)
    """
    logger.info("=" * 50)
    logger.info("Step 6: 训练模型")
    logger.info("=" * 50)

    # 准备特征列（排除非特征列）
    exclude_cols = ["sample_id", "code", "first_date", "label_combined", "label_short"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # 处理缺失值
    X = features_df[feature_cols].fillna(0)
    y = features_df["label_combined"]

    # 划分数据集 (按时间序, 杜绝未来函数)
    splitter = SectorStockSplitter(
        test_ratio=config.get("test_ratio", 0.2),
        valid_ratio=config.get("valid_ratio", 0.1),
    )
    split_method = (config.get("split_method") or "ratio").lower()
    if split_method == "date":
        train_end = config.get("train_end")
        valid_end = config.get("valid_end")
        test_end  = config.get("test_end")
        logger.info(
            f"split_method=date: train_end={train_end}, valid_end={valid_end}, test_end={test_end}"
        )
        splits = splitter.split_by_date_ranges(
            features_df,
            date_col="first_date",
            train_end=train_end,
            valid_end=valid_end,
            test_end=test_end,
        )
    else:
        logger.info(
            f"split_method=ratio: test_ratio={config.get('test_ratio', 0.2)}, "
            f"valid_ratio={config.get('valid_ratio', 0.1)}"
        )
        splits = splitter.split_by_time(features_df, date_col="first_date")

    # 防未来函数: 校验 train.max < valid.min < test.min, 校验特征列命名
    LookAheadValidator.assert_no_time_leakage(splits, date_col="first_date")
    LookAheadValidator.scan_feature_names(feature_cols, raise_error=True)

    X_train = splits["train"][feature_cols].fillna(0)
    y_train = splits["train"]["label_combined"]
    X_valid = splits["valid"][feature_cols].fillna(0)
    y_valid = splits["valid"]["label_combined"]
    X_test = splits["test"][feature_cols].fillna(0)
    y_test = splits["test"]["label_combined"]

    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_valid)}, 测试集: {len(X_test)}")
    logger.info(
        f"时间区间: train [{splits['train']['first_date'].min()} ~ {splits['train']['first_date'].max()}], "
        f"valid [{splits['valid']['first_date'].min()} ~ {splits['valid']['first_date'].max()}], "
        f"test  [{splits['test']['first_date'].min()} ~ {splits['test']['first_date'].max()}]"
    )

    # 训练模型
    model_config = config.get("model", {})
    trainer = LimitUpModelTrainer(model_config)
    trainer.train(X_train, y_train, X_valid, y_valid, feature_names=feature_cols)

    # 保存模型 + 训练元信息
    from datetime import datetime
    from pathlib import Path
    import json

    model_dir = config.get("model_dir", "models")
    registry = ModelRegistry(model_dir)

    # 构建 meta 信息
    meta = {
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "feature_cols": list(feature_cols),
        "params": trainer.params,
        "train_range": [str(splits["train"]["first_date"].min()), str(splits["train"]["first_date"].max())],
        "valid_range": [str(splits["valid"]["first_date"].min()), str(splits["valid"]["first_date"].max())] if len(splits["valid"]) else None,
        "test_range": [str(splits["test"]["first_date"].min()), str(splits["test"]["first_date"].max())] if len(splits["test"]) else None,
    }

    # 保存到版本注册表
    version = registry.save_version(
        model=trainer,
        meta=meta,
        config=config,
    )
    logger.info(f"模型已保存为版本: {version}")
    logger.info(f"latest 指针: {registry.get_latest_version()}")

    # 评估
    evaluator = ModelEvaluator()
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)

    metrics = evaluator.evaluate_multiclass(y_test.values, y_pred, y_pred_proba)

    logger.info("=" * 50)
    logger.info("模型评估结果:")
    logger.info("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if abs(value) < 10:
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    # 特征重要性
    importance_df = trainer.get_feature_importance()
    logger.info("=" * 50)
    logger.info("Top 10 重要特征:")
    logger.info("=" * 50)
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return trainer, evaluator, X_test, y_test, version


def run_backtest(
    features_df: pd.DataFrame,
    trainer,
    daily_data: pd.DataFrame,
    config: dict,
    version: str = None,
) -> tuple:
    """
    运行回测

    Args:
        features_df: 特征表
        trainer: 训练好的模型
        daily_data: 日线数据
        config: 配置
        version: 版本号（用于保存回测结果）

    Returns:
        tuple: (backtester, metrics)
    """
    logger.info("=" * 50)
    logger.info("Step 7: 运行回测")
    logger.info("=" * 50)

    # 生成信号
    feature_cols = [c for c in features_df.columns
                   if c not in ["sample_id", "code", "first_date", "label_combined", "label_short"]]

    # 预测概率
    X = features_df[feature_cols].fillna(0)
    proba = trainer.predict_proba(X)

    # 信号：取"完美标的"概率
    signals = features_df[["first_date", "code"]].copy()
    signals["score"] = proba[:, 2]  # label=2 的概率
    signals = signals.rename(columns={"first_date": "date"})

    # 按回测年份过滤信号与日线数据
    bt_start = config.get("start_date")
    bt_end   = config.get("end_date")
    if bt_start or bt_end:
        signals["date"] = pd.to_datetime(signals["date"])
        daily_data = daily_data.copy()
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        if bt_start:
            signals    = signals[signals["date"]    >= pd.Timestamp(bt_start)]
            daily_data = daily_data[daily_data["date"] >= pd.Timestamp(bt_start)]
        if bt_end:
            signals    = signals[signals["date"]    <= pd.Timestamp(bt_end)]
            daily_data = daily_data[daily_data["date"] <= pd.Timestamp(bt_end)]
        logger.info(
            f"回测窗口: {bt_start or 'beginning'} ~ {bt_end or 'end'} | "
            f"信号 {len(signals)} 条, 日线 {len(daily_data)} 条"
        )

    # 运行回测 — entry_delay=1 表示 T 日产生信号、T+1 开盘成交, 杜绝未来函数
    backtester = Backtester(
        initial_cash=config.get("initial_cash", 10000000),
        topk=config.get("topk", 10),
        sell_n=config.get("sell_n", 5),
        slippage=config.get("slippage", 0.003),
        entry_delay=config.get("entry_delay", 1),
    )
    logger.info(
        f"Backtester entry_delay={backtester.entry_delay} "
        f"(信号日 T 在 T+{backtester.entry_delay} 开盘成交, 杜绝未来函数)"
    )

    report = backtester.run(signals, daily_data)

    if report.empty:
        logger.warning("回测无结果（可能是信号为空）")
        return None, None

    # 输出回测指标
    metrics = backtester.get_metrics()

    logger.info("=" * 50)
    logger.info("回测结果:")
    logger.info("=" * 50)
    logger.info(f"  初始资金: {metrics.get('initial_cash', 0):,.0f}")
    logger.info(f"  最终净值: {metrics.get('final_value', 0):,.0f}")
    logger.info(f"  总收益率: {metrics.get('total_return', 0):.2%}")
    logger.info(f"  年化收益率: {metrics.get('annual_return', 0):.2%}")
    logger.info(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  交易次数: {metrics.get('num_trades', 0)}")

    # 如果有版本号，保存回测结果（复用上面已计算的 signals）
    if version:
        registry = ModelRegistry(config.get("model_dir", "models"))
        archive_dir = registry.update_version_with_backtest(
            version=version, backtester=backtester,
            metrics=metrics, signals_df=signals,
        )
        if archive_dir:
            logger.info(f"回测产物已保存到: {archive_dir}")

    return backtester, metrics


def main():
    """主流程"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="涨停二板主升浪因子挖掘")
    parser.add_argument("--version", type=str, default=None,
                        help="指定版本号加载模型进行回测，如 v001")
    parser.add_argument("--list-versions", action="store_true",
                        help="列出所有已保存的版本")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="模型保存目录")
    parser.add_argument("--source", choices=["cache", "tdx", "auto"], default="cache",
                        help="日线数据源：cache（默认，仅读 daily_cache） / tdx（直读通达信） / auto（cache→tdx 回退）")
    args = parser.parse_args()

    # 列出版本
    if args.list_versions:
        registry = ModelRegistry(args.model_dir)
        registry.list_versions()
        return

    # 加载指定版本
    if args.version:
        registry = ModelRegistry(args.model_dir)
        version_info = registry.load_version(args.version)
        if not version_info:
            logger.error(f"版本 {args.version} 不存在")
            return
        logger.info(f"加载版本: {args.version}")
        logger.info(f"训练时间: {version_info.get('trained_at')}")
        logger.info(f"模型文件: {version_info.get('model_path')}")
        # 加载模型和配置进行回测
        trainer = LimitUpModelTrainer.load(version_info.get('model_path'))
        # TODO: 使用保存的配置进行回测
        logger.info("版本加载成功，可使用 --list-versions 查看所有版本")
        return

    logger.info("=" * 60)
    logger.info("涨停二板主升浪因子挖掘 - 完整流程")
    logger.info("=" * 60)

    # 加载配置
    config = get_config()
    event_config = config.get_section("event")
    feature_config = config.get_section("feature")
    model_config = config.get_section("model")
    backtest_config = config.get_section("backtest")

    # 合并配置
    full_config = {
        **event_config,
        **feature_config,
        "model": model_config,
        **backtest_config,
        # dataset 切分
        "split_method": config.get("dataset.split_method", "ratio"),
        "test_ratio":   config.get("dataset.test_ratio", 0.2),
        "valid_ratio":  config.get("dataset.valid_ratio", 0.1),
        "train_end":    config.get("dataset.train_end"),
        "valid_end":    config.get("dataset.valid_end"),
        "test_end":     config.get("dataset.test_end"),
        # 模型目录
        "model_dir":    config.get("dataset.model_dir", "models"),
    }

    # Step 0: 加载数据
    logger.info("=" * 50)
    logger.info(f"Step 0: 加载数据 (source={args.source})")
    logger.info("=" * 50)
    try:
        daily_data = load_daily_data(config, source=args.source)
    except RuntimeError as exc:
        logger.error(f"加载数据失败: {exc}")
        return
    logger.info(
        f"日线数据准备就绪: {len(daily_data)} 条 × "
        f"{daily_data['code'].nunique()} 只代码"
    )

    # Step 1-4: 事件检测
    events = detect_events(daily_data, full_config)
    samples = events["samples"]

    if len(samples) == 0:
        logger.warning("没有检测到任何样本，请检查数据或调整参数")
        return

    # Step 5: 计算特征
    features_df = calculate_features(samples, daily_data, full_config)

    if features_df.empty:
        logger.warning("特征计算失败")
        return

    # Step 6: 训练模型
    trainer, evaluator, X_test, y_test, version = train_model(features_df, full_config)

    # Step 7: 回测
    run_backtest(features_df, trainer, daily_data, full_config, version=version)

    logger.info("=" * 60)
    logger.info("流程完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
