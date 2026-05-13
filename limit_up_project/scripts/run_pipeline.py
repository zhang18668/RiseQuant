"""
涨停二板主升浪因子挖掘 - 完整流程脚本

该脚本演示了从数据加载、事件检测、特征计算、模型训练到回测的完整流程。

用法:
    python scripts/run_pipeline.py
"""
import sys
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

# 回测
from src.backtest.backtester import Backtester

# 数据加载
from src.data.tdx_loader import TDXDataLoader, load_tdx_data

# 设置日志
setup_logger(log_level="INFO")
logger = get_logger(__name__)


def generate_sample_data(
    n_stocks: int = 100,
    n_days: int = 500,
    start_date: str = "2022-01-01",
) -> pd.DataFrame:
    """
    生成模拟日线数据（用于演示）

    实际使用时替换为真实数据

    Args:
        n_stocks: 股票数量
        n_days: 交易日天数
        start_date: 起始日期

    Returns:
        pd.DataFrame: 模拟日线数据
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=n_days, freq="B")
    codes = [f"{i:06d}" for i in range(n_stocks)]

    records = []

    for code in codes:
        # 初始化价格
        price = np.random.uniform(10, 100)
        prices = [price]
        volumes = [np.random.uniform(1e6, 1e8)]

        # 生成随机游走价格
        for i in range(1, n_days):
            # 偶尔加入涨停
            if np.random.random() < 0.01 and i > 20:  # 约1%概率涨停
                change_pct = np.random.uniform(9.9, 10.0)
            else:
                change_pct = np.random.normal(0, 2)

            price = price * (1 + change_pct / 100)
            volume = np.random.uniform(1e6, 1e8)

            prices.append(price)
            volumes.append(volume)

        # 生成每日数据
        open_prices = [p * np.random.uniform(0.95, 1.0) for p in prices]
        high_prices = [p * np.random.uniform(1.0, 1.05) for p in prices]
        low_prices = [p * np.random.uniform(0.95, 1.0) for p in prices]

        for i, date in enumerate(dates):
            change_pct = (prices[i] / prices[i-1] - 1) * 100 if i > 0 else 0

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "open": round(open_prices[i], 2),
                "high": round(high_prices[i], 2),
                "low": round(low_prices[i], 2),
                "close": round(prices[i], 2),
                "volume": volumes[i],
                "turnover_rate": np.random.uniform(1, 10),
                "change_pct": round(change_pct, 2),
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} records for {n_stocks} stocks over {n_days} days")

    return df


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
    logger.info("Step 5: 计算涨停前走势特征")
    logger.info("=" * 50)

    pre_trend_calc = PreTrendFeatures(
        window_1m=config.get("pre_trend_window_1m", 20),
        window_2m=config.get("pre_trend_window_2m", 40),
    )

    features_list = []

    # 遍历样本计算特征
    for idx, row in samples.iterrows():
        if idx % 100 == 0:
            logger.info(f"计算特征进度: {idx}/{len(samples)}")

        code = row["code"]
        first_date = row["first_date"]

        # 获取该股票历史数据
        stock_data = daily_data[daily_data["code"] == code].sort_values("date")

        # 计算涨停前特征
        features = pre_trend_calc.calculate_at_event(
            code=code,
            event_date=first_date,
            daily_data=stock_data,
        )

        feature_dict = pre_trend_calc.to_dict(features)
        feature_dict["sample_id"] = row["sample_id"]
        feature_dict["code"] = code
        feature_dict["first_date"] = first_date
        feature_dict["label_combined"] = row["label_combined"]
        feature_dict["label_short"] = row["label_short"]

        features_list.append(feature_dict)

    features_df = pd.DataFrame(features_list)
    logger.info(f"计算完成，共 {len(features_df)} 个样本的特征")

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

    # 划分数据集
    splitter = SectorStockSplitter(
        test_ratio=config.get("test_ratio", 0.2),
        valid_ratio=config.get("valid_ratio", 0.1),
    )
    splits = splitter.split_by_time(features_df, date_col="first_date")

    X_train = splits["train"][feature_cols].fillna(0)
    y_train = splits["train"]["label_combined"]
    X_valid = splits["valid"][feature_cols].fillna(0)
    y_valid = splits["valid"]["label_combined"]
    X_test = splits["test"][feature_cols].fillna(0)
    y_test = splits["test"]["label_combined"]

    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_valid)}, 测试集: {len(X_test)}")

    # 训练模型
    model_config = config.get("model", {})
    trainer = LimitUpModelTrainer(model_config)
    trainer.train(X_train, y_train, X_valid, y_valid, feature_names=feature_cols)

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

    return trainer, evaluator, X_test, y_test


def run_backtest(
    features_df: pd.DataFrame,
    trainer,
    daily_data: pd.DataFrame,
    config: dict,
) -> None:
    """
    运行回测

    Args:
        features_df: 特征表
        trainer: 训练好的模型
        daily_data: 日线数据
        config: 配置
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

    # 运行回测
    backtester = Backtester(
        initial_cash=config.get("initial_cash", 10000000),
        topk=config.get("topk", 10),
        sell_n=config.get("sell_n", 5),
        slippage=config.get("slippage", 0.003),
    )

    report = backtester.run(signals, daily_data)

    if report.empty:
        logger.warning("回测无结果（可能是信号为空）")
        return

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


def main():
    """主流程"""
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
        "test_ratio": config.get("dataset.test_ratio", 0.2),
        "valid_ratio": config.get("dataset.valid_ratio", 0.1),
    }

    # Step 0: 加载数据
    logger.info("=" * 50)
    logger.info("Step 0: 加载数据")
    logger.info("=" * 50)

    # 使用通达信本地数据
    try:
        tdx_loader = TDXDataLoader(r"C:\new_tdx\vipdoc")
        logger.info("使用通达信本地数据...")

        if tdx_loader.data_path is None:
            logger.error("未找到通达信数据目录，请检查安装路径")
            logger.info("请设置通达信数据路径，或使用其他数据源")
            return

        # 获取股票列表（从上海和深圳）
        stock_list_sh = tdx_loader.load_stock_list("sh")
        stock_list_sz = tdx_loader.load_stock_list("sz")

        # 过滤主板股票（涨跌幅限制10%）
        # 主板: 600xxx, 601xxx, 603xxx (上海), 000xxx, 001xxx (深圳)
        # 排除: 688xxx (科创板), 300xxx (创业板)
        main_codes = []
        for code in stock_list_sh["code"].tolist():
            if tdx_loader.is_main_board(code):
                main_codes.append(code)
        for code in stock_list_sz["code"].tolist():
            if tdx_loader.is_main_board(code):
                main_codes.append(code)

        logger.info(f"主板股票数量: {len(main_codes)}")

        # 全部加载（建议根据实际数据量调整）
        logger.info(f"准备加载 {len(main_codes)} 只股票数据...")

        # 批量加载日线数据
        daily_data = tdx_loader.load_batch(
            codes=main_codes,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        if daily_data.empty:
            logger.error("通达信数据为空，请检查数据目录")
            return

        logger.info(f"加载了 {len(daily_data)} 条日线数据")

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

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
    trainer, evaluator, X_test, y_test = train_model(features_df, full_config)

    # Step 7: 回测
    run_backtest(features_df, trainer, daily_data, full_config)

    logger.info("=" * 60)
    logger.info("流程完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
