# 涨停二板主升浪因子挖掘项目

> 基于通达信本地数据的 A 股量化因子挖掘项目，目标是训练能够选出"首板涨停后、能够第二次涨停、随后走出主升浪"的股票的因子模型。

---

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [运行测试](#运行测试)
- [配置说明](#配置说明)
- [开发指南](#开发指南)

---

## 项目概述

### 目标

训练一个因子模型，预测涨停股能否走出"二板 + 主升浪"形态：

```
股价
  │
  │                                    ╱ ╲
  │                                  ╱     ╲
  │                               ╱         ╲    主升浪
  │                            ╱             ╲  ╱
  │                         ╱                 ╱
  │                      ╱                   ↑
  │                   ╱                  第二次涨停
  │                ╱                    ↑
  │             ╱                       ╱
  │          ╱                        ╱
  │       ╱                         ╱
  │ ──────────────────────────────
  │         ↑
  │    首板涨停
  └─────────────────────────────────────────────→ 时间
```

### 标签定义

| 标签 | 值 | 定义 |
|------|-----|------|
| 二板失败 | 0 | 首板后 5 个交易日内未出现二板 |
| 二板成功 | 1 | 出现二板，但二板后 10 日涨幅 < 15% |
| 完美标的 | 2 | 出现二板，且二板后 10 日涨幅 ≥ 15% |

---

## 项目结构

```
limit_up_project/
├── config/
│   └── default.yaml              # 配置文件
├── requirements.txt              # 依赖清单
├── pytest.ini                   # pytest 配置
├── README.md                    # 本文档
│
├── src/                         # 源代码
│   ├── __init__.py
│   │
│   ├── utils/                   # 工具模块
│   │   ├── config.py            # 配置管理
│   │   ├── logger.py            # 日志工具
│   │   ├── calendar.py          # 交易日历
│   │   └── validator.py          # 数据校验
│   │
│   ├── data/                   # 数据层
│   │   └── tdx_loader.py          # 通达信本地数据加载器
│   │
│   ├── event/                   # 事件检测层
│   │   ├── limit_up_detector.py     # 涨停事件检测
│   │   ├── second_board_detector.py # 二板事件检测
│   │   ├── main_wave_detector.py    # 主升浪事件检测
│   │   └── event_sequence_builder.py # 事件序列构建
│   │
│   ├── feature/                 # 特征工程层
│   │   ├── price_volume.py          # 基础量价因子
│   │   └── pre_trend_features.py    # 涨停前走势特征（核心）
│   │
│   ├── label/                   # 标签构建层
│   │   └── label_builder.py         # 标签构建器
│   │
│   ├── dataset/                 # 数据集构建层
│   │   └── sector_split.py          # 板块股票划分
│   │
│   ├── model/                   # 模型训练层
│   │   ├── model_trainer.py         # LightGBM 训练器
│   │   └── model_evaluator.py       # 模型评估器
│   │
│   └── backtest/                # 回测引擎层
│       └── backtester.py            # 回测引擎
│
├── tests/                       # 单元测试
│   ├── test_event/
│   │   └── test_limit_up_detector.py
│   ├── test_feature/
│   │   └── test_price_volume.py
│   ├── test_label/
│   │   └── test_label_builder.py
│   ├── test_dataset/
│   │   └── test_sector_split.py
│   ├── test_model/
│   │   └── test_model_trainer.py
│   ├── test_backtest/
│   │   └── test_backtester.py
│   └── test_utils/
│       └── test_calendar.py
│
├── scripts/                     # 脚本
│   └── run_pipeline.py          # 运行完整流程
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_results.ipynb
│
└── data/                       # 数据目录
    └── cache/                  # 缓存目录
```

---

## 快速开始

### 1. 安装依赖

```bash
cd limit_up_project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/default.yaml`：

```yaml
data:
  cache_dir: "./data/cache"               # 缓存路径
  start_date: "2018-01-01"
  end_date: "2023-12-31"

event:
  limit_up_threshold: 9.9     # 涨停阈值%
  second_board_days: 5        # 二板判定窗口
  main_wave_days: 10         # 主升浪观察窗口
  main_wave_return: 0.15      # 主升浪涨幅阈值

feature:
  pre_trend_window_1m: 20    # 涨停前1个月窗口
  pre_trend_window_2m: 40    # 涨停前2个月窗口

model:
  type: "lightgbm"
  params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100

dataset:
  test_ratio: 0.2
  valid_ratio: 0.1

backtest:
  topk: 10
  sell_n: 5
  slippage: 0.003
```

### 3. 数据来源

本项目使用 **通达信本地数据**：

```python
from src.data.tdx_loader import TDXDataLoader

loader = TDXDataLoader()  # 自动搜索通达信安装目录
df = loader.load_day_file("000001", "sh")  # 加载单只股票
df = loader.load_batch(codes, start_date="2023-01-01", end_date="2023-12-31")  # 批量加载
```

确保通达信软件已安装并有本地数据缓存。

### 4. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_event/ -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

---

## 核心模块

### 1. 事件检测 (event/)

#### LimitUpEventDetector - 涨停事件检测

```python
from src.event.limit_up_detector import LimitUpEventDetector

detector = LimitUpEventDetector(exclude_st=True, threshold=9.9)

# 检测涨停事件
limit_up_events = detector.detect(daily_data)

# 判断单次是否涨停
is_limit_up = detector.is_limit_up(change_pct=9.95, is_st=False)
```

#### SecondBoardDetector - 二板事件检测

```python
from src.event.second_board_detector import SecondBoardDetector

detector = SecondBoardDetector(n_days=5)
second_board_events = detector.detect(limit_up_events, daily_data)
```

#### MainWaveDetector - 主升浪事件检测

```python
from src.event.main_wave_detector import MainWaveDetector

detector = MainWaveDetector(n_days=10, return_threshold=0.15)
main_wave_events = detector.detect(second_board_events, daily_data)
```

### 2. 特征工程 (feature/)

#### PreTrendFeatures - 涨停前走势特征（核心）

```python
from src.feature.pre_trend_features import PreTrendFeatures

calculator = PreTrendFeatures(window_1m=20, window_2m=40)

# 在涨停发生时点计算涨停前特征
features = calculator.calculate_at_event(
    code="000001",
    event_date="2023-01-04",
    daily_data=stock_daily_data,
)

# 转换为字典或Series
features_dict = calculator.to_dict(features)
features_series = calculator.to_series(features)
```

**涨停前走势特征包含**：

| 类别 | 特征数 | 说明 |
|------|--------|------|
| 趋势类 | 13个 | 累计收益、均线多头排列、趋势斜率 |
| 量价配合 | 7个 | 量增价涨、缩量洗盘、异动检测 |
| 波动类 | 8个 | 振幅变化、波动率、横盘天数 |
| 支撑压力 | 9个 | 价格位置、筹码分布、压力位突破 |
| 综合评分 | 4个 | 趋势得分、量能得分、形态得分 |

### 3. 标签构建 (label/)

```python
from src.label.label_builder import LabelBuilder

builder = LabelBuilder(
    second_board_days=5,
    main_wave_days=10,
    main_wave_return=0.15,
)

# 构建标签
samples = builder.build(event_data, daily_data)

# 验证标签
is_valid, errors = builder.validate_labels(samples)
```

### 4. 数据划分 (dataset/)

```python
from src.dataset.sector_split import SectorStockSplitter

splitter = SectorStockSplitter(
    test_ratio=0.2,
    valid_ratio=0.1,
    random_seed=42,
)

# 按时序划分
splits = splitter.split_by_time(df)

# 按板块划分
splits = splitter.split_by_sector(df)

# 按股票划分
splits = splitter.split_by_stock(df)

# 联合划分（最严格）
splits = splitter.split_by_sector_and_stock(df)

# 交叉验证
for result in splitter.cross_validate_by_sector(df, n_splits=5):
    X_train, y_train = result["train"]
    X_test, y_test = result["test"]
```

### 5. 模型训练 (model/)

```python
from src.model.model_trainer import LimitUpModelTrainer
from src.model.model_evaluator import ModelEvaluator

# 训练
trainer = LimitUpModelTrainer(config)
trainer.train(X_train, y_train, X_valid, y_valid)

# 预测
predictions = trainer.predict(X_test)
probas = trainer.predict_proba(X_test)

# 评估
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_test, predictions, probas)
print(evaluator.get_metrics_summary(metrics))
```

### 6. 回测引擎 (backtest/)

```python
from src.backtest.backtester import Backtester

backtester = Backtester(
    initial_cash=10000000,
    topk=10,
    sell_n=5,
    slippage=0.003,
)

# 运行回测
report = backtester.run(signals, daily_data)

# 获取指标
metrics = backtester.get_metrics()
print(f"年化收益: {metrics['annual_return']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
```

---

## 运行测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定模块测试

```bash
# 事件检测
pytest tests/test_event/ -v

# 特征工程
pytest tests/test_feature/ -v

# 标签构建
pytest tests/test_label/ -v

# 数据集划分
pytest tests/test_dataset/ -v

# 模型训练
pytest tests/test_model/ -v

# 回测引擎
pytest tests/test_backtest/ -v

# 工具函数
pytest tests/test_utils/ -v
```

### 生成覆盖率报告

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

---

## 配置说明

### 事件配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `limit_up_threshold` | 9.9 | 涨停阈值（%） |
| `exclude_st` | true | 是否排除ST股 |
| `second_board_days` | 5 | 首板后二板判定窗口（交易日） |
| `main_wave_days` | 10 | 主升浪观察窗口（交易日） |
| `main_wave_return` | 0.15 | 主升浪涨幅阈值 |

### 特征配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pre_trend_window_1m` | 20 | 涨停前1个月窗口（交易日） |
| `pre_trend_window_2m` | 40 | 涨停前2个月窗口（交易日） |

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_leaves` | 31 | LightGBM 叶子数 |
| `learning_rate` | 0.05 | 学习率 |
| `n_estimators` | 100 | 树数量 |

### 回测配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_cash` | 10000000 | 初始资金 |
| `topk` | 10 | 持仓最多N只 |
| `sell_n` | 5 | 持仓超过N日强制卖出 |
| `slippage` | 0.003 | 滑点率 |

---

## 开发指南

### 代码规范

- 遵循 PEP 8
- 使用 type hints 类型注解
- 每个模块都要有单元测试
- 测试覆盖率目标 ≥ 85%

### Git 提交规范

```
<type>(<scope>): <subject>

Types:
  - feat: 新功能
  - fix: Bug 修复
  - refactor: 重构
  - test: 测试
  - docs: 文档
  - chore: 构建/工具
```

### 添加新特征

1. 在 `src/feature/` 下创建新文件
2. 实现特征计算逻辑
3. 在 `tests/test_feature/` 下添加测试
4. 更新本文档

### 添加新模型

1. 在 `src/model/` 下创建新文件
2. 实现 `train()` 和 `predict()` 接口
3. 在 `tests/test_model/` 下添加测试
4. 更新 `model_trainer.py` 支持新模型

---

## 数据说明

### 最小数据字段

```python
required_fields = [
    "date",           # 日期
    "code",           # 股票代码（6位）
    "open",           # 开盘价
    "high",           # 最高价
    "low",            # 最低价
    "close",          # 收盘价
    "volume",         # 成交量
    "turnover_rate",  # 换手率
    "change_pct",     # 涨跌幅（%）
]
```

### 数据来源

| 来源 | 说明 |
|------|------|
| 通达信本地数据 | 本项目默认使用，需安装通达信软件 |
| AKShare | 免费财经数据 |
| Tushare | 需要积分 |

---

## 风险提示

| 风险类型 | 说明 | 应对 |
|----------|------|------|
| 过拟合 | 历史规律未必持续 | 多年份、多市场验证 |
| 流动性风险 | 小盘股难以买入 | 限制市值/成交量门槛 |
| 市场风格切换 | 策略有效性可能变化 | 定期模型更新 |
| 样本量稀疏 | 涨停股本身较少 | 扩大时间范围 |

---

*项目文档更新于 2026-05-11*
