# 运行指南 — 涨停二板主升浪因子挖掘

本文档面向第一次接触本项目的开发/研究人员，目标是让你在 30 分钟内：

1. 把环境跑起来
2. 跑通 smoke 测试，验证项目可用
3. 跑通完整 pipeline，看到一次端到端的结果
4. 学会按需调参（窗口、阈值、回测撮合时点、切分策略）

> 项目根目录约定为 `G:\AI\RiseQuant\limit_up_project`。后文 `$ROOT` 即指此目录。

---

## 1. 环境准备

### 1.1 系统要求

- Python 3.10+（项目在 3.10 上验证）
- Windows（通达信本地数据是 Windows 路径，Linux/macOS 需自备数据源）
- 8 GB 内存以上（全市场主板回测推荐 16 GB）

### 1.2 安装依赖

```bash
cd G:\AI\RiseQuant\limit_up_project
pip install -r requirements.txt
```

`requirements.txt` 中的关键依赖：

| 依赖 | 用途 |
|---|---|
| pandas, numpy | 数据处理 |
| lightgbm | 模型训练 |
| scikit-learn | 评估指标 |
| pyyaml | 配置加载 |
| loguru | 日志 |
| pytest | 测试 |

如果只想跑核心流程而暂不训练模型，可以不装 lightgbm — 训练步骤会显式抛 `ImportError`，事件检测、特征、回测等模块仍可独立运行。

### 1.3 设置 PYTHONPATH

项目使用 `from src.xxx import xxx` 风格的绝对导入，需要把项目根目录加入 `PYTHONPATH`。

Windows CMD：

```bat
cd G:\AI\RiseQuant\limit_up_project
set PYTHONPATH=%CD%
```

PowerShell：

```powershell
cd G:\AI\RiseQuant\limit_up_project
$env:PYTHONPATH = (Get-Location).Path
```

Linux/macOS：

```bash
cd G:\AI\RiseQuant\limit_up_project
export PYTHONPATH=$PWD
```

后续所有命令默认你已经做完这一步。

---

## 2. 验证安装（防未来函数 smoke 测试）

最快的验证方式：跑反未来函数 smoke 测试，**10 项校验全过**说明核心模块可用。

```bash
python tests\_smoke_no_lookahead.py
```

预期看到如下输出尾部：

```
  [OK] entry_delay=1: 信号在 T+1 开盘成交 (T+1 open=20.0)
  [OK] entry_delay=0 触发未来函数警告
  [OK] 按股票随机切分时, 启用 date_col 会 raise look-ahead leakage
  [OK] split_by_time: train.max < valid.min < test.min
  [OK] purged_kfold_by_time: 任一训练样本距 test 边界 > embargo_days
  [OK] split_by_index(shuffle=True) 触发未来函数警告
  [OK] split_by_index(shuffle=False) 严格时间序
  [OK] assert_no_time_leakage 检测出 train.max > test.min
  [OK] scan_feature_names 命中 'future_return'
  [OK] assert_features_before_signal 检测出 feature_date > signal_date

ALL PASS — 训练管线已杜绝未来函数
```

任何一项 FAIL 都意味着代码或环境出问题，需先解决后再继续。

---

## 3. 跑既有单元测试

```bash
pytest tests\ -v
```

也可以挑特定模块：

```bash
pytest tests\test_backtest -v
pytest tests\test_dataset -v
pytest tests\test_event -v
```

测试期间会看到形如：

```
UserWarning: 未来函数风险: 按 key 随机切分时未提供 date_col, ...
UserWarning: split_by_index(shuffle=True) 会打乱时间序, ...
```

这些是 **预期的** 防泄漏警告，不是错误。

---

## 3.5 数据缓存构建（阶段 1.x）

完整 pipeline 依赖三层数据缓存，建议按顺序首次构建。后续每日只需重跑同样命令即可增量更新。

```
阶段 1.2  涨停板池缓存       python scripts/build_zt_pool_cache.py
阶段 1.3  日线数据缓存       python scripts/build_daily_cache.py
阶段 1.4  数据一致性校验     python scripts/verify_daily_cache.py
```

**阶段 1.2 涨停板池缓存**：基于 AkShare 拉取 ``stock_zt_pool_em`` 与 ``stock_zt_pool_previous_em``，落地到 ``data/zt_pool_cache/{zt_pool, zt_pool_previous}/YYYYMMDD.parquet``。默认 2020-01-01 至今，首次约 35 分钟，日增量约 5 秒。

**阶段 1.3 日线数据缓存**：默认 universe 为 ``zt_pool_relevant``（两个池里出现过的主板代码并集，~2000-3000 只），用通达信本地 ``.day`` 文件优先、AkShare 兜底，落地到 ``data/daily_cache/<code>.parquet``。常用命令：

```bash
# 默认：增量更新涨停股相关的日线
python scripts/build_daily_cache.py

# 指定数据源 / TDX 路径
python scripts/build_daily_cache.py --source tdx --tdx-path C:/new_tdx/vipdoc

# 全市场主板（更耗时，但事件检测/特征不受限）
python scripts/build_daily_cache.py --universe all
```

**阶段 1.4 一致性校验**：对账 zt_pool 与日线缓存，覆盖度、价格一致性、涨跌幅一致性、首次封板时间分布、空文件连续段、未来函数自检。任一项未达标 ``exit 2``，输出 ``data/verify/daily_consistency_<ts>.{json,md}``。

```bash
python scripts/verify_daily_cache.py
```

只有这一阶段返回 PASS，才进入下面的端到端 pipeline。

---

## 4. 跑完整 pipeline（端到端）

### 4.1 默认数据源：通达信本地数据

`scripts\run_pipeline.py` 默认从 `C:\new_tdx\vipdoc` 读取通达信日线数据。如果你已安装通达信并下载了主板股票的历史日线，直接运行：

```bash
python scripts\run_pipeline.py
```

如果通达信不在默认路径，修改 `scripts\run_pipeline.py` 中的：

```python
tdx_loader = TDXDataLoader(r"C:\new_tdx\vipdoc")
```

把字符串替换为你实际的 vipdoc 路径。

### 4.2 没有通达信怎么办

最简单的替代：用 `scripts\run_pipeline.py` 里自带的 `generate_sample_data` 函数生成模拟数据。把 `main()` 里 "Step 0: 加载数据" 一段替换为：

```python
daily_data = generate_sample_data(n_stocks=100, n_days=500, start_date="2022-01-01")
```

或者使用 `src/data/akshare_loader.py` 接入 AKShare（需要联网）。

### 4.3 流程会做什么

依次执行（控制台会打印每一步的进度）：

1. **加载数据**：日线 OHLCV
2. **事件检测**：涨停 → 首板筛选 → 二板 → 主升浪
3. **样本构建**：把三个事件表 join 成"一次首板事件 = 一行样本"，附 `label_short / label_combined`
4. **特征计算**：在每个 first_date 上回看 1m/2m 窗口，输出 ~25 维量价 + 形态特征
5. **数据集切分**：`split_by_time`（train/valid/test 严格时间序），并调用 `LookAheadValidator` 做泄漏断言
6. **模型训练**：LightGBM 多分类（label_combined ∈ {0, 1, 2}）
7. **回测**：Backtester(entry_delay=1)，T 日信号在 T+1 开盘等权买入，持仓 sell_n 天后平仓

### 4.4 关键日志

```
训练集: 1234, 验证集: 234, 测试集: 234
时间区间: train [2022-01-04 ~ 2023-04-15], valid [2023-04-17 ~ 2023-07-20], test [2023-07-21 ~ 2023-12-31]
trained LGBM with 1234 samples, 25 features
Backtester entry_delay=1 (信号日 T 在 T+1 开盘成交, 杜绝未来函数)
```

如果看到 `entry_delay=0` 的红色 warning，说明配置里 `backtest.entry_delay` 被改成了 0，回测会包含未来函数，**应改回 1**。

---

## 5. 配置项一览（`config\default.yaml`）

```yaml
event:
  limit_up_threshold: 9.9    # 涨停阈值（%）
  exclude_st: true           # 排除 ST
  second_board_days: 5       # 首板后 N 个交易日内出现下一次涨停才算二板
  main_wave_days: 10         # 二板后 N 个交易日窗口内最大涨幅是否达标
  main_wave_return: 0.15     # 主升浪门槛 15%

feature:
  pre_trend_window_1m: 20    # 涨停前 1 个月窗口
  pre_trend_window_2m: 40    # 涨停前 2 个月窗口

model:
  type: "lightgbm"
  params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
    verbose: -1

dataset:
  test_ratio: 0.2
  valid_ratio: 0.1
  split_method: "time"             # 推荐严格时间序
  purged_kfold_embargo_days: 10    # K-Fold 时 train/test 之间留出的禁运天数

backtest:
  initial_cash: 10000000
  topk: 10
  sell_n: 5
  slippage: 0.003
  entry_delay: 1   # >=1 杜绝未来函数; 0 会发出警告
```

修改参数后再次 `python scripts\run_pipeline.py` 即可。

---

## 6. 常见运维操作

### 6.1 仅做事件检测，导出二板样本

```python
from src.event.event_sequence_builder import EventSequenceBuilder
import pandas as pd

daily = pd.read_csv("your_daily.csv")  # 列 date,code,open,high,low,close,volume,change_pct
builder = EventSequenceBuilder()
samples = builder.build(daily)
samples.to_csv("samples.csv", index=False)
```

### 6.2 仅做回测（已有信号）

```python
from src.backtest.backtester import Backtester
import pandas as pd

signals = pd.read_csv("signals.csv")   # 列 date,code,score
daily   = pd.read_csv("daily.csv")
bt = Backtester(entry_delay=1)         # T+1 开盘成交
bt.run(signals, daily)
print(bt.get_metrics())
print(bt.get_trades())
```

### 6.3 加防泄漏断言到任意训练脚本

```python
from src.utils.validator import LookAheadValidator

LookAheadValidator.assert_no_time_leakage(splits, date_col="first_date")
LookAheadValidator.scan_feature_names(feature_cols, raise_error=True)
LookAheadValidator.assert_features_before_signal(features, "event_date", signal_dates)
```

任何一项检测到未来函数都会 raise `ValidationError`，可直接挡在 CI 上。

### 6.4 切换切分策略

| 场景 | 调用 |
|---|---|
| 默认严格时间序 | `splitter.split_by_time(df, date_col="first_date")` |
| 按股票切但确保时间不交叉 | `splitter.split_by_stock(df, stock_col="code", date_col="first_date")` — 若有泄漏会直接 raise |
| 时间序 Purged K-Fold | `splitter.purged_kfold_by_time(df, date_col="first_date", n_splits=5, embargo_days=10)` |

---

## 7. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | 没设 PYTHONPATH，回到 1.3 节 |
| `ModuleNotFoundError: lightgbm` | 没装 lightgbm，`pip install lightgbm`；若只想跑前几步可忽略 |
| `ValidationError: look-ahead leakage` | 切分时检测到 train.max ≥ test.min，请改用 `split_by_time` 或检查 `date_col` 是否传对 |
| Warning: `entry_delay=0` | 回测配置错了，把 `backtest.entry_delay` 改回 `1` |
| `检测到 0 个涨停事件` | 数据涨幅列 `change_pct` 单位不对（应该是百分数，例如 9.9 而不是 0.099）或时间区间内本身没涨停 |
| 通达信路径找不到 | 修改 `scripts\run_pipeline.py` 里的 `TDXDataLoader` 路径，或用 `generate_sample_data` 替代 |

---

## 8. 推荐学习路径

1. 先读 `docs\ARCHITECTURE.md` — 了解模块分层与数据流
2. 跑通 `tests\_smoke_no_lookahead.py`
3. 在 PyCharm/VSCode 里打断点单步走 `scripts\run_pipeline.py`，逐步看每个产物
4. 改改 `config\default.yaml`，对比指标变化
5. 自己接入一个新特征模块（参考 `src\feature\pre_trend_features.py` 的接口约定）
