# 项目架构与实现讲解 — 涨停二板主升浪因子挖掘

> 配套文档：[`RUN_GUIDE.md`](./RUN_GUIDE.md)

本文档帮你**搞懂项目是怎么组织的、每一层在做什么、为什么这么设计**。读完这份文档后，你应当能：

- 在 5 秒内根据需求找到要改的模块
- 理解事件 → 特征 → 标签 → 模型 → 回测的数据流，以及每一步的输入输出约定
- 明白防未来函数（look-ahead bias）的关键防线在哪里、是怎么生效的

---

## 1. 项目目标（业务侧）

在 A 股主板，识别"首板涨停后能否打到二板，并在二板后 10 个交易日内拉出一波 15% 以上的主升浪"。

把它拆成监督学习问题：

- **样本** = 一次首板事件（`code, first_date`）
- **标签**：
    - `label_short`：0/1，二板是否成功
    - `label_combined`：0/1/2 — 失败 / 仅打到二板但无主升浪 / 二板且主升浪
- **特征**：在 `first_date` 当天回看 1 个月 / 2 个月窗口的量价、形态、量能配合等指标

预测目标自然导出策略：**模型给 `label=2` 概率高的票，T+1 开盘买入，持仓 N 天后卖出**。

---

## 2. 顶层目录

```
limit_up_project/
├── config/              # YAML 配置 (default.yaml)
├── scripts/             # 顶层入口 (run_pipeline.py)
├── src/
│   ├── data/            # 数据加载: 通达信 / akshare / cache
│   ├── event/           # 事件检测: 涨停 → 二板 → 主升浪
│   ├── feature/         # 特征工程: 量价 + 涨停前形态 + 板块情绪 + 资金流
│   ├── label/           # 标签构建与校验
│   ├── dataset/         # 数据集切分 (含时间安全 + Purged K-Fold)
│   ├── model/           # 模型训练 / 评估 / 特征重要性
│   ├── backtest/        # 回测引擎 / 绩效分析 / 报表
│   └── utils/           # 配置 / 日志 / 校验器 / 交易日历
├── tests/               # 单元测试 + 防泄漏 smoke 测试
└── docs/                # 本目录
```

---

## 3. 数据流总览

```
                   ┌────────────────────────┐
                   │ 日线 DataFrame (OHLCV) │
                   └──────────┬─────────────┘
                              │
              ┌───────────────▼───────────────┐
              │ LimitUpEventDetector          │  E-001
              │ → 首板/续板事件               │
              └───────────────┬───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  SecondBoardDetector   MainWaveDetector     PreTrendFeatures
  E-002                 E-003                F-005
  → 二板事件            → 主升浪事件          → 特征宽表
        │                     │                     │
        └────────┬────────────┘                     │
                 ▼                                  │
       EventSequenceBuilder                         │
       E-004                                        │
       → 样本表 (含 label_short / label_combined)   │
                 │                                  │
                 └──────────────┬───────────────────┘
                                ▼
                          CustomDataset
                          DS-002
                          + SectorStockSplitter (DS-003) 切分
                                │
                                ▼
                       LimitUpModelTrainer
                       M-001 (LightGBM)
                                │
                                ├─► ModelEvaluator (M-002)
                                ▼
                      预测概率 → 信号 (date, code, score)
                                │
                                ▼
                          Backtester
                          B-001 (entry_delay=1, T+1 开盘)
                                │
                                ▼
                         PerformanceAnalyzer
                         B-002 → 回测指标
```

每一步都有"模块代号"（E-001、F-005、M-001…）。文件头注释里也会标这个代号，方便互相引用。

---

## 4. 分层详解

### 4.1 utils — 横切关注点

| 文件 | 代号 | 职责 |
|---|---|---|
| `config.py` | U-003 | YAML 配置加载，支持 `config.get_section("event")`、`config.get("dataset.test_ratio")` |
| `logger.py` | U-002 | 基于 loguru 的统一日志 |
| `calendar.py` | U-001 | 交易日历：周末过滤 + 注入式节假日。提供 `next_trading_day(d, n)`、`count_trading_days(a, b)` |
| `validator.py` | U-004 | `DataValidator`（日线列校验、价格逻辑、重复行）+ **`LookAheadValidator`（反未来函数）** |

`LookAheadValidator` 是这次重构后最重要的"门禁"，对外 4 个 API：

- `assert_no_time_leakage(splits, date_col)` — 断言 train.max < valid.min < test.min
- `assert_features_before_signal(features, feature_date_col, signal_dates)` — 断言特征日期 ≤ 信号日
- `assert_label_after_feature(feature_dates, label_dates, min_gap_days)` — 断言标签时间在特征之后
- `scan_feature_names(names)` — 命名嗅探，凡含 `future / next_ / fwd_ / ahead / tomorrow` 等关键词的列直接 raise

任一断言失败都抛 `ValidationError`，可在 CI 上当硬卡点。

### 4.2 data — 加载层

| 文件 | 代号 | 职责 |
|---|---|---|
| `tdx_loader.py` | D-001 | 通达信本地 `.day` 二进制日线读取。每条 32 字节 (`<IIIIIfII`)，价格除以 100，自动算 `change_pct` |
| `akshare_loader.py` | D-002 | AKShare 在线行情接口的薄包装 |
| `cache_manager.py` | D-003 | 通用 parquet 缓存层（基于哈希 key） |
| `zt_pool_loader.py` | D-006 | 涨停板池（AkShare `stock_zt_pool_em` / `_previous_em`）：单日拉取、Parquet 缓存、enrich (主板、分档、封板强度) |
| `daily_cache.py` | D-007 | 按 code 维度的日线 Parquet 缓存：universe 提取、增量更新、批量加载、主源/兜底 fetcher 切换。对应阶段 1.3，由 `scripts/build_daily_cache.py` 驱动，`scripts/verify_daily_cache.py` 校验 |

加载层产出的统一 schema：

```
date | code | open | high | low | close | volume | turnover | change_pct
```

后续所有模块只认这个 schema，与具体数据源解耦。

### 4.3 event — 事件检测层（核心）

四个文件，递进定义"什么样的样本要进入训练"：

```
E-001 LimitUpEventDetector
  输入: 日线 DataFrame
  规则: change_pct >= threshold (默认 9.9%), 可选排除 ST
  输出: date | code | change_pct | is_limit_up | limit_up_type | consecutive_n
        limit_up_type ∈ {first_board, continuation}
        consecutive_n 是已连续涨停天数 (首板为 1)

E-002 SecondBoardDetector
  输入: 涨停事件表
  规则: 某只票的两次相邻涨停, 交易日间隔 in [1, n_days]
  输出: code | first_date | second_date | gap_days

E-003 MainWaveDetector
  输入: 二板事件表 + 日线
  规则: second_date 之后 n_days 个交易日内, max(close)/close[second_date] - 1 >= 15%
  输出: code | second_date | main_wave_date | period_return | is_main_wave

E-004 EventSequenceBuilder
  输入: 上面三个表 (或直接传日线一步到位)
  操作: first ⟕ second ⟕ main 三表 left-join
  输出: sample_id | code | first_date | second_date | gap_days
        | main_wave_date | period_return | is_main_wave
        | label_short | label_combined
```

**未来函数视角下的关键点：**

- E-001 仅看当日 close/change_pct，没有未来信息
- E-002 用 `rank_lookup` 跨股票统一交易日序，gap 是 *已发生* 的间隔
- E-003 的 `idx + 1 : idx + 1 + n_days` 显然是 *未来窗口*，但这是 **label 计算专属**，不会进入特征
- E-004 的 join 只是 left-join，不会偷看未来

### 4.4 feature — 特征工程层

| 文件 | 代号 | 输出列前缀 | 内容 |
|---|---|---|---|
| `price_volume.py` | F-001 | `f_ret_*`, `f_vol_*`, `f_ma_pos_*` | 收益、量比、均线位置、振幅 |
| `limit_up_features.py` | F-002 | `f_lu_count_*`, `f_lu_consec`, `f_up_consec` | 历史涨停统计、连板天数 |
| `sector_sentiment.py` | F-003 | `f_mkt_up_ratio`, `f_mkt_limit_up_count` | 市场广度（按日聚合） |
| `money_flow.py` | F-004 | `f_mf_net_*` | 资金流（代理：`sign(change_pct) * turnover`） |
| `pre_trend_features.py` | F-005 | `f_pt_*` | **核心**：涨停前 1m/2m 窗口的形态、量价配合、波动、支撑压力、综合评分 |
| `feature_calculator.py` | F-006 | — | 把上面 5 个模块按事件批量串起来 |

**特征切片的统一约定（防未来函数）：**

```python
df.iloc[event_idx + 1 - n : event_idx + 1]   # [event_idx - n + 1, event_idx], 含 event 当日
df.iloc[event_idx - n]                        # 单点, n 日前
```

均向左切，**绝不出现 `event_idx + k (k>0)`** —— 这是项目最重要的硬约定。`pre_trend_features.py` 全文严格遵守。

### 4.5 label — 标签层

```
L-001 LabelBuilder
  - calc_label_short(row)    -> 0/1
  - calc_label_combined(row) -> 0 (二板失败) / 1 (二板成功无主升浪) / 2 (二板+主升浪)
  - build(event_data, daily_data) -> 样本级标签表
L-002 LabelValidator   (label_validator.py)
  - 分布检查 / 缺失检查 / 标签-事件一致性检查
```

标签构建只依赖 `EventSequenceBuilder` 的输出，不再回看日线，避免泄漏。

### 4.6 dataset — 数据集 + 切分

#### `CustomDataset` (`custom_dataset.py`, DS-002)

把"特征宽表 + 标签表"按 `sample_id` join 起来，自动剔除 `code/event_date/first_date/...` 等非特征列，提供 `X`、`y`、`split_by_time`、`split_by_index`。

- `split_by_time(date_col, train_end, valid_end)` — 严格时间切分
- `split_by_index(shuffle=True)` — 历史随机切分接口，**已加未来函数警告**；推荐 `shuffle=False`，或传 `date_col` 触发断言

#### `SectorStockSplitter` (`sector_split.py`, DS-003)

提供 6 种切分方式：

| 方法 | 时间安全 | 备注 |
|---|---|---|
| `split_by_time` | ✓ | 推荐默认 |
| `split_by_stock_then_time` | ✓ | 时间安全 + 兼容按 stock 抽样 |
| `purged_kfold_by_time` | ✓ | Lopez de Prado 的 Purged K-Fold，`embargo_days` 控制 train/test 之间的禁运带宽 |
| `split_by_sector` | ⚠️ | 按 key 随机；传 `date_col` 后若 train.max >= test.min 直接 raise |
| `split_by_stock` | ⚠️ | 同上 |
| `cross_validate_by_*` | ⚠️ | 同上 |

后三种带 ⚠️ 的接口出于历史兼容保留，**生产路径推荐用上面三种**。

### 4.7 model — 训练 / 评估 / 重要性

```
M-001 LimitUpModelTrainer (model_trainer.py)
  - LightGBM 多分类 (binary 时自动切换 objective)
  - 自动剔除非数值列, X.reindex(columns=feature_names_) 在 predict 时对齐
  - save / load 用 pickle, 含 feature_names_ / classes_ / params

M-002 ModelEvaluator (model_evaluator.py)
  - 分类指标 (acc / precision / recall / f1 / AUC OvR-macro)
  - 量化常用指标: calc_ic / calc_rank_ic / calc_ir

M-003 FeatureImportance (feature_importance.py)
  - LightGBM 原生 importance + permutation importance
```

### 4.8 backtest — 回测引擎

```
B-001 Backtester (backtester.py)
  事件驱动:
    for 每个交易日 T:
      _process_sells(T)        # 持仓达 sell_n 天就按 close 卖
      _process_buys(T, ...)    # 取 T - entry_delay 日的 signals, 按 T 日 open 买
      _mark_to_market(T)
  默认 entry_delay=1 -> 信号 T 日产生, T+1 开盘成交
  entry_delay=0 -> 同日成交, 会同时 logger.warning + warnings.warn

B-002 PerformanceAnalyzer (analyzer.py)
  累计收益 / 年化 / 夏普 / 最大回撤 / 胜率 / 平均盈亏比
B-003 ReportGenerator (report_generator.py)
  HTML / markdown 报告
```

### 4.9 scripts/run_pipeline.py — 端到端入口

骨架：

```
main()
 ├─ get_config()                       # 读取 default.yaml
 ├─ 加载数据 (TDXDataLoader)
 ├─ detect_events()                    # E-001..E-004
 ├─ calculate_features()               # F-005
 ├─ train_model()
 │    ├─ SectorStockSplitter.split_by_time
 │    ├─ LookAheadValidator.assert_no_time_leakage     ← 关键门禁
 │    ├─ LookAheadValidator.scan_feature_names         ← 关键门禁
 │    ├─ LimitUpModelTrainer.train()
 │    └─ ModelEvaluator
 └─ run_backtest()
      └─ Backtester(entry_delay=cfg.backtest.entry_delay)  ← 默认 1
```

---

## 5. 反未来函数（look-ahead bias）防线汇总

这是项目最重要的非功能性约束，三道防线同时生效：

### 防线 1 — 模块内自律

- **特征模块**：所有切片仅向左 (`[event_idx - n + 1 : event_idx + 1]`)
- **label 模块**：只依赖事件表，不回看 daily
- **事件检测**：连板天数、市场广度等均按"截至当日"计算

### 防线 2 — 切分层显式断言

- `SectorStockSplitter._split_by_keys(date_col=...)` 内部调用 `_assert_no_time_leakage`
- `CustomDataset.split_by_index(date_col=...)` 同上
- `purged_kfold_by_time` 用 `embargo_days` 留出禁运带宽

### 防线 3 — Pipeline 顶层校验器

- `LookAheadValidator.assert_no_time_leakage` 在训练前再校验一次
- `LookAheadValidator.scan_feature_names` 扫描列名嗅探词
- `Backtester(entry_delay>=1)` 强制 T+1 成交

任一防线触发即终止训练，**保证管线不会因为后续重构悄悄退化**。

---

## 6. 设计上的几个关键选择

### 6.1 为什么模块代号 (E-001 / F-005 / M-001) 显式标在文件头

便于跨模块讨论时不歧义。例如"E-002 的 gap_days 字段语义"比"二板检测器的某列"清晰得多。代号也方便和原始项目方案文档对齐。

### 6.2 为什么特征切片统一用 `iloc[a:b]` 而不是 `rolling`

`rolling` 在 pandas 里默认 *右闭右对齐*（包含当日），看似一样，但：

- 跨股票 groupby 时 `rolling` 在边界有不同行为
- 显式 `iloc[event_idx + 1 - n : event_idx + 1]` 让"包含 event 当日"这件事一目了然，方便代码审查
- 排错时只要打印 `df.iloc[a:b]` 就能看到精确的切片

### 6.3 为什么 entry_delay 默认 1 而不是 0

业务上 first_date（首板涨停日）的"是否涨停"信息要到 **当日收盘后** 才确定，而特征里又用到了 first_date 当日的 close 等数据。如果同日开盘成交，就相当于「用未来生成信号 + 用过去成交」的双重未来函数。`entry_delay=1` 在最小代价下消除这层泄漏。

### 6.4 为什么 split_by_sector / split_by_stock 没有直接禁用

历史代码可能依赖这些接口，直接删除会破坏既有调用方。改为：

- 默认行为 + warning（兼容旧调用）
- 传 `date_col` 后从 warning 升级到 raise（CI 友好）
- 同时提供时间安全的 `split_by_stock_then_time` 和 `purged_kfold_by_time` 作为推荐路径

---

## 7. 扩展指南

### 7.1 加一个新特征模块

参考 `pre_trend_features.py` 的接口：

```python
class MyNewFeatures:
    def calculate_at_event(self, code, event_date, daily_data) -> pd.Series:
        # 返回的 Series 必须含 code, event_date, 以及若干 f_xx_* 列
        ...
```

然后在 `feature_calculator.py` 里注册：

```python
self.my_new = MyNewFeatures()
...
feats.update(self.my_new.calculate_at_event(code, ev_date, sub).to_dict())
```

**自检 checklist**：

- [ ] 所有切片只向左
- [ ] 返回列名不含 `future / next_ / fwd_ / ahead`
- [ ] 写单测时调用 `LookAheadValidator.scan_feature_names`

### 7.2 接入新数据源

实现一个 loader，返回标准 schema：

```
date | code | open | high | low | close | volume | turnover | change_pct
```

放在 `src/data/` 下即可，对上层完全透明。

### 7.3 换模型（LightGBM → XGBoost / CatBoost / NN）

`LimitUpModelTrainer` 是个薄包装，可直接照搬接口（`train / predict / predict_proba / save / load / feature_importance`）写一个新 trainer。`run_pipeline.py` 里挑选 trainer 即可。

### 7.4 加自己的回测约束（涨停板买不进）

修改 `Backtester._process_buys`：

```python
if change_pct_at_T+1_open >= 9.9:   # 简化伪代码
    continue
```

要做这种 "撮合可行性" 过滤时，记得只用 **撮合日当日** 的开盘信息，不要回看未来 close。

---

## 8. 测试

| 测试目录 | 覆盖 |
|---|---|
| `tests/test_utils/` | 配置、日志、交易日历、校验器 |
| `tests/test_event/` | 涨停 / 二板 / 主升浪 / 事件序列 |
| `tests/test_feature/` | 各特征模块 |
| `tests/test_label/` | 标签构建 / 校验 |
| `tests/test_dataset/` | CustomDataset, SectorStockSplitter |
| `tests/test_model/` | LimitUpModelTrainer |
| `tests/test_backtest/` | Backtester |
| `tests/_smoke_no_lookahead.py` | **10 项防未来函数 smoke 测试** |

`_smoke_no_lookahead.py` 不依赖 pytest，可在任何环境下用 `python tests\_smoke_no_lookahead.py` 一键跑。CI 上推荐 **两条都跑**：`pytest tests/` 走单元测试，smoke 走集成验证。

---

## 9. 一句话总结

这是一个**事件驱动**的因子挖掘项目：
**日线 → 涨停事件 → 二板事件 → 主升浪事件（label） → 涨停前形态特征 → LightGBM 多分类 → Top-K 等权 T+1 开盘策略**。
设计上把"反未来函数"作为头等约束，通过模块自律 + 切分断言 + Pipeline 校验器三层防线锁死。任何后续扩展只要不绕过这三层，理论上就不会引入 look-ahead bias。
