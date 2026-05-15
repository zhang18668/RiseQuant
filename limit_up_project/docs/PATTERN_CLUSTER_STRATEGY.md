# 形态聚类多模型策略（Pattern-Cluster Multi-Model）设计文档 — v2

> **实施状态**（2026-05-15）：
>
> 全部 12 个代码任务已落地（13 步顺序，第 13 步是端到端跑通）。沙盒环境验证了 47 个不依赖 LightGBM 的单测，**lightgbm/sklearn 相关的 3 套单测请在你的 Windows .venv 上跑**。
>
> **快速运行手册**（按顺序）：
>
> 1. **Dry-run 数据诊断**（强烈建议先跑一遍）
>    ```
>    set PYTHONPATH=%CD%
>    python scripts\wash_pattern_dryrun.py --start 2022-01-01 --end 2024-12-31
>    ```
> 2. **单次训练**（同时产 A、B 两种 bundle）
>    ```
>    python scripts\wash_pattern_train.py --start 2022-01-01 --end 2024-12-31 --train-end 2023-12-31 --valid-end 2024-06-30 --test-end 2024-12-31 --golden-lo 0.15 --golden-hi 0.35 --bundle-type both --out models\pattern_cluster
>    ```
> 3. **回测**（同一 bundle 可反复跑）
>    ```
>    python scripts\wash_pattern_backtest.py --bundle models\pattern_cluster\latest --bundle-type per_cluster --bt-start 2024-01-01 --bt-end 2024-12-31
>    ```
> 4. **跑所有单测**
>    ```
>    pytest tests\test_event\test_golden_label_filter.py tests\test_event\test_data_dryrun.py tests\test_pattern\ tests\test_model\test_model_bundle.py tests\test_model\test_cluster_model_trainer.py tests\test_model\test_single_with_cf_trainer.py tests\test_utils\test_walkforward.py -v
>    ```
>
> **新增文件清单**：
>
> - `src/event/golden_label_filter.py` + `tests/test_event/test_golden_label_filter.py`
> - `src/event/data_dryrun.py` + `tests/test_event/test_data_dryrun.py`
> - `src/pattern/__init__.py`、`sequence_extractor.py`、`dtw.py`、`kmedoids.py`、`pattern_clusterer.py`、`pattern_router.py`、`visualizer.py` + 对应单测
> - `src/model/model_bundle.py`、`cluster_model_trainer.py`、`single_with_cf_trainer.py` + 单测
> - `src/utils/walkforward.py` + 单测
> - `scripts/wash_pattern_dryrun.py`、`wash_pattern_train.py`、`wash_pattern_backtest.py`
> - `src/model/model_trainer.py` 增加 `categorical_feature` 参数（方案 B 用）
>
> **关键设计兑现**：
>
> - 🔴 形态序列已改成 20 日回看窗口、10 通道（含 padding 标志），训练推理输入形状一致
> - 🟡 双 bundle 并行：一次训练产出 `bundle_per_cluster/` 与 `bundle_single_with_cf/`
> - 🟡 dry-run 强制前置，金标准事件不达标 abort
> - 🟡 walk-forward 模式可选，按窗口产独立 bundle
> - 🟠 `PatternClusterer.fit(..., train_only=True)` assert 防数据泄露
> - 🟢 训练末尾自动出 4 张图（matplotlib 可选，没装就只写 CSV）
> - 🟢 每个 cluster 单独算 calibration / signal_ratio / precision@top10%
>
> ---
>
> v2 更新（基于 v1 自审与讨论）：
> - **🔴 修正"路由序列对齐"硬伤**：形态序列定义改成"`potential_date` 之前的 20 日回看窗口"，训练与推理看到的输入形状一致（第 2 / 4 节）
> - **🟡 引入数据 dry-run 步骤**：训练前先打印金标准事件数量与分布，样本不够直接退化（第 3 / 7 节）
> - **🟡 双 bundle 并行训练**：同一次训练同时产出 `per_cluster`（每形态一模型）与 `single_with_cluster_feat`（单模型 + cluster_id 类别特征）两种 bundle，回测里 A/B 对比（第 5 / 7 节）
> - **🟡 walk-forward 训练模式**：可选 `--walk-forward` 跑年度滚动重训，回测脚本能加载多 bundle 按时间分段路由（第 6 / 7 节）
> - **🟠 防止数据泄露**：聚类器只能在 train 切片上 fit，valid/test 一律走 route()（第 4 节）
> - **🟢 评估指标补强**：加上校准曲线、可交易信号占比、假信号率（第 8 节）
> - **🟢 训练末尾自动出形态可视化图**（第 7 节）
>
> 关联文档：`WASH_SECOND_STRATEGY.md`（现有 wash-second 策略的完整说明）

---

## 0. 一句话画像

在现有 `wash_second` pipeline 的 (first → 震荡 → second) 三元组之上，叠加一层"**second 之后未来 22 个交易日内的最高 close 涨幅 ∈ [15%, 35%]**"的金标准过滤，把这些**真正的赢家**按"潜伏候选日之前 20 天的回看窗口形态"做 DTW + KMedoids 聚类。对每一类形态：

1. **方案 A（per_cluster）**：单独训练一个 LightGBM 二分类
2. **方案 B（single_with_cluster_feat）**：用单一 LightGBM，把 cluster_id 作为类别特征喂进去

一次训练同时产出 **A、B 两种 bundle**，回测里 A/B 对比，让真实数据决定哪种更稳。回测时给定一个潜伏候选样本，**先用其 20 日回看窗口算出最相似 cluster + DTW 距离**，距离超阈值（视为"陌生形态"）直接弃用，否则用对应方案给的模型打分。

最终交付：
- **一次训练 → 一个 run 目录里同时含 A、B 两个 bundle**，互不影响
- **同一个 bundle 可以被回测脚本反复加载**，跑不同时间窗口、不同参数（topk / min_score / cooldown）的回测
- 可选 walk-forward 模式：滚动训练多份 bundle，按"信号日期 → 应使用的 bundle"自动路由

---

## 1. 与现有 wash_second pipeline 的差异

| 维度 | 现有 wash_second | 新 pattern_cluster |
|---|---|---|
| 正样本定义 | (first, second) 配对成功 | (first, second) 配对成功 **且 second 后 22 日最高 close 涨幅 ∈ [15%, 35%]** |
| 形态先验 | 全量样本同模型，靠特征自己学 | **20 日回看窗口** DTW 聚类，每类独立处理 |
| 模型数量 | 1 个 | 一次训练同时产 **方案 A（N 个 LightGBM）** + **方案 B（1 个 LightGBM + cluster_id 类别特征）** |
| 训练-回测耦合 | 训练完直接回测 | **彻底解耦**：训练产 bundle，回测加载 bundle 跑 N 次 |
| 时间稳定性 | 单次切分 | 内置 walk-forward 模式（可选） |
| 信号路由 | 单模型出概率 | 硬路由：先算 cluster_id + 距离阈值过滤，再用对应方案打分 |
| 复用 | 改回测参数要重训 | bundle 不变，回测想跑几次跑几次 |

**完全保留并复用**的现有模块：
- `WashSecondDetector`（事件检测）
- `WashSampleBuilder`（滚动潜伏样本，加 1 个透传字段）
- `WashFeatures` + `MarketFeatures`（59 维特征）
- `LimitUpModelTrainer`（每个 cluster + 整体单模型都用它）
- `Backtester`（动态止损、冷静期、一字过滤都不动）

---

## 2. 业务定义（v2 重定）

| 概念 | 严格定义 |
|---|---|
| **首板** | 当日涨停（change_pct ≥ 9.9%），且前 3 个交易日内无任何涨停 |
| **第二涨停** | 首板后 [3, 30] 个交易日内出现的下一个涨停，中间无其他涨停，且 close > 首板 close |
| **金标准成功事件** | (first, second) 对 + **second 之后 22 个交易日内任意一日的 close 较 second close 涨幅 ∈ [15%, 35%]** |
| **形态序列（推理时序列）** ✅v2 | **`potential_date` 之前的 20 个交易日回看窗口**，即 `[potential_date - 19, potential_date]` 的 K 线序列。固定长度 20，**训练时和推理时输入形状完全一致** |
| **形态序列通道（9 维）** ✅v2 | 1) close/first_close - 1（相对首板的归一化收盘价偏移），2) (high-low)/first_close（振幅），3) (close-open)/first_close（实体），4) volume/前20日均量 - 1（量比偏离），5) (close-ma5)/ma5，6) (close-ma10)/ma10，7) (close-ma20)/ma20，8) 距首板的交易日天数 / 30（位置编码），9) 距首板以来累计涨跌幅 |
| **训练时的形态序列**（特殊处理） | 对金标准事件，每一个正样本日 `t` 都生成一个 20 日回看窗口序列；同一个 (first, second) 事件会产出多条形态序列（每个 potential_date 一条），但**聚类时每个事件只贡献 1 条**——取该事件**距 second 最近的那个正样本日**对应的回看窗口作为代表 |
| **形态原型（centroid）** | 每个 cluster 的 K-Medoids 中心，是一条真实事件的 20 日回看窗口（可视化/解释性好） |
| **路由距离阈值** | DTW 距离 > train 集内该 cluster 距离的 95 分位 → 视为"不像任何已知形态"，回测时直接弃用 |
| **正样本日（per-cluster）** | 在金标准事件的震荡区间内、距 second_date ≤ 5 天的那几天，且该事件被路由到当前 cluster |
| **负样本日（per-cluster）** | ①失败首板 [first+3, first+30] 的所有天，按 router 打 cluster_id；②"形态像但未达金标准"事件的震荡区间内的所有天，按 router 打 cluster_id；超距离阈值的样本统一打 `cluster_id = -1`，**不进任何训练集** |
| **动态止损价** | 等于首板 open，跌破即出 |

### 🔴 v2 关键变更说明

**旧版（v1）**: 用 first→second 完整序列做聚类原型，推理时拿"first→potential_date"半截序列匹配 → 形状不一致，DTW 距离没意义。

**新版（v2）**: 形态序列定义为"**potential_date 之前的 20 日回看窗口**"。无论训练还是推理，看到的都是"以候选日为终点、长度 20 的窗口"。聚类训练时用"金标准事件距 second 最近的正样本日"作为该事件的代表序列；推理时直接拿候选日的 20 日窗口。**首板信息通过"距首板天数 / 30"和"距首板累计涨跌幅"两个位置编码通道保留**。

---

## 3. 端到端流程图

```
                 通达信日线 (date, code, OHLCV, change_pct)
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 0  DataDryRun           (新增, 强制先跑)   │
        │   只跑 detector + golden filter, 打印:          │
        │     - 金标准事件总数 / 按年分布                 │
        │     - 不同 [lo,hi] 区间下的事件数               │
        │     - 估计的样本/特征维度                       │
        │   < 300 个金标准事件时 abort, 提示放宽阈值      │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 1  WashSecondDetector   (复用)             │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 2  GoldenLabelFilter    (新增)             │
        │   filter(events, daily, lo=0.15, hi=0.35,       │
        │           horizon=22)                           │
        │   附加列: future_max_close, future_max_pct,     │
        │           future_max_date, is_golden            │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 3  WashSampleBuilder    (复用)             │
        │   生成全量潜伏样本                              │
        │   行级附加: is_golden_event (其所属事件是否金标准)│
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 4  SequenceExtractor   (新增)              │
        │   对每条样本切 [potential_date-19, potential_date] │
        │   → 9 通道 × 20 帧的固定形状序列                │
        │   缺数据 (不足 20 日) → 标记 valid=False        │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 5  时序切分 (按 potential_date 切          │
        │         train/valid/test, 与 wash_second 对齐)  │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 6  PatternClusterer.fit (新增) ✅v2        │
        │   ⚠️ 严格在 train 切片上 fit                    │
        │   ⚠️ 每个金标准事件只贡献 1 条代表序列          │
        │       (距 second 最近的正样本日的 20 日窗口)    │
        │   DTW 距离矩阵 + KMedoids                       │
        │   k 自动选 (silhouette, 范围 3-8) 或手动固定    │
        │   输出: cluster_labels, medoids (k×20×9),       │
        │         dist_thresholds (k 个 95 分位距离)      │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 7  PatternRouter.route  (新增)             │
        │   对全量样本 (train/valid/test 都过 router)      │
        │   返回: (cluster_id, dtw_dist)                  │
        │   超阈值 → cluster_id = -1                      │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 8  Features            (复用)              │
        │   WashFeatures + MarketFeatures = 59 维         │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────┬────────────────────────────────┐
        │ Step 9-A       │ Step 9-B                       │
        │ ClusterModel-  │ SingleModelWithCluster-        │
        │ Trainer        │ FeatTrainer                    │
        │ (新增) ✅v2     │ (新增) ✅v2                     │
        │ for c in clusters:│ 全量样本一个 LGBM,           │
        │   筛 cluster_id==c│ 把 cluster_id 作为类别特征  │
        │   训一个 LGBM   │ (LightGBM categorical_feature)│
        │ 输出 N 个模型   │ 输出 1 个模型                  │
        └────────────────┴────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step 10  ModelBundle.save_both   (新增)         │
        │   写到 models/pattern_cluster/run_<ts>/         │
        │     ├─ shared/   (聚类器 + 路由器 + 元信息)    │
        │     ├─ bundle_per_cluster/      (方案 A)        │
        │     ├─ bundle_single_with_cf/   (方案 B)        │
        │     └─ visualization/   (形态原型图等) ✅v2     │
        └─────────────────────────────────────────────────┘

────────────────  以上为训练 pipeline (跑一次)  ────────────────
────────────────  以下为回测 pipeline (按 bundle 反复跑) ───────

                bundle 目录 + 日线 + --bundle-type per_cluster|single_with_cf
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step A  Bundle 加载 (含 router + 模型)          │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step B  在回测窗口内重做 Step 1, 3, 4, 7, 8     │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step C  按 bundle_type 打分                     │
        │   per_cluster: 对 cluster_id==c 的样本           │
        │     用 models[c].predict_proba                  │
        │   single_with_cf: 全量过单模型 + cluster_id 特征│
        │   cluster_id == -1 的样本一律弃用               │
        └─────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │ Step D  Backtester (复用)                       │
        │   写入 backtests/<bundle_path>/run_<ts>/        │
        └─────────────────────────────────────────────────┘
```

### Walk-forward 模式（可选）✅v2

加 `--walk-forward 12` 启动后，训练 pipeline 会按 12 个月窗口滚动：

```
窗口 1: train [2022-01, 2022-12], test [2023-01, 2023-06]  → bundle_2023H1/
窗口 2: train [2022-01, 2023-06], test [2023-07, 2023-12]  → bundle_2023H2/
窗口 3: train [2022-01, 2023-12], test [2024-01, 2024-06]  → bundle_2024H1/
…
```

每个窗口独立跑一次 Step 1-10，产出独立 bundle。回测脚本支持 `--bundle-dir models/pattern_cluster/walkforward_2024/`，按信号日期自动选用对应 bundle。

---

## 4. 模块设计

### 4.1 新增模块

| 文件 | 类 / 函数 | 输入 | 输出 | 责任 |
|---|---|---|---|---|
| `src/event/golden_label_filter.py` | `GoldenLabelFilter.filter(events, daily, lo, hi, horizon)` | events + 日线 | events + 4 新列 | 算 second 后 22 日最高 close 涨幅，标 is_golden |
| `src/event/data_dryrun.py` | `DataDryRun.summarize(daily, cfg)` | 日线 + 配置 | 打印报告 dict | 训练前必跑，金标准数 < 300 抛错 |
| `src/pattern/sequence_extractor.py` | `SequenceExtractor.extract(daily_by_code, code, end_date, first_date, lookback=20)` | 日线分组 + 候选日 + 首板日 | ndarray (20, 9) 或 None | 切 20 日回看窗口，归一化 9 通道 |
| `src/pattern/pattern_clusterer.py` | `PatternClusterer.fit(sequences, k_range=(3,8), k_fixed=None, random_states=range(10))` | List[ndarray (20,9)] | `(labels, medoids, dist_thresholds, best_k, silhouette)` | DTW + KMedoids，**输入约束为 train 切片** |
| `src/pattern/pattern_router.py` | `PatternRouter(medoids, dist_thresholds).route(sequence) -> (cluster_id, dist)` | 单条 (20, 9) 序列 | (id, dist)，超阈值 id = -1 | 推理路由 |
| `src/model/model_bundle.py` | `BundleArchive(run_dir).save_per_cluster(...) / save_single_with_cf(...) / save_shared(...)`; `ModelBundle.load(dir)` | 训练产物 / 目录 | bundle 对象 | 双 bundle 持久化 + 加载 |
| `src/model/cluster_model_trainer.py` | `train_per_cluster(samples_with_cluster, features, full_cfg) -> List[artifact]` | 已打 cluster_id 的样本 + 特征 | 每个 cluster 一个 artifact | 内部调 `LimitUpModelTrainer` |
| `src/model/single_with_cf_trainer.py` | `train_single_with_cluster_feat(samples_with_cluster, features, full_cfg) -> artifact` | 同上 | 一个 artifact | 单模型 + cluster_id 类别特征 |
| `src/pattern/visualizer.py` | `plot_prototypes(medoids, save_path)`, `plot_cluster_stats(...)` | medoids / 样本统计 | PNG 文件 | 训练末尾自动出图 |
| `src/utils/walkforward.py` | `WalkForwardScheduler.generate_windows(start, end, step_months)` | 起止日期 | List[Window] | walk-forward 窗口生成器 |
| `scripts/wash_pattern_dryrun.py` | `main()` | 配置 | 打印报告 | 独立 dry-run 入口 |
| `scripts/wash_pattern_train.py` | `main()` | 配置 + 数据 | bundle 目录 | **训练 pipeline 入口（一次性 or walk-forward）** |
| `scripts/wash_pattern_backtest.py` | `main()` | bundle + 回测参数 | backtest 子目录 | **回测 pipeline 入口（可反复跑）** |

### 4.2 修改的模块（小改动）

| 文件 | 改动 |
|---|---|
| `src/event/wash_sample_builder.py` | `build()` 返回的 DataFrame 增加 `is_golden_event` 列；新增可选参数 `event_to_cluster: Dict[(code, first_date), int]` |
| `src/model/model_trainer.py` | 增加 `categorical_feature` 参数透传到 LightGBM（用于方案 B） |
| `src/utils/run_archive.py` | 不继承，新建 `BundleArchive`（用组合，持有 RunArchive 实例），新建 `BacktestArchive` |

### 4.3 数据泄露防线 ✅v2

代码层强制约束：

- `PatternClusterer.fit()` 第一行 `assert` 输入序列对应的 `potential_date` 全部 `<= train_end`
- `WashSamples` 不携带 second_date 之后的任何字段进入特征
- 单测覆盖：故意混入 valid 样本到 fit → 必须抛错

### 4.4 完全复用（不改）

- `WashSecondDetector`、`WashFeatures`、`MarketFeatures`、`ModelEvaluator`、`Backtester`、`SectorStockSplitter`

---

## 5. 模型 Bundle 产物结构 ✅v2

### 单次训练（非 walk-forward）

```
models/pattern_cluster/run_20260515_1830/
├─ shared/                                        ← 两种方案共享
│  ├─ shared_meta.json                            ← 配置、特征列、训练日期范围
│  ├─ prototypes.pkl                              ← {cluster_id: ndarray (20, 9)}
│  ├─ router.pkl                                  ← PatternRouter (含距离阈值)
│  ├─ cluster_stats.csv                           ← 每个 cluster 的样本数/年份分布
│  ├─ samples_with_cluster.parquet                ← 训练用样本+cluster_id (可追溯)
│  └─ features.parquet                            ← 训练用特征 (可追溯)
├─ bundle_per_cluster/                            ← 方案 A
│  ├─ bundle_meta.json                            ← 含 default_min_score_per_cluster
│  ├─ models/
│  │  ├─ cluster_0/
│  │  │  ├─ model.pkl
│  │  │  ├─ train_metrics.json                    ← auc, precision@top10, calibration
│  │  │  ├─ feature_importance.csv
│  │  │  └─ split_info.json
│  │  ├─ cluster_1/  …
│  │  └─ training_summary.json                    ← 每 cluster 一行
│  └─ usability_flags.json                        ← {"cluster_0": true, "cluster_3": false}
├─ bundle_single_with_cf/                         ← 方案 B
│  ├─ bundle_meta.json
│  ├─ model.pkl
│  ├─ train_metrics.json                          ← 总体 + per-cluster 拆分
│  ├─ feature_importance.csv
│  └─ split_info.json
└─ visualization/                                  ← v2 新增
   ├─ cluster_prototypes.png                       ← 各 cluster 原型 K 线叠加图
   ├─ cluster_feature_importance.png               ← 各 cluster top10 特征横向对比 (仅方案 A)
   ├─ cluster_year_distribution.png                ← 各 cluster 样本年度分布
   ├─ calibration_per_cluster.png                  ← 校准曲线 (仅方案 A)
   └─ calibration_single_with_cf.png               ← 校准曲线 (仅方案 B)
```

### Walk-forward 训练

```
models/pattern_cluster/walkforward_2026Q1/
├─ schedule.json                                  ← 各窗口起止 + bundle 名映射
├─ window_2023H1/
│  └─ (与上面单次训练目录结构相同)
├─ window_2023H2/
├─ window_2024H1/
└─ window_2024H2/
```

### 回测产物

```
backtests/<bundle_full_path>/<bundle_type>/run_<ts>/
├─ run_config.json                                ← 这次回测用的所有参数
├─ signals.csv                                    ← 含 code, date, score, cluster_id, dtw_dist
├─ trades.csv
├─ equity_curve.csv
├─ backtest_metrics.json                          ← 总体 + per-cluster 拆分
└─ summary.json
```

---

## 6. 关键参数与默认值

| 类别 | 参数 | 默认 | 说明 |
|---|---|---|---|
| dry-run | `min_golden_events_required` | 300 | 不足直接 abort |
| 金标准 | `golden_pct_lo` | 0.15 | second 后 22 日最高 close 涨幅下限 |
| 金标准 | `golden_pct_hi` | 0.35 | 涨幅上限 |
| 金标准 | `golden_horizon_days` | 22 | 未来一个月 |
| 序列 | `seq_lookback` | 20 | 回看窗口长度 ✅v2 |
| 序列 | `seq_channels` | 9 | 见第 2 节 ✅v2 |
| 聚类 | `cluster_k_range` | (3, 8) | silhouette 自动选 k |
| 聚类 | `cluster_k_fixed` | None | 不为 None 则跳过自动选 k |
| 聚类 | `dtw_window` | 5 | Sakoe-Chiba 带宽 |
| 聚类 | `kmedoids_init_n` | 10 | 多次初始化取最优 |
| 聚类 | `route_dist_quantile` | 0.95 | 距离阈值取 train 集 95 分位 |
| 训练 | `min_samples_per_cluster` | 80 | 不足时方案 A 该 cluster 标 unusable |
| 训练 | `model_params` | 同 wash_second | 沿用 LightGBM 默认 |
| 训练 | `train_end / valid_end / test_end` | 同 wash_second | 时序切分 |
| 训练 | `categorical_feature_in_method_b` | `["cluster_id"]` | 方案 B 类别特征列 |
| walk-fwd | `walk_forward_months` | None | 设为 6/12 启用 walk-forward |
| walk-fwd | `walk_forward_min_train_months` | 12 | 首窗口至少 12 个月训练数据 |
| 回测 | `bundle_type` | `per_cluster` | per_cluster / single_with_cf / both |
| 回测 | `topk` | 5 | 同 wash_second |
| 回测 | `min_score` | 0.6（全局） | per-cluster 字典可覆盖（仅 A） |
| 回测 | `min_score_per_cluster` | None | JSON/YAML 路径，可选 |
| 回测 | `cooldown_after_stop` | 5 | 同 wash_second |
| 回测 | `skip_zhangting_open` | True | 同 wash_second |
| 回测 | `dedupe_same_day_same_code` | True | 同日同股取 score 最高那条 ✅v2 |

**配置文件**：`config/config.yaml` 加 `pattern_cluster:` 命名空间。

---

## 7. CLI 设计

### Dry-run（强烈建议先跑）

```bash
set PYTHONPATH=%CD%
python scripts\wash_pattern_dryrun.py ^
    --start 2022-01-01 --end 2024-12-31 ^
    --golden-lo 0.10 --golden-hi 0.40 ^
    --golden-lo 0.15 --golden-hi 0.35 ^
    --golden-lo 0.20 --golden-hi 0.30
```

输出（终端）：

```
=== 数据 Dry-run 报告 ===
日线区间: 2022-01-01 ~ 2024-12-31, 主板 4123 只
首板事件总数: 18450
有 second 配对: 4220
金标准事件（[lo,hi] 不同口径下）:
  [0.10, 0.40]  →  812 个    ✅ 充足
  [0.15, 0.35]  →  487 个    ✅ 充足
  [0.20, 0.30]  →  213 个    ⚠️ 不足，建议放宽
按年分布（[0.15, 0.35]）:
  2022: 156 / 2023: 178 / 2024: 153
预计聚成 5 类，最少 cluster 约 78 样本（按金标准 487 / 5 * 0.8 估）
```

### 训练（一次性）

```bash
python scripts\wash_pattern_train.py ^
    --start 2022-01-01 --end 2024-12-31 ^
    --train-end 2023-12-31 --valid-end 2024-06-30 --test-end 2024-12-31 ^
    --golden-lo 0.15 --golden-hi 0.35 ^
    --k-range 3 8 ^
    --bundle-type both ^
    --out models\pattern_cluster
```

末尾打印：

```
✅ shared/ written (router.pkl, prototypes.pkl, ...)
✅ bundle_per_cluster/ written (5 models)
✅ bundle_single_with_cf/ written (1 model)
✅ visualization/ written
bundle path → models\pattern_cluster\run_20260515_1830
latest → models\pattern_cluster\latest
```

### 训练（walk-forward）

```bash
python scripts\wash_pattern_train.py ^
    --start 2022-01-01 --end 2025-04-30 ^
    --walk-forward-months 6 ^
    --min-train-months 12 ^
    --golden-lo 0.15 --golden-hi 0.35 ^
    --out models\pattern_cluster\walkforward_2026Q1
```

### 回测（多次）

```bash
:: 方案 A 全年
python scripts\wash_pattern_backtest.py ^
    --bundle models\pattern_cluster\latest ^
    --bundle-type per_cluster ^
    --bt-start 2024-01-01 --bt-end 2024-12-31

:: 方案 B 同窗口
python scripts\wash_pattern_backtest.py ^
    --bundle models\pattern_cluster\latest ^
    --bundle-type single_with_cf ^
    --bt-start 2024-01-01 --bt-end 2024-12-31

:: 方案 A 限定某些 cluster + 提高阈值
python scripts\wash_pattern_backtest.py ^
    --bundle models\pattern_cluster\latest ^
    --bundle-type per_cluster ^
    --clusters 0,2,4 --min-score 0.7

:: walk-forward bundle 回测
python scripts\wash_pattern_backtest.py ^
    --bundle-dir models\pattern_cluster\walkforward_2026Q1 ^
    --bundle-type per_cluster ^
    --bt-start 2024-01-01 --bt-end 2025-04-30
```

每次回测都新建 `backtests/<bundle_path>/<bundle_type>/run_<ts>/`，互不污染。

---

## 8. 验证清单 ✅v2

### 训练后 — 单 bundle 内每个 cluster（方案 A）

| 指标 | 期望 | 含义 |
|---|---|---|
| `n_samples_train / valid / test` | 各 ≥ 80 | 样本数底线，否则模型不可信 |
| `pos_ratio_train` | 5% - 30% | 正样本占比 |
| `auc_test` | ≥ 0.65 | 排序能力 |
| `precision@top10%_test` | ≥ 0.40 | 高分样本中真正样本比例 |
| `calibration_brier_score` ✅v2 | ≤ 0.20 | 概率校准（越低越好） |
| `usable_signal_ratio` ✅v2 | `(score > min_score).mean() ≥ 0.5%` | 该 cluster 在实盘出手频次不能太低 |
| `top_features` | 前 5 集中在量能/均线/MACD/位置 | 不该全是 `f_w_today_*` 单日噪声 |

**任一不达标 → `usable=false`，回测自动跳过该 cluster。**

### 训练后 — 方案 A vs 方案 B 横向对比

在 `training_summary.json` 顶层：

| 指标 | 方案 A 总和 | 方案 B |
|---|---|---|
| `test_auc_overall` | 各 cluster AUC 按样本数加权 | 总体 AUC |
| `test_precision@top10%` | 加权 | 总体 |
| `n_usable_clusters` | 整数 | N/A |
| `n_test_samples_covered` | 仅 usable cluster 覆盖到的 | 全量 |

### 回测后 ✅v2

`backtest_metrics.json` 增加三组诊断：

- **总体**：胜率、年化、夏普、最大回撤、`sell_reasons` 分布
- **per-cluster 拆分**（仅方案 A）：每个 cluster 各自的胜率、交易数、平均盈亏比
- **假信号率**：信号触发但 ≤ 3 日内触发动态止损的比例（< 30% 算健康）

---

## 9. 与现有 wash_second_pipeline 的兼容性

- 新 pipeline 完全独立的脚本，**不修改也不调用** `wash_second_pipeline.py`
- 共享底层模块（detector / sample_builder / features / trainer / backtester）
- 新增模块都放在 `src/pattern/`、`src/event/{golden_label_filter,data_dryrun}.py`、`src/model/{model_bundle, cluster_model_trainer, single_with_cf_trainer}.py`、`src/utils/walkforward.py`
- 现有 `models/wash_second/` 目录不受影响
- 配置加新命名空间 `pattern_cluster:`，与 `wash_second:` 并列

---

## 10. 剩余开放问题（v1 时遗留 + v2 新增）

下面这些请逐项确认，**我有"v2 倾向"的我用 ★ 标了**：

| # | 问题 | v2 倾向 | 备选 |
|---|---|---|---|
| Q1 | 聚类时是否要按"年份"或"市场状态"分层？ | 不分层★，让大盘特征 + walk-forward 模式自己处理 | 按训练区间整体聚 |
| Q2 | cluster 样本不足 80 时是否合并到最近 cluster？ | 不合并，直接 `usable=false`★ | 合并到最近邻 |
| Q3 | 同一天 (date, code) 被多个潜伏样本覆盖怎么去重？ | 取 score 最高的一条★（已写入参数 `dedupe_same_day_same_code=True`） | 完全不去重 |
| Q4 | DTW 计算量大时采样？ | 训练样本（事件代表序列）通常 < 1000，全量算★；超过 3000 自动采样到 3000 | 全量 |
| Q5 | 失败首板的负样本是否也走 router 打标？ | 是★，超阈值丢弃 | 平均分配 |
| Q6 | bundle 里是否存 samples_with_cluster + features？ | 存★，磁盘换可追溯 | 不存 |
| Q7 | 回测 `--per-cluster-min-score` 用 JSON 文件？ | 支持★，命令行 `--min-score` 作兜底 | 单一全局 |
| Q8 | 同一 bundle 在不同股票池回测？ | 允许★，加 `--code-list` 参数 | 强制相同 |
| Q9 ✅v2 | 9 通道里"距首板天数 / 30"分母是固定 30 还是用实际 gap？ | 固定 30★（max_gap 上限），保证不同事件之间可比 | 用实际 gap，归一化到 [0,1] |
| Q10 ✅v2 | 方案 B 的 cluster_id 类别特征，是用 LightGBM 内置 categorical 处理还是手动 one-hot？ | LightGBM 内置 categorical_feature★（更省内存、Split 直接基于类别） | one-hot 5-8 维 |
| Q11 ✅v2 | walk-forward 中是否每个窗口都重新聚类？还是聚类用首窗口的固定原型？ | 每窗口重新聚类★（捕捉市场风格漂移），但 schedule.json 里能看到不同窗口的 cluster 数变化 | 固定原型 |
| Q12 ✅v2 | 不足 20 日历史的样本（首板第 4-19 天的潜伏候选）怎么处理？ | 用 padding（前面补 0，再加一个 `is_padded` 通道）★，避免直接丢★ | 直接丢弃 |

> Q12 备注：这条会影响第 4 节 SequenceExtractor 的输出维度。如果同意 padding，9 通道会变成 10 通道（多一个 `is_padded`）。**这条尤其要请你拍板**。

---

## 11. 实施顺序（确认后按此写代码）

1. `src/event/golden_label_filter.py` + 单测
2. `src/event/data_dryrun.py` + `scripts/wash_pattern_dryrun.py` + 单测
3. `src/pattern/sequence_extractor.py` + 单测（含 padding 单测）
4. `src/pattern/pattern_clusterer.py` + 单测（含数据泄露防护单测）
5. `src/pattern/pattern_router.py` + 单测
6. `src/model/model_bundle.py` + `src/utils/run_archive.py` 扩展 + 单测
7. `src/model/cluster_model_trainer.py` + 单测（方案 A）
8. `src/model/single_with_cf_trainer.py` + 单测（方案 B）
9. `src/pattern/visualizer.py`（无单测，靠人工看图）
10. `src/utils/walkforward.py` + 单测
11. `scripts/wash_pattern_train.py`（先单次模式跑通，再加 walk-forward）
12. `scripts/wash_pattern_backtest.py`
13. 端到端集成跑通 + 本文件补充"实测产物"小节

每个模块按"先单测后接入"节奏。第 7 步完成后即可端到端跑通方案 A，第 8 步完成后双方案 ready，第 10 步后 walk-forward ready。

---

## 12. 你需要核对的核心四件事

1. **业务定义**（第 2 节）—— 特别是 v2 重定的"形态序列 = 20 日回看窗口 + 9 通道"和"金标准事件代表序列 = 距 second 最近的正样本日"是否同意
2. **双 bundle 并行策略**（第 5 / 7 节）—— 是否同意一次训练同时产出方案 A 与方案 B
3. **walk-forward 作为可选模式**（第 3 / 6 / 7 节）—— 是否同意默认关闭、加 `--walk-forward-months` 开启
4. **Q9-Q12 四个 v2 新增问题**（第 10 节）—— 尤其 Q12 padding 决定影响通道维度

确认后我按第 11 节的 13 步顺序写代码。
