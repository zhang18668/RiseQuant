# ARCHITECTURE

> Harness 架构说明：用一个稳定的外壳把“数据、训练、回测、优化、可视化”串起来。核心原则是脚本可独立运行，API 只做任务编排，前端只观察任务和产物。

## 目标问题

项目面向 A 股主板，围绕“首板涨停后是否能打到二板，并在二板后形成主升浪”构建事件驱动因子模型。

核心标签：

| 标签 | 含义 |
|---|---|
| `label_short=0/1` | 首板后窗口内是否出现二板 |
| `label_combined=0` | 二板失败 |
| `label_combined=1` | 二板成功，但主升浪失败 |
| `label_combined=2` | 二板成功且主升浪成功 |

当前实现里并行存在三条策略路径：原始 limit-up pipeline、wash-second pipeline、pattern-cluster pipeline。Harness 的职责是让这些路径共享数据缓存、任务队列、产物归档和 dashboard 查询。

## 分层

```text
                        config/default.yaml
                                 |
              +------------------+------------------+
              |                                     |
          数据与缓存层                           Harness 层
    TDX / AkShare / parquet                FastAPI / jobs / runner
              |                                     |
              v                                     v
        事件与样本层                         Vue dashboard / API
 limit-up / wash / pattern labels                 |
              |                                     |
              v                                     |
          特征工程层                               |
  pre-trend / wash / market / technical            |
              |                                     |
              v                                     |
          训练与模型层 <----------------------------+
 LightGBM / cluster models / bundle archive
              |
              v
          回测与优化层
 Backtester / grid-search / sensitivity / reports
```

## 数据流

### 离线数据缓存

```text
AkShare stock_zt_pool_em
      -> scripts/build_zt_pool_cache.py
      -> data/zt_pool_cache/{zt_pool,zt_pool_previous}/YYYYMMDD.parquet

TDX vipdoc .day + AkShare fallback
      -> scripts/build_daily_cache.py
      -> data/daily_cache/<code>.parquet

zt_pool_cache + daily_cache
      -> scripts/verify_daily_cache.py
      -> data/verify/daily_consistency_<ts>.json/md
```

### 原始端到端流水线

```text
TDXDataLoader.load_batch()
  -> LimitUpEventDetector
  -> SecondBoardDetector
  -> MainWaveDetector
  -> EventSequenceBuilder
  -> FeatureCalculator / PreTrendFeatures
  -> SectorStockSplitter + LookAheadValidator
  -> LimitUpModelTrainer
  -> Backtester(entry_delay=1)
  -> models/<version> + backtest artifacts
```

### pattern-cluster Harness 流水线

```text
Vue form
  -> POST /api/train
  -> api.jobs.create_job(kind=train)
  -> api.runner.run_train_via_script()
  -> scripts/wash_pattern_train.py subprocess
  -> models/pattern_cluster/latest or run_*
  -> GET /api/jobs/{id}
  -> GET /api/bundles/detail?bundle_dir=...

Vue form
  -> POST /api/backtest
  -> api.runner.run_backtest_via_script()
  -> scripts/wash_pattern_backtest.py subprocess
  -> backtests/run_*
  -> GET /api/bundles/backtest-detail?run_dir=...
```

## Harness 组件职责

| 组件 | 职责 |
|---|---|
| `api/main.py` | 创建 FastAPI app，配置 CORS，挂载 data/train/backtest/bundles/optimize 路由 |
| `api/jobs.py` | 轻量后台任务状态机，状态落盘到 `models/_jobs/*.json` |
| `api/runner.py` | 把 fetch/clean/dryrun/train/backtest 转成纯函数；训练与回测通过 subprocess 调脚本 |
| `api/schemas.py` | 前后端契约，集中定义请求参数 |
| `api/routers/data.py` | 数据拉取、清洗、dry-run 入口 |
| `api/routers/train_bt.py` | 训练与回测入口 |
| `api/routers/bundles.py` | 查询模型 bundle、可视化、回测明细 |
| `api/routers/optimize.py` | 网格/随机搜索与单参数敏感性分析 |
| `webui/src/api.js` | 前端 API 客户端 |

长任务设计取舍：

- API 线程不直接训练模型，训练/回测使用 `subprocess.run()` 调已有脚本，减少内存污染和模块状态泄漏。
- 任务创建后立刻返回 `{id, kind, state}`，前端轮询 `GET /api/jobs/{id}`。
- 任务状态、日志尾部、结果和错误都写入 `models/_jobs/<id>.json`，服务重启后仍能查询历史任务。

## 数据契约

标准日线字段：

```text
date | code | open | high | low | close | volume | turnover/turnover_rate | change_pct
```

样本关键字段：

```text
sample_id | code | first_date | second_date | gap_days
main_wave_date | period_return | is_main_wave
label_short | label_combined
```

信号字段：

```text
date | code | score
```

bundle 与回测产物通常包含：

```text
config.json
summary.json / train_metrics.json / backtest_metrics.json
model.pkl
feature_importance.csv
signals.csv
trades.csv
equity_curve.csv
visualizations/*.png
```

## 防未来函数边界

项目把防未来函数当成架构约束，而不是事后检查。

1. 特征只允许使用事件当日及之前的数据。典型切片是 `event_idx + 1 - n : event_idx + 1`。
2. 标签允许看未来窗口，但标签构造与特征构造分离，标签列不得回流到特征列。
3. 数据切分优先使用时间切分或 Purged K-Fold。`SectorStockSplitter` 和 `CustomDataset` 提供时间泄漏断言。
4. `LookAheadValidator.scan_feature_names()` 扫描 `future / next_ / fwd_ / ahead / tomorrow` 等可疑列名。
5. 回测默认 `entry_delay=1`，信号 T 日形成，T+1 开盘成交。

## 关键依赖

| 类别 | 依赖 | 用途 |
|---|---|---|
| 数据 | pandas、numpy、pyarrow | 表处理与 parquet 缓存 |
| 数据源 | akshare、通达信 vipdoc | 涨停池、日线、兜底数据 |
| 模型 | lightgbm、scikit-learn | 分类模型、评估、辅助算法 |
| 回测 | 自研 Backtester，保留 backtrader 依赖 | 事件驱动交易模拟 |
| API | fastapi、uvicorn、pydantic | Dashboard 后端 |
| 前端 | Vue 3、Element Plus、ECharts、Vite | 任务提交与产物查看 |
| 测试 | pytest、pytest-cov | 单元测试与覆盖率 |
| 工具 | pyyaml、loguru、tqdm | 配置、日志、进度 |

## 产物布局

```text
data/
├── zt_pool_cache/
│   ├── zt_pool/
│   └── zt_pool_previous/
├── daily_cache/
├── verify/
└── cache/

models/
├── _jobs/
├── _cache_fetch/
├── pattern_cluster/
├── wash_second/
└── wash_ambush/

backtests/
└── run_*/
```

## 运行时边界

- CLI 脚本是事实执行入口；API 和前端不复制训练逻辑。
- `config/default.yaml` 是默认参数中心，但策略脚本允许命令行覆盖。
- TDX 本地路径默认是 `C:/new_tdx/vipdoc`，机器不一致时必须显式传参或改配置。
- 当前 API CORS 为 `allow_origins=["*"]`，仅适合开发环境。
