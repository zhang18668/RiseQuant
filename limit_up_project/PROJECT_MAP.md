# PROJECT_MAP

> Harness 视角的项目地图。这里记录“代码在哪里、入口是什么、核心模块负责什么”。深度设计说明见 `ARCHITECTURE.md`，运行手册见 `RUNBOOK.md`。

## 顶层结构

```text
limit_up_project/
├── api/                 # FastAPI 后端：异步任务、接口路由、脚本调度
├── config/              # YAML 配置，目前主配置为 default.yaml
├── data/                # 本地数据缓存与校验产物
├── docs/                # 策略、数据源、专题说明
├── models/              # 模型、bundle、任务状态、临时 fetch 缓存
├── scripts/             # 命令行入口：数据构建、训练、回测、dashboard
├── src/                 # 业务核心代码
├── tests/               # 单元测试与防未来函数 smoke
├── web/                 # 旧版静态 dashboard
├── PROJECT_MAP.md
├── ARCHITECTURE.md
├── TASKS.md
├── DECISIONS.md
├── RUNBOOK.md
└── requirements.txt
```

仓库根目录另有 `webui/`，是当前 Vue + Vite 前端；`start_dev.bat` 会同时启动 `limit_up_project/api` 与 `webui`。

## 主要入口

| 入口 | 类型 | 用途 |
|---|---|---|
| `scripts/run_pipeline.py` | CLI | 原始端到端流水线：TDX 日线 -> 事件 -> 特征 -> LightGBM -> 回测 |
| `scripts/wash_second_pipeline.py` | CLI | wash-second 策略流水线，含训练、参数搜索、回测归档 |
| `scripts/wash_ambush_pipeline.py` | CLI | wash-ambush 策略流水线 |
| `scripts/wash_pattern_train.py` | CLI | pattern-cluster 训练入口，生成 bundle |
| `scripts/wash_pattern_backtest.py` | CLI | pattern-cluster bundle 回测入口 |
| `scripts/wash_pattern_dryrun.py` | CLI | pattern 标签/样本 dry-run 诊断 |
| `scripts/build_zt_pool_cache.py` | CLI | 构建 AkShare 涨停池缓存 |
| `scripts/build_daily_cache.py` | CLI | 构建按股票代码分片的日线缓存 |
| `scripts/verify_daily_cache.py` | CLI | 校验日线缓存与涨停池一致性 |
| `api/main.py` | API | FastAPI 应用入口，`uvicorn api.main:app --reload --port 8000` |
| `api/runner.py` | API adapter | 把长任务包装成可 JSON 化的函数，供后台任务调用 |
| `api/jobs.py` | API harness | 轻量任务队列与 JSON 状态持久化 |

## 核心模块

### `src/data*` 数据层

| 路径 | 职责 |
|---|---|
| `src/data/tdx_loader.py` | 读取通达信本地 `.day` 日线文件，补齐标准 OHLCV schema |
| `src/data/akshare_loader.py` | AkShare 在线数据封装 |
| `src/data/zt_pool_loader.py` | 涨停池抓取、分档、主板过滤、parquet 缓存 |
| `src/data/daily_cache.py` | `DailyCacheManager`，按 `code.parquet` 管理日线缓存 |
| `src/data/cache_manager.py` | 通用缓存工具 |
| `src/data_fetch/*` | API runner 使用的数据源抽象与 TDX/AkShare fetcher |
| `src/data_clean/*` | 数据清洗与质量校验 |

### `src/event` 事件层

| 路径 | 职责 |
|---|---|
| `limit_up_detector.py` | 涨停、首板、连板检测 |
| `second_board_detector.py` | 首板后窗口内二板检测 |
| `main_wave_detector.py` | 二板后主升浪标签窗口检测 |
| `event_sequence_builder.py` | 首板、二板、主升浪事件合并为训练样本 |
| `wash_sample_builder.py` | pattern/wash 样本构建 |
| `wash_second_detector.py` | wash-second 事件识别 |
| `wash_ambush_sample_builder.py` | ambush 样本构建 |
| `golden_label_filter.py` | golden 区间标签过滤 |
| `data_dryrun.py` | 样本/标签 dry-run 诊断报告 |

### `src/feature` 特征层

| 路径 | 职责 |
|---|---|
| `feature_calculator.py` | 批量特征编排 |
| `pre_trend_features.py` | 首板前 1m/2m 形态、量价、支撑压力、综合评分 |
| `price_volume.py` | 收益、量比、均线、波动等基础量价因子 |
| `limit_up_features.py` | 历史涨停与连板统计 |
| `market_features.py` | 市场代理宽度与环境特征 |
| `sector_sentiment.py` | 板块/市场情绪特征 |
| `money_flow.py` | 资金流代理特征 |
| `technical_indicators.py` | 技术指标 |
| `wash_features.py` | wash 策略专用特征 |

### `src/label`、`src/dataset`

| 路径 | 职责 |
|---|---|
| `label/label_builder.py` | 构建 `label_short` 与 `label_combined` |
| `label/label_validator.py` | 标签分布与一致性校验 |
| `dataset/custom_dataset.py` | 特征/标签合并、非特征列剔除、数据切分 |
| `dataset/sector_split.py` | 时间安全切分、按股票/板块切分、Purged K-Fold |

### `src/model` 模型层

| 路径 | 职责 |
|---|---|
| `model_trainer.py` | LightGBM 分类训练、预测、保存/加载 |
| `single_with_cf_trainer.py` | 单模型 + conformal filtering 训练产物 |
| `cluster_model_trainer.py` | 按形态聚类训练子模型 |
| `model_bundle.py` | bundle 归档与加载 |
| `model_registry.py` | 版本化模型注册表 |
| `model_evaluator.py` | 分类指标、IC/RankIC/IR |
| `feature_importance.py` | 原生与 permutation importance |

### `src/pattern` 形态聚类层

| 路径 | 职责 |
|---|---|
| `sequence_extractor.py` | 从样本附近抽取价格/成交量序列 |
| `dtw.py` | DTW 距离 |
| `kmedoids.py` | k-medoids 聚类 |
| `pattern_clusterer.py` | 聚类训练与结果封装 |
| `pattern_router.py` | 新样本路由到最近形态簇 |
| `visualizer.py` | 原型、年份分布、特征重要性、校准图 |

### `src/backtest` 回测层

| 路径 | 职责 |
|---|---|
| `backtester.py` | 事件驱动回测，默认 `entry_delay=1`，信号 T 日产生、T+1 开盘成交 |
| `ambush_backtester.py` | ambush 策略扩展回测器 |
| `analyzer.py` | 收益、回撤、夏普、胜率等指标 |
| `report_generator.py` | HTML/Markdown 报告 |

### `src/utils`

| 路径 | 职责 |
|---|---|
| `config.py` | YAML 配置加载与点路径读取 |
| `logger.py` | loguru 统一日志 |
| `calendar.py` | 交易日历、前后交易日 |
| `validator.py` | 数据校验与 `LookAheadValidator` 防未来函数 |
| `walkforward.py` | walk-forward 窗口调度 |
| `run_archive.py` | 运行产物归档 |

## API 与前端

```text
api/
├── main.py          # FastAPI app，挂载 routers
├── jobs.py          # pending/running/done/failed 状态机，写入 models/_jobs/*.json
├── runner.py        # fetch/clean/dryrun/train/backtest 适配层
├── schemas.py       # Pydantic 请求/响应模型
└── routers/
    ├── data.py      # /api/data/fetch, /clean, /dryrun
    ├── train_bt.py  # /api/train, /api/backtest
    ├── bundles.py   # bundle、可视化、回测明细查询
    └── optimize.py  # grid/random 搜索与敏感性分析
```

`webui/` 使用 Vue 3、Element Plus、ECharts、axios、vue-router。前端通过 Vite 代理访问 `/api/*`，核心 API 封装在仓库根的 `webui/src/api.js`。

## 测试矩阵

| 目录 | 覆盖范围 |
|---|---|
| `tests/_smoke_no_lookahead.py` | 10 项防未来函数 smoke，可独立运行 |
| `tests/test_utils/` | config、logger、calendar、validator、walkforward |
| `tests/test_data/` | daily cache |
| `tests/test_data_clean/` | cleaners、validator |
| `tests/test_event/` | 涨停、二板、主升浪、golden、dryrun |
| `tests/test_feature/` | 量价、涨停、pre-trend、sector/money flow、编排 |
| `tests/test_label/` | label builder/validator |
| `tests/test_dataset/` | custom dataset、sector split |
| `tests/test_model/` | trainer、bundle、cluster、single_with_cf |
| `tests/test_pattern/` | sequence、router、clusterer |
| `tests/test_backtest/` | backtester |
