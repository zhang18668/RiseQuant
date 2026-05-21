# DECISIONS

> 重要技术决策记录。新的架构级变化请继续追加，保留“为什么这么做”，方便以后不反复讨论同一个问题。

## D-001 使用脚本作为生产执行入口，API 只做 Harness

- 状态：已采用
- 背景：训练、回测、聚类等任务耗时长、内存占用高，并且已有 CLI 脚本可独立运行。
- 决策：FastAPI 不在进程内重写训练逻辑；`api/runner.py` 对重任务使用 subprocess 调 `scripts/*.py`。
- 影响：脚本仍是单一事实来源；API 进程更稳定；缺点是 stdout/stderr 只保留尾部，需要脚本自己归档完整产物。

## D-002 轻量 JSON 任务队列优先于引入 Celery/RQ

- 状态：已采用
- 背景：当前主要是单机 Windows 开发环境，任务规模尚不需要分布式队列。
- 决策：`api/jobs.py` 用 FastAPI `BackgroundTasks` + 内存注册表 + `models/_jobs/*.json` 持久化任务状态。
- 影响：部署简单、易排查；不适合多进程多实例并发调度，后续生产化可替换为数据库队列。

## D-003 数据缓存采用 parquet 文件，而不是先上数据库

- 状态：已采用
- 背景：日线、涨停池、训练样本都是批处理表数据，读写模式以离线顺序扫描为主。
- 决策：涨停池按日期分片，日线按股票代码分片，统一落 parquet。
- 影响：读写简单、可直接 pandas 加载；缺点是并发写和复杂查询能力有限。

## D-004 TDX 本地数据优先，AkShare 作为补充

- 状态：已采用
- 背景：本地 TDX 日线速度快、可离线；AkShare 覆盖方便但依赖网络与接口稳定性。
- 决策：日线缓存构建默认 `auto`：优先 TDX，缺失时 AkShare fallback；涨停池继续使用 AkShare 东方财富接口。
- 影响：本机数据路径成为运行前置条件；需要在 `RUNBOOK.md` 中明确排查步骤。

## D-005 防未来函数是硬约束

- 状态：已采用
- 背景：量化策略极容易因为特征切片、切分方式、回测成交假设引入未来函数。
- 决策：在特征、数据切分、pipeline 顶层和回测成交上多层约束：左向切片、时间安全切分、`LookAheadValidator`、`entry_delay=1`。
- 影响：短期可能降低漂亮回测结果，但提高可信度。任何新特征必须说明可用时点。

## D-006 回测默认 T+1 开盘成交

- 状态：已采用
- 背景：首板信号和特征通常依赖 T 日收盘后才能确认。
- 决策：`Backtester(entry_delay=1)` 为默认；`entry_delay=0` 仅用于特殊实验，并会告警。
- 影响：避免“收盘确认信号却同日开盘买入”的泄漏。

## D-007 保留多策略路径，不强行提前统一

- 状态：已采用
- 背景：仓库中已有原始 limit-up、wash-second、wash-ambush、pattern-cluster 多条实验路径。
- 决策：先通过 Harness 统一入口、任务、产物查询；策略内部逻辑暂不强行抽象成一个大框架。
- 影响：迭代快、风险低；代价是脚本参数和产物格式需要持续收敛。

## D-008 pattern-cluster 产物用 bundle 组织

- 状态：已采用
- 背景：聚类策略同时包含聚类器、路由器、按簇模型、可视化和指标，单个 `model.pkl` 不够表达。
- 决策：使用 bundle 目录承载模型、配置、summary、可视化和回测产物，API 通过 `routers/bundles.py` 查询。
- 影响：前端可以按 bundle 展示训练结果；后续需要稳定 bundle schema。

## D-009 前端使用 Vue/Vite，与 FastAPI 分离启动

- 状态：已采用
- 背景：当前 dashboard 需要表单、任务轮询、图表和结果明细，静态 HTML 已不够顺手。
- 决策：仓库根 `webui/` 使用 Vue 3 + Element Plus + ECharts；后端 `limit_up_project/api` 单独启动。
- 影响：开发时需要两个服务；`start_dev.bat` 提供一键启动。

## D-010 当前文档以项目根五件套作为 Harness 事实索引

- 状态：已采用
- 背景：原有文档分散且部分内容已过时或编码显示异常。
- 决策：以 `PROJECT_MAP.md`、`ARCHITECTURE.md`、`TASKS.md`、`DECISIONS.md`、`RUNBOOK.md` 作为当前维护入口。
- 影响：新同学或未来自己先读五件套，再读 `docs/` 下专题文档。

## D-011 涨停池历史深度依赖 AkShare 接口实际供给

- 状态：已采用
- 背景：构建 zt_pool 缓存时发现 AkShare `stock_zt_pool_em` 对超过约 1 个月的历史日期常返回空（首次构建 1667 工作日 / 1652 个空文件）。
- 决策：`ZtPoolLoader.fetch_one` 对空数据写空 parquet 占位；阶段 1.4 校验的"连续空段"设为 WARN 而非 FAIL；`build_zt_pool_cache.py` 设计为每日累积式而非一次性 backfill。
- 影响：深度历史涨停池只能逐日累积；需要 2020-2024 历史时需另寻数据源（Tushare Pro `limit_list_d` 或自建涨停判定器）。

## D-012 数据一致性校验阈值现实化（阶段 1.4）

- 状态：已采用
- 背景：首次跑 `verify_daily_cache.py` 时硬阈值导致 3 项 FAIL，但都不是真 bug：T 日盘后未刷新、除权除息日 change_pct 跳变、下午弱封板客观占 20-25%。
- 决策：
  - 覆盖度阈值 99% → 95%（容忍停牌/退市/T 日缺失）
  - 价格容忍 1% → 2%（覆盖前复权/不复权差异）
  - chg_out_of_band 硬 0 → 占比 ≤ 1%
  - Tier1-3 占比 95% → 70%
  - 新增除权日豁免：股名带 `XD/DR/XR`，或 daily 跳空 < -8% 且 zt > 8%
  - 默认 `--exclude-today`，盘后跑加 `--include-today`
- 影响：验收门槛贴近真实数据特征，阶段 1.4 一次跑过；除权日相关样本不再误判。后续如改用前复权日线，可把价格容忍调回 1%。

## D-013 build_daily_cache 默认 universe 为 zt_pool_relevant

- 状态：已采用
- 背景：1.3 阶段日线回溯可以选"全市场主板"或"涨停池相关代码"。
- 决策：默认 `zt_pool_relevant`（zt_pool ∪ zt_pool_previous 主板代码并集，约 2-3k 只）；`--universe all` 兜底。
- 影响：首次构建从 6-8 GB 降到 < 2 GB；将来引入"非涨停股反例"时需切 `all` 重跑。
