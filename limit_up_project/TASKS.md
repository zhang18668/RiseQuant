# TASKS

> 当前任务、完成状态和下一步。最后更新：2026-05-21。

## 已完成

### T-001 涨停池接口验证

- 入口：`scripts/verify_zt_pool.py`
- 内容：验证 AkShare 涨停池字段、空值、主板代码、封板时间分档。
- 状态：完成。

### T-002 涨停池缓存

- 入口：`scripts/build_zt_pool_cache.py`
- 产物：`data/zt_pool_cache/zt_pool/*.parquet` 与 `data/zt_pool_cache/zt_pool_previous/*.parquet`
- 当前覆盖：`2020-01-01` 到 `2026-05-21`
- 状态：完成。

### T-003 日线缓存代码

- 入口：`scripts/build_daily_cache.py`
- 核心：`src/data/daily_cache.py` 的 `DailyCacheManager`
- 数据源：TDX 优先，AkShare fallback。
- 测试：`tests/test_data/test_daily_cache.py`
- 状态：代码完成，依赖本机真实数据持续刷新。

### T-004 日线一致性校验代码

- 入口：`scripts/verify_daily_cache.py`
- 产物：`data/verify/daily_consistency_<ts>.json/md`，必要时输出 mismatch CSV。
- 校验项：覆盖率、价格一致性、涨跌幅越界、涨停规则、时间分档、防未来函数自检。
- 状态：代码完成。

### T-005 核心建模流水线

- 入口：`scripts/run_pipeline.py`
- 内容：TDX 日线加载、事件检测、特征计算、时间切分、LightGBM 训练、T+1 回测。
- 状态：已实现，可作为原始 baseline。

### T-006 策略实验流水线

- 入口：`scripts/wash_second_pipeline.py`、`scripts/wash_ambush_pipeline.py`、`scripts/wash_pattern_train.py`、`scripts/wash_pattern_backtest.py`
- 内容：wash-second、wash-ambush、pattern-cluster 的训练与回测。
- 状态：已实现，仍在策略迭代。

### T-007 API Harness

- 入口：`api/main.py`
- 内容：fetch、clean、dryrun、train、backtest、bundle 查询、优化搜索。
- 状态：已实现。

### T-008 前端 Dashboard

- 入口：仓库根 `webui/`
- 内容：Vue 3 + Element Plus + ECharts，提交任务、查看 job、查看 bundle 与回测结果。
- 状态：已实现开发版。

### T-009 Harness 文档五件套

- 文件：`PROJECT_MAP.md`、`ARCHITECTURE.md`、`TASKS.md`、`DECISIONS.md`、`RUNBOOK.md`
- 状态：本次已补齐。

## 进行中

### T-101 让主 pipeline 优先读取 `DailyCacheManager`

- 背景：`scripts/run_pipeline.py` 当前仍直接 `TDXDataLoader.load_batch()`。
- 目标：优先从 `data/daily_cache/<code>.parquet` 读取，缺失时提示先构建缓存或走 fallback。
- 价值：训练启动更快，减少重复读取 TDX 文件。
- 建议优先级：高。

### T-102 收敛脚本参数和产物 schema

- 背景：多条策略脚本参数命名、产物字段存在差异。
- 目标：统一 `config.json`、`summary.json`、`signals.csv`、`trades.csv`、`equity_curve.csv`、`backtest_metrics.json` 的最低字段。
- 价值：前端和 API bundle 查询更稳。
- 建议优先级：高。

### T-103 API job 日志增强

- 背景：当前 subprocess 只保留 stdout/stderr 尾部。
- 目标：把脚本完整日志路径写入 job result，或实时追加到 job logs。
- 价值：失败排查更快。
- 建议优先级：中。

## 下一步候选

### T-201 Dashboard 接入新缓存层

- 当前：`api/runner.py` 的 fetch 路径偏向直接 TDX 读取。
- 目标：优先使用 `DailyCacheManager` 和 `ZtPoolLoader`。
- 收益：前端操作不再强依赖 TDX 路径即时可读。

### T-202 加入资金流特征

- 目标：接入 AkShare 资金流或本地缓存，新增 `f_mf_*` / `f_w_*` 特征。
- 风险：接口稳定性与历史回溯速度。
- 收益：策略解释力和排序质量可能提升。

### T-203 分钟线特征

- 目标：解析 TDX `.lc5`，补充首板当日分时强弱、封板质量、炸板恢复等特征。
- 风险：数据体量大、清洗复杂。

### T-204 walk-forward 训练入口

- 目标：将 `src/utils/walkforward.py` 接入训练脚本，形成滚动训练/验证/回测报告。
- 收益：更接近真实上线更新节奏。

### T-205 任务队列生产化

- 目标：当任务量提升后，用 SQLite/PostgreSQL + worker 代替当前 JSON 任务队列。
- 触发条件：需要多进程、多机器、任务取消、排队优先级或权限审计。

### T-206 缓存/模型清理任务

- 目标：清理 `models/_cache_fetch`、旧 job、旧 backtest 产物。
- 注意：保留用户标记的基准 run。

## 已知风险

- TDX 路径硬编码默认 `C:\new_tdx\vipdoc`，换机器时容易失败。
- API CORS 开发期全开放，生产前必须收窄。
- 当前 git 工作区含大量数据缓存和 `__pycache__`，后续提交前要谨慎筛选。
- 多条策略 pipeline 并存，短期灵活，长期需要统一产物契约。

## 验收命令

```powershell
cd G:\AI\RiseQuant\limit_up_project
$env:PYTHONPATH = (Get-Location).Path
pytest tests/ -v
python tests/_smoke_no_lookahead.py
```
