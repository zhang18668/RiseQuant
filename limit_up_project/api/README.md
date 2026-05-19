# FastAPI Backend (RiseQuant Pattern-Cluster)

REST API for the Vue frontend. Wraps existing pipelines as async jobs.

## 启动

```cmd
cd G:\AI\RiseQuant\limit_up_project
set PYTHONPATH=%CD%
pip install fastapi uvicorn pydantic
uvicorn api.main:app --reload --port 8000
```

`http://127.0.0.1:8000/docs` — OpenAPI 文档。

## 接口总览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET  | /api/health | 健康检查 |
| GET  | /api/jobs?kind=&limit= | 列任务 |
| GET  | /api/jobs/{id} | 单个任务详情（含 logs/result/error） |
| POST | /api/data/fetch | 提交数据获取任务（TDX） |
| POST | /api/data/clean | 提交清洗任务（基于已 fetch 的 cache） |
| POST | /api/data/dryrun | 提交 dry-run 诊断任务 |
| POST | /api/train | 提交训练任务（subprocess 调 wash_pattern_train.py） |
| POST | /api/backtest | 提交回测任务（subprocess 调 wash_pattern_backtest.py） |
| GET  | /api/bundles?base= | 列所有 bundle |
| GET  | /api/bundles/detail?bundle_dir= | 单个 bundle 详情（含 per-cluster metrics） |
| GET  | /api/bundles/visualization-list?bundle_dir= | 该 bundle 的可视化文件名 |
| GET  | /api/bundles/visualization?bundle_dir=&name= | 取单张 PNG（base64） |
| GET  | /api/bundles/backtests?base= | 列所有回测 run |
| GET  | /api/bundles/backtest-detail?run_dir= | 单次回测明细（含 equity_curve/trades 前 1000 行） |
| POST | /api/optimize/search | 参数搜索（grid/random，按 objective 排序） |
| POST | /api/optimize/sensitivity | 单参敏感性（其他参数固定，扫一组值） |

## 任务模型

每个 POST 立刻返回 `{id, kind, state: pending, params, ...}`，后台开始 BackgroundTask 执行。
状态变迁：`pending → running → done | failed`。
前端通过轮询 `GET /api/jobs/{id}` 拿最新状态、日志、结果。

任务状态文件落到 `limit_up_project/models/_jobs/{job_id}.json`。

## 长任务设计取舍

- **训练 / 回测** 用 `subprocess` 调现成的 `scripts/*.py`，避免 FastAPI 进程内吃满内存
- 每次 `subprocess` 会捕获 stdout/stderr 的最后 2000 字符放进 job.result
- 真实日志通过 Python logging handler 同步进 job.logs

## 优化任务

- `optimize/search`: 笛卡尔积或随机抽样跑 N 次回测，按目标指标排名
- `optimize/sensitivity`: 固定 base_params，对单个参数扫一组值，画曲线用

## 跨域

默认 `allow_origins=["*"]`，开发时方便；上线请改 `api/main.py` 里的 `CORSMiddleware`。
