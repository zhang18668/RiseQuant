# RUNBOOK

> 如何启动、测试、部署和排查。命令默认在 Windows PowerShell 中运行。

## 环境准备

```powershell
cd G:\AI\RiseQuant\limit_up_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install fastapi uvicorn pydantic pyarrow
$env:PYTHONPATH = (Get-Location).Path
```

前端依赖：

```powershell
cd G:\AI\RiseQuant\webui
npm install
```

TDX 默认路径是 `C:\new_tdx\vipdoc`。如果本机路径不同，需要修改 `config/default.yaml` 的 `data.tdx_vipdoc`，或在相关脚本/API 参数里传 `tdx_path`。

## 一键启动开发环境

在仓库根目录运行：

```cmd
G:\AI\RiseQuant\start_dev.bat
```

启动后：

- 后端 API: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`
- 前端: `http://127.0.0.1:5173`

## 手动启动后端

```powershell
cd G:\AI\RiseQuant\limit_up_project
$env:PYTHONPATH = (Get-Location).Path
uvicorn api.main:app --reload --port 8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## 手动启动前端

```powershell
cd G:\AI\RiseQuant\webui
npm run dev
```

前端 API base 是 `/api`，开发时由 Vite 代理到后端。

## 数据缓存构建

验证涨停池接口：

```powershell
cd G:\AI\RiseQuant\limit_up_project
$env:PYTHONPATH = (Get-Location).Path
python scripts/verify_zt_pool.py
```

构建涨停池缓存：

```powershell
python scripts/build_zt_pool_cache.py --start 2020-01-01 --end 2026-05-21
```

构建日线缓存：

```powershell
python scripts/build_daily_cache.py
```

校验日线缓存：

```powershell
python scripts/verify_daily_cache.py
```

校验产物在 `data/verify/`，其中 `daily_consistency_<ts>.md` 适合人工阅读。

## 训练与回测

原始 pipeline：

```powershell
python scripts/run_pipeline.py
```

wash-second：

```powershell
python scripts/wash_second_pipeline.py
```

wash-ambush：

```powershell
python scripts/wash_ambush_pipeline.py
```

pattern-cluster 训练示例：

```powershell
python scripts/wash_pattern_train.py `
  --start 2021-01-01 --end 2025-12-31 `
  --train-end 2023-12-31 `
  --valid-end 2024-06-30 `
  --test-end 2025-12-31 `
  --bundle-type both `
  --out ./models/pattern_cluster
```

pattern-cluster 回测示例：

```powershell
python scripts/wash_pattern_backtest.py `
  --bundle ./models/pattern_cluster/latest `
  --bundle-type per_cluster `
  --bt-start 2025-01-01 --bt-end 2025-12-31 `
  --topk 5 --sell-n 8 --min-score 0.6
```

## API 提交任务

提交训练任务后轮询 job：

```powershell
$body = @{
  start = "2021-01-01"
  end = "2025-12-31"
  train_end = "2023-12-31"
  valid_end = "2024-06-30"
  test_end = "2025-12-31"
  bundle_type = "both"
  out = "./models/pattern_cluster"
} | ConvertTo-Json

$job = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/train `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod http://127.0.0.1:8000/api/jobs/$($job.id)
```

任务状态文件在：

```text
models/_jobs/<job_id>.json
```

## 测试

完整单元测试：

```powershell
cd G:\AI\RiseQuant\limit_up_project
$env:PYTHONPATH = (Get-Location).Path
pytest tests/ -v
```

防未来函数 smoke：

```powershell
python tests/_smoke_no_lookahead.py
```

按模块测试：

```powershell
pytest tests/test_event -v
pytest tests/test_feature -v
pytest tests/test_model -v
pytest tests/test_backtest -v
```

## 部署注意

- 当前 API CORS 为 `*`，正式部署前收窄域名。
- 当前任务队列是单机 JSON 文件队列，不适合多实例同时写同一个 `models/_jobs`。
- 训练/回测依赖本机 TDX 数据路径，生产机器要先确认 `C:\new_tdx\vipdoc` 或配置替代路径。
- 大量 parquet 缓存和模型产物不建议直接进 Git，应该按数据盘或对象存储管理。
- 前端正式构建：

```powershell
cd G:\AI\RiseQuant\webui
npm run build
```

## 常见排查

### 后端启动失败：找不到 `src`

确认在 `limit_up_project` 目录启动，并设置：

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### TDX 数据为空

检查：

- `config/default.yaml` 的 `data.tdx_vipdoc`
- 路径下是否存在 `sh/lday`、`sz/lday`
- 通达信是否已经下载本地日线数据

### API 任务一直 pending

检查后端窗口是否仍在运行；查看：

```powershell
Get-ChildItem G:\AI\RiseQuant\limit_up_project\models\_jobs |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5
```

再打开对应 JSON，看 `state`、`error`、`logs`。

### 训练任务 failed

优先看 `models/_jobs/<id>.json` 的：

- `error.traceback`
- `result.stderr_tail`
- `result.stdout_tail`

然后复制其中实际执行的 `scripts/*.py` 命令，在 PowerShell 中直接跑，通常能得到更完整报错。

### 前端 404 或接口无响应

确认：

- 后端在 `8000`
- 前端在 `5173`
- Vite 代理配置未改坏
- 浏览器访问 `http://127.0.0.1:8000/api/health` 返回 `{"ok": true}`

### parquet 写入失败

安装 pyarrow：

```powershell
pip install pyarrow
```

代码中部分路径会 fallback 到 CSV，但训练/缓存路径建议保持 parquet 可用。

### verify_daily_cache.py 报 FAIL

按 step 看 `data/verify/daily_consistency_<ts>.md`：

| 现象 | 原因 / 处理 |
|---|---|
| Step 2 缺失全是今日 | TDX 当日盘后未下载完。默认已 `--exclude-today`；盘后跑请加 `--include-today` |
| Step 3 chg_out_of_band 仍有少量 | 看 `mismatch_<ts>.csv`，若都是除权日（XD/DR/XR 或 daily 跳空 < -8%）则属正常，扩大豁免范围即可 |
| Step 3 价格不一致大于 1% | 通常是前复权 / 不复权差异。当前 PRICE_REL_TOL=2% 已覆盖，若爆出需检查 TDX 是否被通达信复权写脏 |
| Step 4 Tier1-3 < 70% | 数据本身就这样；策略侧应该用 `time_tier_rank ≤ 3` 过滤，而不是要求数据 |
| Step 5 空段最长 ~1649 天 | D-011 已知：AkShare 历史深度限制，仅 WARN 不影响 PASS |

强制忽略今日校验跑：

```powershell
python scripts/verify_daily_cache.py
```

盘后跑（把今天也算进来）：

```powershell
python scripts/verify_daily_cache.py --include-today
```

### test_event/test_golden_filter.py 收集错误

文件在 165 行处疑似被保存工具截断（`events = pd.DataFrame([{` 未闭合），与 daily_cache 工作无关。临时绕过：

```powershell
pytest tests/ --ignore=tests/test_event/test_golden_filter.py
```

待用 git 历史或备份恢复完整文件后再纳入测试。

### test_label/test_label_builder.py 报 AttributeError: LABEL_PERFECT

测试期望 `LabelBuilder.LABEL_PERFECT` 常量但实现已迁移到 `calc_label_combined` 返回 0/1/2。测试侧老化，对齐到 `assert b.calc_label_combined({...}) == 2` 即可。
