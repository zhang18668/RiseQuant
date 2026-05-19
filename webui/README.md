# RiseQuant WebUI

Vue 3 + Vite + Element Plus 前端，配合 `limit_up_project/api` 的 FastAPI 后端，提供：

- **数据**：数据获取（TDX 本地日线）/ 清洗 / dry-run 诊断
- **策略编写**：表单选参数 → 提交训练任务
- **回测**：选 bundle + 参数 → 提交回测，看实时日志和结果
- **优化**：参数搜索（grid/random） + 单参敏感性分析
- **训练产物**：浏览 bundle 列表、cluster 原型图、per-cluster 指标、特征重要性
- **任务列表**：所有后台任务的状态轮询 + 日志查看

## 启动

### 1) 启动后端 (FastAPI)

```cmd
cd G:\AI\RiseQuant\limit_up_project
set PYTHONPATH=%CD%
uvicorn api.main:app --reload --port 8000
```

第一次运行前确认装了依赖：

```cmd
pip install fastapi uvicorn pydantic
```

后端默认监听 `http://127.0.0.1:8000`，OpenAPI 文档在 `http://127.0.0.1:8000/docs`。

### 2) 启动前端 (Vite dev server)

```cmd
cd G:\AI\RiseQuant\webui
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`，Vite 已配好 `/api` 代理到后端。

## 一键启动 (Windows)

仓库根目录的 `start_dev.bat`：

```cmd
G:\AI\RiseQuant\start_dev.bat
```

会同时拉起后端和前端两个窗口。

## 项目结构

```
webui/
├─ package.json
├─ vite.config.js          ← /api 代理到 :8000
├─ index.html
└─ src/
   ├─ main.js              ← Vue 入口 + ElementPlus 注册
   ├─ App.vue              ← 侧边栏布局
   ├─ router.js            ← 路由 (hash 模式)
   ├─ api.js               ← axios 实例 + 各业务 API
   ├─ components/
   │  └─ JobWatcher.vue    ← 任务状态轮询组件
   └─ views/
      ├─ Data.vue          ← 数据 (fetch/clean/dryrun)
      ├─ Strategy.vue      ← 策略编写 (训练表单)
      ├─ Backtest.vue      ← 回测
      ├─ Optimize.vue      ← 优化 (搜索 + 敏感性)
      ├─ Bundles.vue       ← 训练产物可视化
      └─ Jobs.vue          ← 任务列表
```

## 常见问题

- **CORS 报错**：后端已开 `allow_origins=["*"]`，但需重启后端生效。
- **bundle 列表为空**：先在「策略编写」跑一次训练，或确认 `models/pattern_cluster/` 下有 `run_*` / `latest` 目录。
- **回测找不到 bundle**：路径需绝对路径或相对 `limit_up_project/` 根目录。
- **训练超时**：训练是 subprocess 跑，前端只轮询状态，不会超时；如长时间 pending 看 `models/_jobs/<job_id>.json` 排查。
