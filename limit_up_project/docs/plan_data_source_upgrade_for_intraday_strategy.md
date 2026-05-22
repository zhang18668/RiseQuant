# 涨停回踩策略升级 · 数据源完善计划

> 目标：支持新的 T+1 分时择时规则、按日 1 层/总 5 层的仓位管理，并在 dashboard 中
> 点击股票名查看「涨停日 + 后 3 日分时 + 前后 2 个月日K，标注买卖点与收益」。
>
> 本计划只写步骤 + 命令 + 关键代码片段，由你手动执行。

---

## 0. 现状速览（已实测）

| 数据源 | 现状 | 满足新需求？ |
|---|---|---|
| `data/daily_cache/*.parquet` | 683 支 × 2020-01-02 ~ 2026-05-20，列：`date code open high low close volume turnover change_pct` | 缺 `turnover_rate / 流通市值 / 总市值 / name`，需补 |
| `data/zt_pool_cache/zt_pool/*.parquet` | 1667 个文件，**仅 15 个非空**（2026-04-28~05-21） | **必须重灌** |
| `data/zt_pool_cache/zt_pool_previous/*.parquet` | 同上 | **必须重灌** |
| 股票名字典 | 仅 `scripts/dashboard_server.py` 从 `C:\new_tdx\T0002\hq_cache\*.tnf` 反查；机器无 TDX 即拿不到 | 需独立离线字典 |
| 分时（minute bar） | **完全没有**。`dashboard_server.py:528` 直接返回 "本地未发现通达信分钟数据文件" | **必须新建** |
| 回测引擎 | `src/strategy/limit_up_pullback.py::build_trade_ledger` 仅支持「次日 close 平仓 / 盘中触及 +3%」 | 需新建分钟级模拟器 |

---

## 1. 修复 zt_pool 缓存（最高优先级，**改用 tinyshare**）

> 2026-05-22 验证结果：akshare/eastmoney 历史接口 `rows=0`（被风控），
> **tinyshare `limit_list_d` 一次性返回 84 行**，是唯一通的数据源。
> 全量回填一律走 tinyshare 通道，对应脚本 `scripts/build_zt_pool_cache_tushare.py`。

### 1.1 准备 TINYSHARE_TOKEN

`scripts/validate_limit_pool_sources.py` 已经验证 token 配好了（rows=84）。
确认环境变量在你常用的 shell 里也能拿到：

```powershell
# PowerShell
echo $env:TINYSHARE_TOKEN

# 如果是空的，永久写入（仅当前用户）
[Environment]::SetEnvironmentVariable("TINYSHARE_TOKEN", "<你的token>", "User")
```

### 1.2 现有缓存如何处理

当前你的 `data/zt_pool_cache/` 状态：

| 文件 | 数量 | 来源 | 列结构 |
|---|---|---|---|
| 15 个非空 parquet（2026-04-28~05-21）| 15 | AkShare/eastmoney | 含 `涨停价/首次封板时间/最后封板时间/炸板次数` |
| 1652 个空 parquet（其它日期）| 1652 | AkShare 失败留下的占位 | 空 |

新脚本 `write_if_needed()` 的逻辑：

```python
if path.exists() and not force_refresh and parquet 非空:
    return "cached"            # 跳过，不调 API
# 否则 fetch + 写入
```

所以不论选哪种回填策略，1652 个空文件都会被 tinyshare 数据覆盖。
**唯一的差别在 15 个非空 AkShare 文件**：保留还是用 tinyshare 重写。

### 1.3 推荐：强制全量重拉（列结构统一）

```powershell
cd G:\AI\RiseQuant\limit_up_project

python scripts\build_zt_pool_cache_tushare.py `
    --start 20200101 `
    --end   20260522 `
    --pool both `
    --workers 4 `
    --rate-limit-per-min 120 `
    --force-refresh `
    --assert-monthly-non-empty `
    --min-monthly-non-empty 18
```

为什么推荐这个：
- 多调用 ~45 次接口（多 ~30 秒），换来**所有 parquet 文件列结构 100% 一致**，dashboard
  复盘时不会出现"前几天能看到字段，后几天看不到"的情况。
- 不需要 `Move-Item` 备份，新脚本直接原地覆盖。

参数说明：
- 单日 3 次接口调用（`limit_list_d` + `daily` + `daily_basic`）。客户端内置滑动窗口限速器
  会自动把**所有线程合计**调用速率压在 120 次/分钟内，**不会超额**。
- `--workers 4` 把 HTTP RTT 隐藏在限速窗口内，总耗时收敛到 4500 调用 ÷ 120 / min ≈ **38 分钟**（限速下限）。
  单线程顺序跑会被 RTT 拖到 ≈ 50 分钟以上。
- 若你的 tinyshare 套餐配额不同，把 `--rate-limit-per-min` 改成对应数字即可；
  例如付费版 500/min 写 `--rate-limit-per-min 500 --workers 8`，回填时间能压到 ~10 分钟。
- 跑完会自动断言每月非空 ≥ 18 文件，并实时打印 ETA。

### 1.4 备选方案 A：增量回填（保留 15 天 AkShare 数据）

```powershell
cd G:\AI\RiseQuant\limit_up_project

# 不加 --force-refresh，已有非空文件会被跳过
python scripts\build_zt_pool_cache_tushare.py `
    --start 20200101 `
    --end   20260522 `
    --pool both `
    --workers 4 `
    --rate-limit-per-min 120 `
    --assert-monthly-non-empty `
    --min-monthly-non-empty 18
```

- 节省 ~30 秒
- 15 天 AkShare 数据 vs 1500 天 tinyshare 数据，列字段不一致
- 不推荐除非你确定 AkShare 那 15 天的 `首次封板时间/炸板次数` 你以后会用，但 tinyshare 不能拿到

### 1.5 备选方案 B：先备份再强制重拉（保险派）

```powershell
cd G:\AI\RiseQuant\limit_up_project

# 1) 备份现有目录，留作随时回滚
Move-Item data\zt_pool_cache data\zt_pool_cache.bak_20260522

# 2) 全量重拉
python scripts\build_zt_pool_cache_tushare.py `
    --start 20200101 --end 20260522 --pool both `
    --workers 4 --rate-limit-per-min 120 `
    --assert-monthly-non-empty --min-monthly-non-empty 18
```

- 备份目录占 ~5 MB（绝大多数是空 parquet）
- 出问题可以 `Move-Item data\zt_pool_cache.bak_20260522 data\zt_pool_cache` 一键回滚

### 1.6 完成后立即校验

```powershell
python -c "from pathlib import Path; import pandas as pd; ne=[p for p in sorted(Path('data/zt_pool_cache/zt_pool').glob('*.parquet')) if len(pd.read_parquet(p))>0]; print('non-empty:', len(ne), 'first:', ne[0].stem, 'last:', ne[-1].stem)"

python -c "from pathlib import Path; import pandas as pd; ne=[p for p in sorted(Path('data/zt_pool_cache/zt_pool_previous').glob('*.parquet')) if len(pd.read_parquet(p))>0]; print('previous non-empty:', len(ne))"
```

**通过标准**：两个池子的非空文件数都接近交易日总数（约 1500+ 个），首文件是 `20200102`。

### 1.7 不再使用 akshare 回填脚本

`scripts/build_zt_pool_cache.py`（基于 akshare）保留代码不删，仅作为兜底；
日常**不再调用**。`scripts/validate_limit_pool_sources.py` 已经把
sources 列表精简到只剩 `tinyshare_limit_list_d`。

### 1.2 完成后立即校验

```bash
# 通过项目内 verify 脚本
python scripts\verify_zt_pool.py

# 或快速 Python 一行
python -c "from pathlib import Path; import pandas as pd; ne=[p for p in sorted(Path('data/zt_pool_cache/zt_pool').glob('*.parquet')) if len(pd.read_parquet(p))>0]; print('non-empty:', len(ne), 'first:', ne[0].stem, 'last:', ne[-1].stem)"
```

**通过标准**：`non_empty` ≥ 月份数 × 18，且 `missing=0`。

### 1.3 重跑研究脚本回归

```bash
python scripts\run_limit_up_pullback_research.py --source cache --start 2022-01-01 --end 2026-05-21
```

通过标准：`num_samples > 2000`，`samples.csv` 日期跨度覆盖 2022~2026。

---

## 2. 给 daily_cache 补齐 turnover_rate / 流通市值 / 总市值 / name

### 2.1 数据源

- 主：`akshare.stock_zh_a_spot_em()` → 当日全 A 快照，含 `换手率 / 流通市值 / 总市值 / 名称`
- 历史：`akshare.stock_zh_a_hist(symbol, adjust="qfq")` 不带换手率；换手率历史需 `akshare.stock_individual_info_em` 或 zt_pool 反推
- 备选：`mootdx.quotes.Quotes` 直读 TDX 服务器（含市值）

### 2.2 新增脚本 `scripts/enrich_daily_cache.py`

骨架（手写时按此思路落地即可）：

```python
"""
将 zt_pool / zt_pool_previous 中的 turnover_rate / float_market_cap / total_market_cap
回填到 data/daily_cache/<code>.parquet 中。
非涨停日的换手率，用 akshare.stock_zh_a_hist_min_em (period=daily) 兜底，
或保留 NaN，由策略层用涨停日数据近似。
"""
from pathlib import Path
import pandas as pd
from src.data.zt_pool_loader import ZtPoolLoader, ZT_POOL_TYPE, ZT_PREV_TYPE

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily_cache"

loader = ZtPoolLoader(cache_dir=ROOT / "data" / "zt_pool_cache", require_akshare=False)
zt = loader.load_cached_range("20200101", "20260522", pool_type=ZT_POOL_TYPE)
prev = loader.load_cached_range("20200101", "20260522", pool_type=ZT_PREV_TYPE)
combined = (
    pd.concat([zt, prev])
      .drop_duplicates(["trade_date", "代码"], keep="last")
      .rename(columns={"trade_date":"date", "代码":"code",
                       "换手率":"turnover_rate",
                       "流通市值":"float_market_cap",
                       "总市值":"total_market_cap",
                       "名称":"name"})
)
combined["date"] = pd.to_datetime(combined["date"])
combined["code"] = combined["code"].astype(str).str.zfill(6)

for parquet in DAILY_DIR.glob("*.parquet"):
    code = parquet.stem.zfill(6)
    daily = pd.read_parquet(parquet)
    daily["date"] = pd.to_datetime(daily["date"])
    add = combined[combined["code"] == code][
        ["date", "turnover_rate", "float_market_cap", "total_market_cap", "name"]
    ]
    merged = daily.merge(add, on="date", how="left")
    # 名称用前向填充（绝大多数股票名字稳定）
    merged["name"] = merged["name"].ffill().bfill()
    merged.to_parquet(parquet, index=False)
```

> 注意：当前 `daily_cache` 来自通达信 `.day`，不含换手率；akshare 历史接口也不提供日级换手率。
> 实务上**只在涨停/曾涨停的日子有 turnover_rate**就够策略 `wash_turnover_tier` 用，其余日子留 NaN。

### 2.3 同步修改 `src/strategy/limit_up_pullback.py::_turnover_rate_col`

现在该函数找不到任何列时返回 None → 全量条件直接 True。补齐后会真正走 tier 检查。
**不需要改代码**，只要 daily_cache 里出现 `turnover_rate` 列即可（已支持）。

---

## 3. 建立独立的股票名字典（不依赖本地 TDX 安装）

### 3.1 数据源

```python
import akshare as ak
df = ak.stock_info_a_code_name()   # 上交 + 深交 + 北交 全 A 静态字典
df.columns = ["code", "name"]
df["code"] = df["code"].astype(str).str.zfill(6)
df.to_parquet("data/dict/stock_name.parquet", index=False)
```

### 3.2 给 dashboard_server 加 fallback

`scripts/dashboard_server.py:114 load_stock_names()` 改为：

```python
def load_stock_names() -> dict[str, str]:
    global _STOCK_NAMES
    if _STOCK_NAMES is not None:
        return _STOCK_NAMES
    names: dict[str, str] = {}

    # 1) 优先读 parquet 字典
    parquet = Path(__file__).resolve().parent.parent / "data" / "dict" / "stock_name.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        names.update(dict(zip(df["code"].astype(str).str.zfill(6), df["name"].astype(str))))

    # 2) 再补 TDX (本机安装时)
    for path in (Path(r"C:\new_tdx\T0002\hq_cache\shs.tnf"),
                 Path(r"C:\new_tdx\T0002\hq_cache\szs.tnf"),
                 Path(r"C:\new_tdx\T0002\hq_cache\bjs.tnf")):
        if not path.exists():
            continue
        # ... 现有 TDX 解析逻辑 ...

    _STOCK_NAMES = names
    return names
```

---

## 4. 新建分钟级行情（minute bar）数据源 ★最大工作量

新策略与 UI 都依赖分钟数据，必须落地这一层。

### 4.1 三种可选数据源

| 方案 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| **A. mootdx**（推荐） | Python 直连 TDX 行情服务器，可拉 1m/5m/15m/30m/60m，速度快，免费 | 历史最长约 800 个 1m bar/股，不够全历史；可拉所有股票 5m | 历史回测用 5m，最近 30 日用 1m |
| **B. akshare** `stock_zh_a_hist_min_em` | 接口稳定 | 单次只能拉一只一天 1m，速度慢；返回字段含 `开盘 最高 最低 收盘 成交量 成交额` | 兜底/最新数据 |
| **C. 通达信 `.lc1` / `.lc5` 文件** | 本地零延迟，全历史 | 需要本机有装 TDX 且每日开过盘 | 已有 TDX 用户最佳 |

**推荐组合**：mootdx 5m 全历史 + akshare 1m 仅"涨停日及其前后 5 日"按需拉取。

### 4.2 缓存目录与文件结构

```
data/
└── intraday_cache/
    ├── 5m/
    │   └── <code>.parquet      # date(YYYYMMDD HHMM) | code | open | high | low | close | volume | amount
    └── 1m/
        └── <code>/
            └── <date>.parquet
```

### 4.3 新文件 `src/data/intraday_loader.py` 关键骨架

```python
"""分时数据加载器：mootdx 5m + akshare 1m 兜底。"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from mootdx.quotes import Quotes

CACHE_5M = Path("data/intraday_cache/5m")
CACHE_1M = Path("data/intraday_cache/1m")

class IntradayLoader:
    def __init__(self):
        self.client = Quotes.factory(market="std")  # 通达信标准行情

    def fetch_5m_full(self, code: str) -> pd.DataFrame:
        """单股 5m 全历史，落盘 parquet。"""
        market = 1 if code.startswith(("6", "5", "9")) else 0
        df_all = []
        offset = 0
        while True:
            df = self.client.bars(symbol=code, frequency=1, offset=offset, count=800)
            # frequency: 0=5m, 1=15m, 2=30m, 3=60m, 4=day, 7=1m, 8=tick
            if df is None or df.empty:
                break
            df_all.append(df)
            if len(df) < 800:
                break
            offset += 800
        out = pd.concat(df_all).drop_duplicates(["datetime"]).sort_values("datetime")
        out["code"] = code
        path = CACHE_5M / f"{code}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        return out

    def fetch_1m_around(self, code: str, center_date: str, days_before=1, days_after=3):
        """从 akshare 拉指定 [center-N, center+M] 区间 1m 数据，按日落盘。"""
        import akshare as ak
        from datetime import timedelta
        start = pd.to_datetime(center_date) - timedelta(days=days_before + 4)
        end   = pd.to_datetime(center_date) + timedelta(days=days_after + 4)
        df = ak.stock_zh_a_hist_min_em(
            symbol=code, period="1",
            start_date=start.strftime("%Y-%m-%d 09:00:00"),
            end_date=end.strftime("%Y-%m-%d 15:30:00"),
            adjust="",
        )
        # 落盘按日切分
        df["date"] = pd.to_datetime(df["时间"]).dt.date
        for date, g in df.groupby("date"):
            path = CACHE_1M / code / f"{date:%Y%m%d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            g.to_parquet(path, index=False)
```

### 4.4 批量构建脚本 `scripts/build_intraday_cache.py`

骨架：

```bash
# 5m 全宇宙
python scripts\build_intraday_cache.py --freq 5m --universe zt_pool --start 20200101 --end 20260522

# 1m 仅围绕涨停样本
python scripts\build_intraday_cache.py --freq 1m --from-samples models\rule_limit_up_pullback_research\run_<latest>\samples.csv --days-before 1 --days-after 3
```

### 4.5 dashboard_server 改造 `/api/minute`

```python
elif path == "/api/minute":
    code = params.get("code", [""])[0].zfill(6)
    date = params.get("date", [""])[0]     # YYYY-MM-DD，涨停日
    freq = params.get("freq", ["1m"])[0]
    self.send_json(intraday_for_code(code, date, freq))
```

`intraday_for_code` 返回涨停日 + 后 3 日的 1m bars，前端做 4 张分时图。

---

## 5. 新策略：T+1 分钟级择时引擎

旧引擎 `build_trade_ledger` 仅有「日级」决策点。新规则必须在分钟数据上模拟。

### 5.1 状态机定义（写进 `src/strategy/intraday_exit.py`）

```python
"""T+1 日内择时退出引擎。"""
from dataclasses import dataclass
import pandas as pd

@dataclass
class IntradayExitParams:
    morning_lock_until: str = "10:00"     # 10:00 前不卖
    profit_check_at:    str = "10:00"     # 10:00 决策点 A
    second_check_at:    str = "10:30"     # 10:30 决策点 B
    profit_target_pct:  float = 0.03      # +3% 止盈
    eod_loss_threshold_pct: float = -0.02 # 收盘跌 ≥2% 触发加仓
    addon_multiplier:   float = 1.0       # 加仓倍数（同等仓位）
    next_day_breakeven_exit: bool = True  # 加仓次日回本卖

def simulate_trade(
    bars_1m: pd.DataFrame,  # 列: datetime, open, high, low, close
    buy_price: float,
    params: IntradayExitParams,
) -> dict:
    """
    单笔交易状态机：
      - 09:30 ~ 10:00  : 持有（即便涨停也不卖）
      - 10:00 决策 A   : 若当前价 / buy_price - 1 > 3% 且未触及涨停 → 直接 10:00 卖出
                       : 否则进入 [10:00, 10:30] 观察窗
      - 观察窗内       : 任一分钟 high/buy_price-1 ≥ 3% → 按 buy_price*1.03 卖出（止盈）
      - 10:30 决策 B   : 若当前收盘价 > buy_price（红盘）→ 无条件按此价卖出
                       : 否则进入「持有到收盘」分支
      - 收盘 close     : 若 close/buy_price-1 <= -2% → 加仓一倍（记录 addon_buy）
                       : 否则按 close 卖出
      - T+2 日         : 若有 addon，回本（avg_cost）卖出
    返回: dict( sell_price, sell_time, qty_multiple, addon_buy, t2_sell_price, gross_pnl, return_pct )
    """
    bars_1m = bars_1m.sort_values("datetime").reset_index(drop=True)
    # ... 状态机实现，逐分钟扫描 ...
```

### 5.2 仓位管理（`src/strategy/portfolio.py`）

```python
@dataclass
class PortfolioParams:
    capital: float = 1_000_000.0
    layer_pct: float = 0.20              # 1 层 = 20% 仓位
    max_layers_total: int = 5            # 最高 5 层
    max_layers_per_code_per_day: int = 1 # 每只每日最多 1 层

def allocate_orders(
    candidate_df: pd.DataFrame,           # 当日所有进场信号，含 score 排序
    open_layers_by_code: dict[str, int],  # 当前持仓快照
    params: PortfolioParams,
) -> list[dict]:
    """按 score 降序逐一分配仓位，受总层数与单股限制约束。"""
```

### 5.3 引擎组装 `src/strategy/intraday_backtest.py`

```python
def run_intraday_backtest(
    samples: pd.DataFrame,
    intraday_loader: "IntradayLoader",
    exit_params: IntradayExitParams,
    pf_params: PortfolioParams,
) -> dict:
    """
    返回:
      trades:        DataFrame, 每笔交易明细
      equity_curve:  DataFrame, 日度净值
      layer_usage:   DataFrame, 日度仓位占用层数
    """
```

### 5.4 CLI `scripts/run_intraday_backtest.py`

```bash
python scripts\run_intraday_backtest.py ^
    --samples models\rule_limit_up_pullback_research\run_<latest>\samples.csv ^
    --capital 1000000 --layer-pct 0.20 --max-layers 5 ^
    --out models\rule_intraday_v1
```

产物落到 `models/rule_intraday_v1/run_<ts>/`:
- `trades.csv` 含 `buy_price, sell_price_t1, sell_time_t1, addon_price, sell_price_t2, qty_layers, return_pct`
- `equity_curve.csv` 含 `date, cash, market_value, total_value, layers_used`
- `layer_usage.csv`
- `summary.json`

---

## 6. Dashboard：点击股票名弹出分时 + K线 + 标注

### 6.1 后端新增 `/api/trade_detail`

```python
elif path == "/api/trade_detail":
    code = params.get("code", [""])[0].zfill(6)
    buy_date = params.get("buy_date", [""])[0]   # 涨停日 = limit_date
    self.send_json(trade_detail(model, code, buy_date))
```

`trade_detail()` 返回结构：

```json
{
  "code": "603063",
  "name": "汇洁股份",
  "limit_date": "2026-05-06",
  "intraday": {
    "limit_day":  [{"t":"09:30","o":..,"h":..,"l":..,"c":..,"v":..}, ...],
    "t_plus_1":   [...],
    "t_plus_2":   [...],
    "t_plus_3":   [...]
  },
  "kline_2m": [ {"date":"2026-03-01","o":..,...}, ... ],   // ±2 月日K
  "markers": [
    {"date":"2026-05-08","time":"10:00","action":"BUY",  "price":51.50, "layers":1},
    {"date":"2026-05-08","time":"10:00","action":"SELL", "price":56.65, "return_pct":10.0}
  ],
  "summary": {"holding_minutes": 30, "return_pct": 10.0}
}
```

### 6.2 前端 `web/index.html`

在股票名渲染处加 `onclick` 调出弹窗（React/纯 JS 都行）：

```js
async function openTradeDetail(code, buyDate) {
  const r = await fetch(`/api/trade_detail?code=${code}&buy_date=${buyDate}`).then(r=>r.json());
  // 用 echarts 渲染 4 张分时 + 1 张日K，再用 markPoint 标注买卖
  renderModal(r);
}
```

依赖 echarts ≥ 5.4（已用），分时图用 `xAxis.type="category"` 不画夜盘。

### 6.3 在 trades 表行的"股票"列加触发器

```html
<a href="javascript:void(0)" onclick="openTradeDetail('{code}','{buy_date}')">{name}</a>
```

---

## 7. 落地顺序与里程碑

| # | 阶段 | 工作量 | 完成判定 |
|---|---|---|---|
| M1 | 修复 zt_pool 缓存（§1） | 0.5 day（含 akshare 限速） | `verify_zt_pool.py` 通过；研究脚本 num_samples > 2000 |
| M2 | 补齐 daily 缓存字段（§2）| 0.5 day | `samples.csv` 的 `wash_turnover_tier` 不再全 `fixed` |
| M3 | 股票名字典（§3） | 0.2 day | dashboard 在无 TDX 机器上也能正确显示中文名 |
| M4 | mootdx 5m 全宇宙 + 缓存（§4.1-4.4） | 1 day | `data/intraday_cache/5m/*.parquet` 至少 600 支非空 |
| M5 | akshare 1m 围绕样本拉取（§4.4） | 0.5 day | 每个样本日有 ≥ 4 天 1m 数据 |
| M6 | 新分钟级择时引擎（§5.1-5.2） | 2 day | 单元测试覆盖：跳空高开/涨停/跌停/收盘加仓 4 种场景 |
| M7 | 回测引擎组装 + CLI（§5.3-5.4）| 1 day | `equity_curve.csv` 与 `trades.csv` 落盘，可被 dashboard 读取 |
| M8 | dashboard `/api/trade_detail` + 弹窗（§6）| 1 day | 点击任意股票弹出 4 张分时 + 1 张日K 且有买卖标注 |

总计：约 **7 个工作日**。强烈建议按 M1 → M2 → M3 → (M4+M5 并行) → M6 → M7 → M8 顺序推进。

---

## 8. 风险与坑

1. **akshare 限速**：`stock_zh_a_hist_min_em` 高频调用会被限流；批量构建务必加 `time.sleep(0.5)` 与重试。
2. **mootdx 历史长度**：单股 1m 最多 800 根 ≈ 3 天，所以 1m 历史只能靠 akshare；mootdx 用 5m 即可覆盖回测。
3. **涨停信号在分钟数据里识别**：判定"是否涨停"用 `high == limit_up_price`，而非 `change_pct ≥ 0.0995`，否则 ST/科创板会误判。需先确定每只票的每日涨停价（昨收 × 1.10 / 1.20 / 1.05）。
4. **加仓 + T+2 回本卖**：要小心 T+1 不能卖当日买入的股票（A 股规则），所以"收盘加仓"实际上是次日开盘加仓；建议参数化此细节。
5. **5 层总仓位约束在分钟级被打破**：早盘 9:30 一批信号同时进场会瞬时超 5 层，需要把 entry 决策放到 9:30 → 10:00 第一根 5m 后，并按 score 排序逐笔过 portfolio。

---

## 9. 立刻能动手的 2 条命令（tinyshare 通道，推荐项）

如果只想先把"数据空"问题解决，今天就跑（无需备份，新脚本会原地覆盖空文件 + 重写非空文件）：

```powershell
cd G:\AI\RiseQuant\limit_up_project

# 1) tinyshare 强制全量回填（4 线程并发 + 限速 120/min），约 38~45 分钟
python scripts\build_zt_pool_cache_tushare.py `
    --start 20200101 --end 20260522 --pool both `
    --workers 4 --rate-limit-per-min 120 --force-refresh `
    --assert-monthly-non-empty --min-monthly-non-empty 18

# 2) 校验非空数（应该 ≈ 1500+）
python -c "from pathlib import Path; import pandas as pd; ne=[p for p in sorted(Path('data/zt_pool_cache/zt_pool').glob('*.parquet')) if len(pd.read_parquet(p))>0]; print('non-empty:', len(ne), 'first:', ne[0].stem, 'last:', ne[-1].stem)"

# 3) 重跑研究，看样本量是否回归正常（期望 > 2000）
python scripts\run_limit_up_pullback_research.py --source cache --start 2022-01-01 --end 2026-05-21
```

如果不想强制覆盖那 15 个 AkShare 文件，把 `--force-refresh` 去掉即可（详见 §1.4）。

注：第 3 步如果样本仍然偏少，说明 daily_cache 的 `turnover_rate / 流通市值` 还是空的——
跳到 §2 写一次 enrich_daily_cache 再回头跑研究脚本即可。

跑完这 3 条，再开始 M3-M8 的字典 + 分钟级 + 弹窗工程。
