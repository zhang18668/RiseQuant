# A 股数据源指南（针对 Wash-Second 策略 C 区改进）

> 目标：为 C1 分时、C2 概念板块、C3 资金流 三类需求选择合适的数据源。  
> 原则：**先用免费+本地**，等策略验证有正向收益再上付费。

---

## 0. 快速决策表

| 你的需求 | 推荐数据源（按优先级） | 备注 |
|---|---|---|
| **首板分时数据**（5/15/30/60分钟K线） | ① 通达信本地 `.lc5/.lc1` 文件 → ② AKShare 东方财富分钟接口 → ③ Tushare Pro | 通达信你已经在用，零成本；AKShare 免费但有限速 |
| **首板分时数据**（盘口/逐笔/Level-2） | ① 券商 QMT/PTrade（华泰/中信/国信免费给）→ ② 万得 L2 | L2 数据没有"免费"选项，至少需要券商账户 |
| **概念板块/题材热度** | ① AKShare 东财概念 → ② Tushare 概念分类 → ③ 同花顺 iFind | AKShare 的东财接口对个人完全够用 |
| **个股资金流（主力/超大单）** | ① AKShare 东财 → ② 通达信本地 → ③ Tushare Pro | AKShare 调用 `stock_individual_fund_flow` 即可 |
| **北向资金** | ① AKShare 东财 / 港交所 → ② Tushare Pro | 完全免费 |
| **龙虎榜** | ① AKShare → ② Tushare | 都免费 |

**最务实的组合（零成本起步）**：通达信（你已经有日线和分时）+ AKShare（概念/资金流/龙虎榜）+ 自维护 SQLite 缓存。

---

## 1. 免费数据源

### 1.1 通达信本地数据（你已经在用）

**文件位置**：`C:\new_tdx\vipdoc\`
- 日线：`sh\lday\sh600000.day` / `sz\lday\sz000001.day`
- **5 分钟**：`sh\fzline\sh600000.lc5` / `sz\fzline\sz000001.lc5`（每条 32 字节，格式同 `.day`）
- **1 分钟**：`sh\minline\sh600000.lc1` / `sz\minline\sz000001.lc1`
- 概念分类：`T0002\hq_cache\block.dat`（二进制，需逆向）

**优点**：完全免费、本地访问、覆盖 2005 年至今  
**缺点**：分时数据要每天点击通达信"盘后下载分钟数据"，否则没有最新数据  
**接入难度**：低，照搬现有 `TDXDataLoader` 写一个 `TDXMinuteLoader`

实现示例（`src\data\tdx_minute_loader.py` 骨架）：
```python
_MINUTE_RECORD_FMT = "<HHfffffii"  # date(uint16), time(uint16), open,high,low,close,amount(f32), volume,reserved(i32)

def parse_lc5(path):
    """5 分钟 K 线: date 编码 = ((year-2004)*2048 + month*100 + day),
                time = hour*60 + minute"""
    ...
```
**官方协议**：参见 https://www.tdx.com.cn/page_111.html

### 1.2 AKShare（强烈推荐）

**主页**：https://akshare.akfamily.xyz/  
**安装**：`pip install akshare --upgrade`  
**速率**：无强制限速，但每分钟 > 50 次会被东财风控（建议 sleep 0.5s/次）

#### 分时数据
```python
import akshare as ak
# 1 分钟数据 (只有近 5 个交易日, 适合实盘)
df = ak.stock_zh_a_hist_min_em(symbol="000001", period="1",
                                 start_date="2024-01-01 09:30:00",
                                 end_date="2024-01-02 15:00:00",
                                 adjust="qfq")
# 5/15/30/60 分钟 (历史可拉到 2 年, 但中间常有缺口)
df = ak.stock_zh_a_hist_min_em(symbol="000001", period="5", adjust="qfq")
```

#### 概念板块
```python
# 全部概念板块列表
boards = ak.stock_board_concept_name_em()
# 单个板块成分股
cons = ak.stock_board_concept_cons_em(symbol="人工智能")
# 板块历史行情 (按日)
hist = ak.stock_board_concept_hist_em(symbol="人工智能", period="daily")
```

#### 资金流
```python
# 个股资金流明细 (主力/超大单/大单/中单/小单)
flow = ak.stock_individual_fund_flow(stock="000001", market="sz")
# 大盘资金流
mkt_flow = ak.stock_market_fund_flow()
# 北向资金每日
north = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
# 龙虎榜
lhb = ak.stock_lhb_detail_em(start_date="20240101", end_date="20241231")
```

**优点**：免费、接口齐全、覆盖东财/同花顺/雪球数据  
**缺点**：网络抖动会导致接口偶尔超时；历史回看时段限制（分钟数据只回看 1-2 年）  
**接入难度**：低，直接 `import akshare as ak` 调函数

### 1.3 Tushare（基础版免费，Pro 付费）

**主页**：https://tushare.pro/  
**注册**：免费注册，需绑定手机；积分 ≥ 120 可用 Pro 接口（学生认证或邀请好友即可）

#### 你需要的接口
- `pro.daily()` 日线（基础）
- `pro.moneyflow()` 个股资金流（Pro，**需 2000 积分**）
- `pro.moneyflow_hsgt()` 北向资金（免费）
- `pro.concept()` / `pro.concept_detail()` 概念分类
- `pro.stk_mins()` 分钟数据（**需 5000 积分 / 每月 200 元**）
- `pro.top_list()` 龙虎榜

**优点**：数据干净、字段统一、有官方 SDK  
**缺点**：分钟和资金流是付费墙；高频接口积分门槛高  
**接入难度**：极低，`pip install tushare; ts.pro_api(token).daily(...)`

### 1.4 东方财富网页直爬（适合极特殊需求）

如果 AKShare 没有的接口，可以直接抓东财页面：
- 概念热度：`https://push2.eastmoney.com/api/qt/clist/get?fs=m:90+t:3`
- 行业资金流：`https://push2.eastmoney.com/api/qt/clist/get?fs=m:90+t:2`
- 个股 L1 分时：`https://push2his.eastmoney.com/api/qt/stock/trends2/get?secid=1.600000`

**风险**：东财风控比较严，IP 频繁请求会被封 1-24 小时。建议加随机 sleep + 失败重试。

### 1.5 其他免费数据

- **百度股市通** - 接口可用但维护不积极
- **新浪财经** - L1 分时可用 `sina.com.cn/realstock/company/000001/hisdata/`
- **腾讯证券** - `qt.gtimg.cn` 实时行情接口
- **港交所** - 北向资金权威源 `https://www.hkex.com.hk/eng/cbbc/...`

---

## 2. 付费数据源（按性价比排序）

### 2.1 Tushare Pro（最便宜，¥200/月）

200 元/月 = 全部接口包括分钟、L1 资金流、龙虎榜。**个人开发者首选**。

### 2.2 聚宽（JoinQuant）

- 网页平台：https://www.joinquant.com/
- 免费回测：可用日线 + 部分分钟
- 数据下载：付费会员可导出本地，¥150-300/月
- **优势**：研究环境（Jupyter）+ 因子库 + 模拟交易一站式

### 2.3 米筐（RiceQuant）

- 网页：https://www.ricequant.com/
- 数据：日线/分钟/L2 全有
- 价格：¥500+/月起，个人版便宜
- 与聚宽差不多

### 2.4 同花顺 iFind

- ¥3000/年起（学生版可能便宜）
- 概念板块和龙虎榜数据最全
- 数据下载 API 完整

### 2.5 万得 Wind

- ¥30000+/年（个人很难承担）
- L2 历史数据、上市公司公告全文、行业研报
- 量化机构标配

### 2.6 券商 QMT / PTrade（开户即免费！）

**这是被严重低估的免费 L2 数据源**。

| 券商 | 系统 | 资金门槛 | 你能拿到的数据 |
|---|---|---|---|
| 华泰 | 涨乐 / 涨乐扫单 | 100万 | L1 实时 + 5 年历史 L2 |
| 中信 | 信e投 / 量化平台 | 50万 | L1 + L2 委托队列 |
| 国信 | 金太阳 / QMT | 30-50万 | 类似 |
| 国金 | 佣金宝 | 0 起步 | L1 + 部分历史，PTrade |
| 国君 | 君弘 / QMT | 100万 | L1 + L2 |

**如果你的账户有 30 万+，去问开户经理"能不能开 QMT"**，免费就拿到分钟+L2 数据。

---

## 3. 三类数据的具体接入建议

### C1. 分时数据

**目标**：算出首板分时形态特征（早封/晚封、封板时间点、封单金额、是否炸板）

**最简单可行方案**：
```python
# 用 AKShare 拉首板日的 5 分钟 K 线
import akshare as ak
from datetime import datetime
def get_first_board_intraday(code, first_date):
    sym = code  # e.g. "000001"
    df = ak.stock_zh_a_hist_min_em(
        symbol=sym, period="5",
        start_date=first_date.strftime("%Y-%m-%d 09:30:00"),
        end_date=  first_date.strftime("%Y-%m-%d 15:00:00"),
        adjust="qfq"
    )
    return df  # 列: 时间, 开盘, 收盘, 最高, 最低, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 换手率
```

可以衍生的特征：
- `first_board_lock_time` 首次涨停的分钟数（早封 = 早盘前 60 分钟）
- `first_board_open_count` 涨停被打开的次数
- `first_board_close_volume_ratio` 收盘前 30 分钟成交量 / 全天
- `first_board_amount_at_lock` 封板时的累计成交额

**进阶**：拉前几天分时算"近期分时形态"，但数据量大，建议先做首板单日的。

**注意**：AKShare 分钟数据**只能回看 1-2 年**。如果你要回测 2020 年的数据，必须用通达信本地的 `.lc5`。

---

### C2. 概念板块 / 题材热度

**目标**：知道你这只票当日所属概念，以及这些概念的热度（涨停数、上涨家数、净流入）

**接入方案**：
```python
import akshare as ak

# Step 1: 个股 → 概念映射 (一次性, 缓存)
def build_code_to_concepts():
    """返回 dict: code -> [concept1, concept2, ...]"""
    boards = ak.stock_board_concept_name_em()
    mapping = {}
    for _, b in boards.iterrows():
        cons = ak.stock_board_concept_cons_em(symbol=b["板块名称"])
        for _, c in cons.iterrows():
            mapping.setdefault(c["代码"], []).append(b["板块名称"])
    return mapping

# Step 2: 每日概念热度
def get_concept_hot(date):
    """返回 dict: concept -> {涨停数, 上涨数, 涨幅, 资金流入}"""
    df = ak.stock_board_concept_name_em()
    return df.set_index("板块名称")[["涨跌幅", "总市值", "换手率", "上涨家数", "下跌家数"]].to_dict("index")
```

衍生特征：
- `f_concept_max_hot` 该票所属概念中最热概念的当日涨幅
- `f_concept_leader_lock` 该概念是否有龙头连板
- `f_concept_lu_count_5d` 该概念近 5 日涨停数

**注意**：东财概念**当日横切容易，历史回溯难**。如果要回测 2023 年，要么自己每天用 ak 抓存 SQLite，要么用 Tushare Pro 的 `concept_detail` 接口（有时间戳）。

---

### C3. 资金流

**目标**：在震荡期识别"主力净流入"（潜伏特征）

**接入方案**（AKShare）：
```python
import akshare as ak

# 个股逐日资金流 (近 100 个交易日)
flow = ak.stock_individual_fund_flow(stock="000001", market="sz")
# 列: 日期, 收盘价, 涨跌幅, 主力净流入-净额, 主力净流入-净占比,
#     超大单净流入-净额, 大单净流入-净额, 中单净流入-净额, 小单净流入-净额

# 北向资金每日
north = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")

# 北向资金持股明细 (按股票)
hold = ak.stock_hsgt_individual_em(stock="000001")
```

衍生特征：
- `f_w_main_inflow_avg_5d` 震荡期主力净流入 5 日均值
- `f_w_main_inflow_streak` 主力连续净流入天数
- `f_w_super_large_ratio` 超大单/总成交占比
- `f_w_north_change` 北向持股比例近 5 日变化

**注意**：AKShare 的 `stock_individual_fund_flow` **只返回近 100 个交易日**。要更长历史：
- Tushare Pro 的 `moneyflow` 接口（需 2000 积分）
- 或自己每天抓取入库

---

## 4. 推荐的工程方案

```
┌──────────────────────────────────────────────────┐
│  本地数据层 (SQLite / Parquet)                    │
│  /data/                                          │
│    daily.parquet           ← 通达信 .day         │
│    minute_5m.parquet       ← 通达信 .lc5         │
│    concept_map.parquet     ← AKShare 一次性抓    │
│    concept_hot_<date>.parquet  ← 每日定时         │
│    flow_<code>.parquet     ← AKShare 个股流      │
│    north_flow.parquet      ← AKShare 北向        │
└──────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  数据访问层 src/data/                             │
│    tdx_loader.py           已有                  │
│    tdx_minute_loader.py    新增 (C1)             │
│    akshare_concept.py      新增 (C2)             │
│    akshare_flow.py         新增 (C3)             │
│    cache_manager.py        已有 (parquet 缓存)    │
└──────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  特征层 src/feature/                              │
│    intraday_features.py    新增 — F-009          │
│    concept_features.py     新增 — F-010          │
│    fund_flow_features.py   新增 — F-011          │
└──────────────────────────────────────────────────┘
```

每个新数据源的接入步骤：
1. 写一个 `XxxLoader` 类，把"远程 / 本地原始格式 → 标准 DataFrame"
2. 配置缓存（parquet/sqlite），避免重复请求
3. 写对应的 `XxxFeatures` 类，按 `(code, event_date)` 输出特征
4. 在 `WashSecondPipeline.compute_features` 中 merge

---

## 5. 我推荐的实施顺序（按 ROI）

| 顺序 | 数据源 | 投入 | 预期对胜率的提升 |
|---|---|---|---|
| **第 1 周** | AKShare 资金流（C3）| 1-2 天 | +3~5% 胜率（最直接） |
| **第 2 周** | AKShare 概念板块（C2）| 2-3 天 | +2~5% 胜率 |
| **第 3 周** | 通达信分时（C1）| 3-5 天 | +5~10% 胜率（最大但最难） |
| 第 4 周+ | 北向资金/龙虎榜 | 1 天 | +1~2% 胜率（边际） |

**给你的具体建议**：

1. **优先做 C3 资金流** — AKShare 一个接口就拿到，代码量小、特征明确（主力净流入是经典因子）
2. **C1 分时**用通达信本地 → 零成本，但要写 `.lc5` 解析器（30 字节/条，仿照你已经在用的 `.day` 解析）
3. **C2 概念**最麻烦 — 历史板块归属变动频繁，建议每天自动跑 `ak.stock_board_concept_cons_em` 入库，未来回测就有数据

如果你已经有券商账户且资金 ≥ 30 万，**强烈建议开 QMT** — L2 数据带来的胜率提升远超人工接 AKShare。

需要我现在就开始写 `akshare_flow.py` 接入 AKShare 资金流吗？这是 C 区 ROI 最高的一项。
