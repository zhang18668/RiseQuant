# 首板-震荡-第二涨停（Wash-Second）策略详解

> 本文档对应 `scripts\wash_second_pipeline.py` 和它依赖的全部模块。  
> 目的：让你能在不读代码的前提下，**判断现在的胜率低到底卡在哪一步**，并给出可改进点。

---

## 0. TL;DR — 一句话画像

我们要在一个**正在震荡洗盘的股票**身上识别"明天/后天/大后天可能直接拉涨停"的那几天，并在 T+1 开盘潜伏买入，跌破首板 open 即出。

数学上这是个**二分类问题**：
- 1 个样本 = 1 只股票在震荡区间的 1 天
- label = 1 表示这一天距下一次"创新高的非连板涨停" ≤ K 天（默认 K=3）
- label = 0 表示这一天没有这种"前几天潜伏"的价值

---

## 1. 业务定义（必须先对齐）

| 概念 | 严格定义 | 关键代码 |
|---|---|---|
| **首板** | 当日涨停（change_pct ≥ 9.9%），且**前 3 个交易日内没有任何涨停** | `WashSecondDetector.cooldown_days=3` |
| **震荡洗盘期** | 首板**之后**的 [3, 20] 个交易日，期间**不能再出现涨停**（否则那个涨停就是"真二板"，不是我们要的） | `min_gap=3`, `max_gap=20` |
| **第二涨停** | 震荡期结束时的那一根涨停。必须满足：(a) close > 首板 close（创新高） (b) 与首板间隔 ≥ 3 个交易日（排除连板） | `WashSecondDetector.detect` |
| **正样本日** | 震荡期内、距第二涨停 ≤ K=3 天的那几天。<br>例如 second_date=Day 16，positive_window=3，则 Day 13/14/15 为正样本 | `WashSampleBuilder.positive_window=3` |
| **负样本日** | 来自"失败首板"——严格首板，但 max_gap=20 内**没有**爆出符合条件的第二涨停。在这种 first 之后 [3, 20] 内的每一天都是负样本 | `WashSampleBuilder._find_failed_firsts` |
| **动态止损价** | 等于**首板的 open**。持仓期间当日 low ≤ stop_loss_price 即止损出场，按 stop_loss_price 成交 | `Position.dynamic_stop_price` |

⚠️ **这里有几个**容易被忽视的关键约束**，会直接影响胜率统计**：

1. **"中间不能有其他涨停"** — 这是 detector 里强约束。这导致一只票如果首板后第 5 天又涨停了，就不算 (first, second) 对。你预期的"震荡 + 涨停"必须是**不包含中间任何涨停的窗口**。
2. **"第二涨停 close > 首板 close"** — 必须创新高，否则不进训练集。这条件其实非常严格，会把"二板炸板低开"那类滤掉。
3. **"前 3 日无涨停"的首板** — 比传统"首板=前 1 日不涨停"严格得多。

---

## 2. 端到端流程图

```
        通达信日线 (date, code, OHLCV, change_pct)
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │  Step 1  WashSecondDetector             │
   │  - 扫描每只票, 找严格首板                │
   │  - 在 [first+3, first+20] 找下一个涨停    │
   │  - 校验创新高 + 中间无涨停                │
   │  输出: events 表 (first, second, gap, stop_loss_price=first_open)
   └─────────────────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │  Step 2  WashSampleBuilder              │
   │  - 正样本: 每个 (first, second) 对中, 震荡区间内距 second ≤ 3 天的每一天
   │  - 负样本: "失败首板"在 [first+3, first+20] 的每一天
   │  输出: samples 表 (sample_id, code, first_date, potential_date,
   │                    second_date|NaT, label_pre, stop_loss_price)
   └─────────────────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │  Step 3  WashFeatures                   │
   │  - 对每个 (first_date, potential_date) 算 28+ 维特征
   │  - 7 大类: 价格位置 / 均线粘合 / MACD收敛 / RSI / 量能 / 形态 / 当日动作
   │  - 切片严格 [first_idx, potential_idx], 无未来函数
   │  输出: features_df (f_w_* 列)
   └─────────────────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │  Step 4  SectorStockSplitter.split_by_date_ranges
   │  按 potential_date 切 train/valid/test (严格时间序 + 日期边界对齐)
   │  Step 5  LimitUpModelTrainer (LightGBM 二分类)
   │  Step 6  保存到 models/wash_second/run_<ts>/
   │          model.pkl, config.json, train_metrics.json,
   │          feature_importance.csv
   └─────────────────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │  Step 7  生成信号 + Backtester           │
   │  - score = P(label_pre=1)               │
   │  - 取 score ≥ min_score (默认 0.6)       │
   │  - T+1 开盘买, 一字涨停跳过              │
   │  - 持仓 ≤ sell_n 天                      │
   │  - 跌破 stop_loss_price (首板 open) 立即出
   │  Step 8  保存到 models/wash_second/run_<ts>/
   │          backtest_metrics.json, trades.csv, equity_curve.csv
   └─────────────────────────────────────────┘
```

---

## 3. 每一步的实现细节 + 可能影响胜率的点

### 3.1 事件检测（WashSecondDetector）

**关键代码（伪码）**：
```python
for 每只股票 g:
    for i in range(len(g)):
        if not g[i] 涨停:
            continue
        # 首板必须前 3 日内无涨停
        if g[max(0, i-3) : i] 中有任何涨停:
            continue
        # 在 [i+1, i+20] 区间找下一个涨停
        for j in range(i+1, min(n, i+max_gap+1)):
            if g[j] 涨停:
                if j >= i + min_gap = i + 3:
                    second_idx = j
                break    # 中间已有涨停, 不管是否符合 gap 都停
        if second_idx is None: continue
        if g[second_idx].close <= g[i].close: continue   # 必须创新高
        记录 (i, second_idx)
```

**胜率风险点**：

1. **`max_gap=20` 太短** — A 股震荡可能持续 30+ 个交易日。把 max_gap 调到 30 甚至 40，会让样本变多，但负样本质量也下降。
2. **`positive_window=3` 偏紧** — 模型必须在第二涨停**前 3 天**抓到信号，留给"潜伏 + 形态确认"的窗口很窄。如果第二涨停前 5 天就有明显信号（缩量、均线粘合、MACD金叉），模型也学不到，因为它们被标成负样本。
3. **"中间不能有涨停"** — 实际洗盘中可能出现一个"假涨停"被瞬间打开，但 detector 一刀切跳过。这会把一些**真实存在的"震荡-第二涨停"对**误标成负样本（因为这些 first 没有匹配到 second，被划进失败首板的负样本池），**污染训练数据**。
4. **"创新高"** — 第二涨停 close 必须 > first close。但实务中"次新高的二板"（close 略低于首板 close 但开盘是涨停板）也是有效信号，被排除了。

### 3.2 样本构造（WashSampleBuilder）

**正样本逻辑**：
```python
对每个 (first, second) 对:
    震荡区间 = [first+3, second-1]
    对其中每一天 t:
        days_to_second = second - t       # 距第二涨停的天数
        label = 1 if days_to_second <= 3 else 0
```
- 注意：**震荡区间内 days_to_second > 3 的天也被记录**，label=0。比如 gap_days=10, second 在 first+10，那么 [first+3, first+6] 是负样本（距 second 4-7 天），[first+7, first+9] 是正样本（距 second 1-3 天）。

**负样本逻辑**：
```python
失败首板 = 严格首板 - 已成功首板
对每个失败首板:
    在 [first+3, first+20] 的每一天都生成 label=0 样本
```

**胜率风险点**：

1. **严重的类不平衡** — 假设震荡区间平均长度 8 天，每个成功 (first,second) 对最多给 3 个正样本（距 second 1/2/3 天的那 3 天），但失败首板可能贡献 18 个负样本。**负样本数量是正样本的 3-10 倍**。LightGBM 默认不处理类不平衡，模型倾向把所有样本预测为 0，**正样本召回率低 → 信号稀少 → 实际可交易机会少**。
2. **"震荡前 7 天的样本是负样本"** — 这个 label 设计让模型学到的是"是不是处于震荡末段"而不是"是不是好潜伏点"。如果你想"第二涨停前 5-7 天就开始建仓"，需要把 positive_window 拉大到 7。
3. **失败首板样本 = 完整 [first+3, first+20]** — 这里所有样本都算"负样本"。但其中可能包含一些"形态非常像正样本（缩量、粘合）"的日子，这些反例对模型学习反而有害（标签噪声）。

### 3.3 特征工程（WashFeatures）

特征分为 7 大类共 28 维（具体列名以 `f_w_*` 前缀）：

| 类别 | 列名 | 含义 |
|---|---|---|
| **价格位置** | `f_w_close_to_fc`, `f_w_close_to_fo`, `f_w_close_to_fh`, `f_w_open_vs_fc`, `f_w_broke_stop`, `f_w_max_drawdown_from_fc`, `f_w_max_high_over_fc` | 当前 close 相对首板 close/open/high 的位置，是否曾跌破首板 open（止损位） |
| **均线** | `f_w_ma5`, `f_w_ma10`, `f_w_ma20`, `f_w_ma_cohesion`, `f_w_ma_bullish`, `f_w_close_above_all_ma` | MA 值、粘合度（标准差越小越粘合）、多头排列、close 是否在所有 MA 之上 |
| **MACD** | `f_w_macd_dif`, `f_w_macd_dea`, `f_w_macd_hist`, `f_w_macd_golden_x`, `f_w_macd_hist_up3` | DIF/DEA/HIST、金叉信号、柱状连续 3 日放大 |
| **RSI** | `f_w_rsi`, `f_w_rsi_neutral` | RSI14 + 是否在 40-60 中性区间 |
| **量能** | `f_w_vol_shrink`, `f_w_vol_shrink_streak`, `f_w_recent_vol_ratio` | 震荡均量/首板量、连续缩量天数、近 3 日量/震荡均量 |
| **形态** | `f_w_consol_days`, `f_w_amp_mean`, `f_w_amp_recent5`, `f_w_amp_convergence` | 横盘天数（振幅<3%）、平均振幅、最近 5 日振幅、振幅收敛比 |
| **当日动作** | `f_w_today_ret`, `f_w_today_amp`, `f_w_today_vol_ratio` | 今日相对昨日涨跌、今日振幅、今日量比 |

**胜率风险点**：

1. **缺关键因子** — 没有：
   - **筹码集中度** / **股东户数变化** / **大单净流入**（实务中"第二涨停前主力建仓"才是核心信号）
   - **题材热度** / **板块联动**（板块龙头补涨是常见模式）
   - **分时数据**（首板分时形态决定后续走势，但我们只用日线）
2. **均线粘合定义粗糙** — 当前用 MA5/10/20 标准差/close。但真正的"粘合发散"是要看**最近 5-10 天均线之间距离的变化趋势**，而不是当日的瞬时值。建议加：
   ```
   f_w_ma_cohesion_trend = 最近 5 日 ma_cohesion 的斜率
   f_w_ma_distance_5d_avg = 最近 5 日 (max(MA)-min(MA))/close 的均值
   ```
3. **`f_w_today_*` 类特征噪声大** — 单日数据对涨停预测的信号噪比不高，可以聚合成"最近 3 日"或"最近 5 日"统计量。
4. **没有"距首板涨停的天数"作为单独特征** — `gap_days_so_far` 字段已经在 samples 里有，但**没有作为模型特征**。建议把它加到 feature_cols 里。
5. **没有相对市场的强度** — 同样的"缩量横盘"，在熊市和牛市意义完全不同。可以加入大盘特征（已有 F-008 模块，但没接进 wash pipeline）。

### 3.4 训练（LightGBM 二分类）

```python
LimitUpModelTrainer({"params": full_cfg.model_params})
  ↓
LGBMClassifier(objective="binary", ...)
  ↓ fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

**胜率风险点**：

1. **没设 `scale_pos_weight` 或 `is_unbalance=True`** — 类不平衡问题（见 3.2）没有在模型层缓解。
2. **没用 `early_stopping_rounds`** — 默认 n_estimators=100，可能过拟合或欠拟合。
3. **valid 集太小** — 如果你设的 valid_end - train_end 只有几个月，valid loss 噪声大，模型选错点。
4. **没做特征筛选** — 28 维特征里有些噪声特征（如单日动作），会稀释信号。

### 3.5 信号 + 回测

```python
proba = trainer.predict_proba(X)
score = proba[:, 1]                  # P(label_pre=1)
signals = (date, code, score, stop_loss_price)
Backtester:
  - score ≥ min_score (默认 0.6)
  - T+1 open 买入
  - 一字涨停跳过 (open >= 9.5% 高开)
  - 持仓 ≤ sell_n 天 (默认 8)
  - 当日 low ≤ stop_loss_price → 触发 SELL_STOP_DYNAMIC
```

**胜率风险点**：

1. **`min_score=0.6` 在类不平衡下偏低** — 模型预测概率分布会整体偏向 0，0.6 可能让大量噪声信号也通过。建议先看 `signals.csv` 里 score 的分布再调阈值。如果 P>0.6 的样本占比 < 1%，说明阈值合理；如果 > 10% 说明阈值太低。
2. **`topk=5` 但 score 分散** — 一只票在震荡期可能连续 3 天都通过阈值（同 first_date 的多个 potential_date 都被打高分），但你只能持 5 个仓位，导致同一只票被反复挂仓。当前 backtester 有 `held_codes` 去重，但如果其他票分数更高，就抢不到位置。
3. **回测的 score 不含**：
   - 没有"出场后冷静期" — 同一只票止损后第 2 天又满足条件，会立刻再买（可能继续亏）
   - 没有"日内择时" — T+1 开盘一刀切，但如果 T+1 开盘已经跳水 5%，止损可能直接被打穿
4. **一字涨停过滤可能误杀** — 阈值 9.5% 比较严格，**温和高开 5-9% 的也是难买进的**。实务中应该放宽到 6-7%。

---

## 4. 现在胜率低的可能原因（排查清单）

请按下面顺序检查 `models\wash_second\run_<ts>\` 下的产物。如果你能把这些数据贴出来，我可以帮你定位问题。

### 4.1 看 `train_metrics.json`

- 关注 `accuracy`、`precision`、`recall`、`f1`、`auc`
- 如果 `auc < 0.55` → **特征不够好或标签噪声大**，问题在 3.2/3.3
- 如果 `auc > 0.7` 但回测胜率低 → **回测撮合或类阈值有问题**，问题在 3.5

### 4.2 看 `feature_importance.csv`

- **Top 5 是哪些列？** 应该集中在量能、均线、MACD。
  - 如果 Top 5 都是 `f_w_today_*` 单日特征 → 模型在记噪声
  - 如果 `f_w_close_above_all_ma`、`f_w_macd_golden_x` 排前 → 学到了你预期的形态
- **重要性是否过于集中**？前 3 个特征加起来 > 70% 重要性，说明其他特征基本没用，特征工程要重做

### 4.3 看 `backtest_metrics.json` 的 `sell_reasons`

```json
{
  "SELL_STOP_DYNAMIC": 80,
  "SELL_TIME_STOP": 20,
  "SELL_TAKE_PROFIT": 0
}
```

- **STOP_DYNAMIC 占比高** → 大量信号是"刚买就跌破首板 open"，**模型选了根本不该出手的票**
- **TIME_STOP 占比高且 return 接近 0** → 持仓 8 天后被动平仓，**没等到第二涨停**，说明 positive_window 设置和实际 gap 分布不匹配
- **TAKE_PROFIT=0** → 你目前没设硬止盈，正常

### 4.4 看 `trades.csv` 的 return_pct 分布

```python
import pandas as pd
t = pd.read_csv("models/wash_second/latest/trades.csv")
sells = t[t["action"].str.startswith("SELL")]
print(sells["return_pct"].describe())
print(sells.groupby("action")["return_pct"].agg(["mean", "median", "count"]))
```

- **平均盈亏比** = mean(正收益) / |mean(负收益)|，应该 ≥ 1.5
- 如果 STOP_DYNAMIC 的平均亏损 > 5%（远超首板 open 的理论 -2% 左右），说明**很多止损在开盘跳空触发**，实际亏损被低估

### 4.5 检查正负样本比例（在 train_metrics.json 的 split_sizes 附近）

如果训练集 label=1 占比 < 5%，**类不平衡严重**，需要：
- 训练参数加 `is_unbalance=True` 或 `scale_pos_weight=(n_neg/n_pos)`
- 或者降采样负样本（每个失败首板只取 5 天而不是 18 天）

---

## 5. 改进路线建议（按 ROI 排序）

### A. 立即可做（半小时见效）

1. **加 class_weight**：训练参数加 `is_unbalance: true`，让模型对正样本敏感
2. **放宽 `max_gap` 到 30**，`positive_window` 到 5：让正样本数量翻倍
3. **回测加冷静期**：同一只票止损后 5 个交易日内不再买入

### B. 短期改进（半天到一天）

4. **`gap_days_so_far` 进特征列**：让模型知道"我已经潜伏了多少天"
5. **大盘特征接进来**：用 `MarketFeatures`（F-008 已实现），让模型看大盘环境
6. **重新设计 label**：把"距 second_date 的天数"做成回归标签，按 days_to_second 倒数加权，让"距离近"的样本权重更大
7. **特征加聚合统计**：把单日特征改成"最近 5 日均值/标准差"

### C. 中期改进（需要补数据）

8. **接入分时数据**：首板分时形态（早封/尾盘秒板/封单大小）
9. **接入概念板块**：题材热度 + 板块联动
10. **接入资金流**：主力净流入、大单占比、北向资金（如果你有数据源）

### D. 框架层（重构）

11. **改用 GroupKFold by code**：避免同一只票在 train/valid 都有样本（虽然时间不交叉，但同股票相关性强）
12. **多模型集成**：LightGBM + XGBoost + CatBoost，取概率均值降噪
13. **离散化关键特征**：把 RSI、MACD HIST 做分桶，可解释性变好

---

## 6. 我会建议你先做的事

1. **打开 `models\wash_second\latest\train_metrics.json`** → 看 auc
2. **打开 `models\wash_second\latest\feature_importance.csv`** → 看 Top 10
3. **打开 `models\wash_second\latest\backtest_metrics.json`** → 看 `sell_reasons` 和 `sell_avg_return / sell_max_loss`
4. **用 Python 看一眼** `trades.csv` 里所有 SELL 的 return_pct 分布

把这 4 处的内容贴给我，我能给出**针对你具体数据**的下一步改动，而不是给一份通用建议。
