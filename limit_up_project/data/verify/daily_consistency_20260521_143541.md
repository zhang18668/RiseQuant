# 日线缓存一致性报告 (20260521_143541)

- 区间: `20200101 ~ 20260521`
- daily_cache_dir: `G:\AI\RiseQuant\limit_up_project\data\daily_cache`
- zt_cache_dir: `G:\AI\RiseQuant\limit_up_project\data\zt_pool_cache`
- **总体结论: PASS**

## 覆盖度
- total_rows=1004, missing=3, coverage=0.9970 -> PASS

## 价格/涨跌幅
- n_checked=1001
- 价格不一致=2 (0.20%) -> PASS
- 涨跌幅越界=3 (原始 7, 除权日豁免 5) -> PASS
- 涨停规则违反=0 -> PASS

## 首次封板时间分布
- Tier1-3 占比=75.10% -> PASS
- 分布: {1: 74, 2: 433, 3: 247, 5: 250}

## zt_pool 空文件连续段
- 异常段数: 1, 最长 1649 天 (仅 WARN)

## 未来函数自检
- scan_feature_names: OK