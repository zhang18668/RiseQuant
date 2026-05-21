"""
verify_zt_pool.py — 涨停板池接口可用性 / 字段完整度验证脚本

用途
----
- 验证 AkShare 的 ``stock_zt_pool_em`` 接口在本地环境下是否可正常访问
- 确认 17 个字段是否齐全、格式是否符合策略需要
- 统计主板个股数量、首次封板时间分档、封板强度的分布
- 输出样例 CSV 供人工检查

策略相关
--------
本脚本对应《策略规划.md》第 8.1 节"数据源分工"与第 5.1 节"多信号去重与排序"。
关键验证目标：``首次封板时间``、``封板资金``、``流通市值`` 三个字段必须可用。

运行
----
::

    cd G:/AI/RiseQuant/limit_up_project
    python scripts/verify_zt_pool.py

验收标准
--------
- 至少 3 个交易日成功拉取数据
- 17 个字段齐全
- 主板个股占比合理（一般 60-80%）
- ``首次封板时间`` 格式为 6 位 HHMMSS
- ``封板资金`` 与 ``流通市值`` NA 率 < 10%

通过则进入阶段 1.2：实现正式的 ``zt_pool_loader.py``。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("[FAIL] akshare 未安装。运行: pip install akshare")
    sys.exit(1)


# ---------------------------- 配置 ----------------------------
LOOKBACK_DAYS = 14          # 往前回溯候选天数
TARGET_TRADE_DAYS = 5       # 期望验证的交易日数量
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "verify"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "序号", "代码", "名称", "涨跌幅", "最新价", "成交额",
    "流通市值", "总市值", "换手率", "封板资金",
    "首次封板时间", "最后封板时间", "炸板次数",
    "涨停统计", "连板数", "所属行业",
]

CRITICAL_FIELDS = ["首次封板时间", "封板资金", "流通市值"]


# ---------------------------- 工具函数 ----------------------------
def get_recent_trade_dates(lookback: int = 14) -> list[str]:
    """从今天起往前找 ``lookback`` 个工作日，返回 YYYYMMDD 字符串列表（最新在前）。

    实际是否为交易日由 ``stock_zt_pool_em`` 返回是否为空判断。
    """
    dates: list[str] = []
    d = datetime.now().date()
    while len(dates) < lookback:
        d = d - timedelta(days=1)
        if d.weekday() < 5:  # 周一到周五
            dates.append(d.strftime("%Y%m%d"))
    return dates


def is_main_board(code: str) -> bool:
    """主板判定。沪市 600/601/603/605 + 深市 000/001/002（剔除创业板 300/301、科创板 688）。"""
    code = str(code).zfill(6)
    return code.startswith((
        "600", "601", "603", "605",
        "000", "001", "002",
    ))


def time_tier(time_str) -> str:
    """``首次封板时间`` 6 位 HHMMSS 字符串映射到时间档位。"""
    if pd.isna(time_str):
        return "未知"
    t = str(time_str).zfill(6)
    if t <= "093000":
        return "Tier1_一字板/开盘封"
    if t <= "100000":
        return "Tier2_9:30-10:00"
    if t <= "113000":
        return "Tier3_10:00-11:30"
    if t < "130000":
        return "中午_异常"
    return "下午_排除"


def verify_columns(df: pd.DataFrame, date: str) -> dict:
    """字段完整度与 NA 率检查。"""
    result: dict = {"date": date, "rows": len(df)}
    if df.empty:
        result["status"] = "EMPTY"
        return result

    actual_cols = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
    extra = [c for c in actual_cols if c not in EXPECTED_COLUMNS]
    result["missing_cols"] = missing
    result["extra_cols"] = extra
    result["status"] = "OK" if not missing else "MISSING_COLS"
    result["na_rate"] = {
        c: float(df[c].isna().mean()) if c in df.columns else None
        for c in CRITICAL_FIELDS
    }
    return result


# ---------------------------- 主流程 ----------------------------
def main() -> int:
    print("=" * 70)
    print("涨停板池接口验证 - akshare.stock_zt_pool_em")
    print(f"akshare version: {ak.__version__}")
    print("=" * 70)

    candidate_dates = get_recent_trade_dates(lookback=LOOKBACK_DAYS)
    print(f"\n候选验证日期（最近 {LOOKBACK_DAYS} 个工作日，自动跳过非交易日）：")
    for d in candidate_dates:
        print(f"  - {d}")

    all_results: list[dict] = []
    sample_saved = False
    actual_trade_days = 0

    for date in candidate_dates:
        if actual_trade_days >= TARGET_TRADE_DAYS:
            break

        print(f"\n[{date}] 拉取中 ...", end=" ", flush=True)
        try:
            df = ak.stock_zt_pool_em(date=date)
        except Exception as e:  # noqa: BLE001
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            all_results.append({"date": date, "status": "ERROR", "error": str(e)})
            continue

        if df is None or df.empty:
            print("空数据（非交易日 / 节假日 / 无涨停）")
            all_results.append({"date": date, "status": "EMPTY", "rows": 0})
            continue

        actual_trade_days += 1
        check = verify_columns(df, date)
        all_results.append(check)
        print(f"OK，共 {len(df)} 行")

        # ---- 主板过滤 ----
        df = df.copy()
        df["_is_main"] = df["代码"].apply(is_main_board)
        n_total = len(df)
        n_main = int(df["_is_main"].sum())

        # ---- 时间分档（主板）----
        df_main = df[df["_is_main"]].copy()
        df_main["_tier"] = df_main["首次封板时间"].apply(time_tier)
        tier_dist = df_main["_tier"].value_counts().to_dict()

        # ---- 封板强度（主板）----
        if not df_main.empty and "封板资金" in df_main.columns and "流通市值" in df_main.columns:
            df_main["_strength_pct"] = (
                df_main["封板资金"] / df_main["流通市值"] * 100
            )
            s_desc = df_main["_strength_pct"].describe(
                percentiles=[0.25, 0.5, 0.75, 0.9]
            )
        else:
            s_desc = None

        print(f"  总涨停: {n_total} | 主板: {n_main} | 非主板(创业/科创等): {n_total - n_main}")
        print(f"  主板时间分档:")
        for tier in sorted(tier_dist.keys()):
            cnt = tier_dist[tier]
            mark = "→排除" if tier.startswith("下午") else ""
            print(f"    {tier}: {cnt} {mark}")
        if s_desc is not None:
            print(
                f"  主板封板强度(%): "
                f"中位={s_desc['50%']:.2f}, "
                f"P75={s_desc['75%']:.2f}, "
                f"P90={s_desc['90%']:.2f}, "
                f"max={s_desc['max']:.2f}"
            )

        # ---- 样例 CSV ----
        if not sample_saved and not df_main.empty:
            sample_path = OUTPUT_DIR / f"sample_zt_pool_{date}.csv"
            df_main.sort_values("首次封板时间").to_csv(
                sample_path, index=False, encoding="utf-8-sig"
            )
            sample_saved = True
            print(f"  [SAVED] 主板样例 CSV: {sample_path}")

    # ============================ 总结 ============================
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    if actual_trade_days == 0:
        print("[FAIL] 没有拿到任何交易日数据。")
        print("可能原因：")
        print("  1. 网络问题，无法访问 push2ex.eastmoney.com")
        print("  2. AkShare 版本过低，建议升级: pip install -U akshare")
        print("  3. 接口被东财调整或限流（短期高频访问触发）")
        return 1

    print(f"成功验证交易日数: {actual_trade_days} / {TARGET_TRADE_DAYS}")

    field_ok_days = sum(1 for r in all_results if r.get("status") == "OK")
    print(f"字段齐全的交易日: {field_ok_days} / {actual_trade_days}")

    for r in all_results:
        if r.get("status") == "MISSING_COLS":
            print(f"  [{r['date']}] 缺失字段: {r['missing_cols']}")
        elif r.get("status") == "ERROR":
            print(f"  [{r['date']}] 异常: {r.get('error')}")

    print("\n关键字段 NA 率（按交易日）:")
    for r in all_results:
        if "na_rate" in r:
            line = ", ".join(
                f"{k}={v:.1%}" if v is not None else f"{k}=N/A"
                for k, v in r["na_rate"].items()
            )
            print(f"  [{r['date']}] {line}")

    print(f"\n样例数据保存目录: {OUTPUT_DIR}")
    print("请打开 CSV 人工检查 3 件事：")
    print("  1. 首次封板时间是否为 6 位数字（如 093000）")
    print("  2. 封板资金单位是否为元（一般在 1e6 ~ 1e9 量级）")
    print("  3. 主板个股代码前缀是否符合 600/601/603/605/000/001/002")

    print("\n验收结论:")
    if actual_trade_days >= 3 and field_ok_days >= 3:
        print("  [PASS] 接口可用，字段齐全 → 可以进入阶段 1.2 (zt_pool_loader 正式实现)")
        return 0
    else:
        print("  [FAIL] 接口或字段有问题，需要排查后再继续")
        return 2


if __name__ == "__main__":
    sys.exit(main())
