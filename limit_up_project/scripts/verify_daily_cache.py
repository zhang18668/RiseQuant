"""verify_daily_cache.py — 日线缓存 ↔ 涨停池一致性校验（阶段 1.4）

校验项：
1. 覆盖度 — zt_pool 每条 (code, date) 都能在日线缓存找到
2. 价格一致性 — 最新价 ≈ close (容忍 PRICE_REL_TOL)
3. 涨跌幅一致性 — 涨跌幅 ≈ change_pct (容忍 CHG_ABS_TOL_PP)，除权日自动豁免
4. 涨停业务规则 — 主板 [9.5, 10.5]，ST [4.5, 5.5]
5. 首次封板时间分布 — Tier1-3 占比
6. zt_pool 空文件连续段 (仅 WARN)
7. 未来函数自检

输出 data/verify/daily_consistency_<ts>.{json,md}。任一硬验收项失败 -> exit 2。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.data.daily_cache import DailyCacheManager  # noqa: E402
from src.data.zt_pool_loader import (  # noqa: E402
    ZtPoolLoader, is_main_board, time_tier_rank,
)


DEFAULT_START = "20200101"
DEFAULT_END = datetime.now().strftime("%Y%m%d")
DEFAULT_DAILY_CACHE_DIR = _PROJ_ROOT / "data" / "daily_cache"
DEFAULT_ZT_CACHE_DIR = _PROJ_ROOT / "data" / "zt_pool_cache"
DEFAULT_VERIFY_DIR = _PROJ_ROOT / "data" / "verify"

# 验收门槛
THRESHOLD_COVERAGE_MIN = 0.95
THRESHOLD_PRICE_MISMATCH_MAX = 0.01
THRESHOLD_CHG_OUT_OF_BAND_RATE = 0.01
THRESHOLD_TIER_OK_MIN = 0.70
EMPTY_RUN_THRESHOLD = 5
PRICE_REL_TOL = 0.02
CHG_ABS_TOL_PP = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="日线缓存与涨停池一致性校验（阶段 1.4）")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--daily-cache-dir", default=str(DEFAULT_DAILY_CACHE_DIR))
    p.add_argument("--zt-cache-dir", default=str(DEFAULT_ZT_CACHE_DIR))
    p.add_argument("--verify-dir", default=str(DEFAULT_VERIFY_DIR))
    p.add_argument("--main-board-only", action="store_true", default=True)
    p.add_argument("--max-mismatch-rows", type=int, default=200)
    p.add_argument("--exclude-today", action="store_true", default=True,
                   help="校验时排除今日 (TDX 盘后下载前会假阳性, 默认开)")
    p.add_argument("--include-today", dest="exclude_today", action="store_false")
    return p.parse_args()


def load_zt_pool_long(zt_loader: ZtPoolLoader, start: str, end: str,
                      main_board_only: bool) -> pd.DataFrame:
    df = zt_loader.load_cached_range(start, end, pool_type="zt_pool")
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["code"] = df["代码"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["trade_date"])
    if main_board_only:
        df = df[df["code"].apply(is_main_board)].copy()
    return df.reset_index(drop=True)


def check_coverage(zt_long: pd.DataFrame, mgr: DailyCacheManager
                   ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    if zt_long.empty:
        return ({"total_rows": 0, "missing": 0, "coverage": 1.0,
                 "threshold": THRESHOLD_COVERAGE_MIN, "pass": True,
                 "missing_codes": 0},
                pd.DataFrame(), pd.DataFrame())

    missing_records: List[dict] = []
    matched_rows: List[pd.DataFrame] = []
    for code in sorted(zt_long["code"].unique()):
        zt_sub = zt_long[zt_long["code"] == code]
        daily = mgr.load_code(code)
        if daily.empty:
            for _, r in zt_sub.iterrows():
                missing_records.append(
                    {"code": code, "date": r["date"].strftime("%Y-%m-%d"),
                     "reason": "code_missing"})
            continue
        joined = zt_sub.merge(daily, on=["code", "date"], how="left",
                              suffixes=("_zt", "_d"), indicator=True)
        miss = joined[joined["_merge"] == "left_only"]
        for _, r in miss.iterrows():
            missing_records.append(
                {"code": code, "date": r["date"].strftime("%Y-%m-%d"),
                 "reason": "date_missing"})
        ok = joined[joined["_merge"] == "both"].drop(columns=["_merge"])
        if not ok.empty:
            matched_rows.append(ok)

    total = len(zt_long)
    missing = len(missing_records)
    coverage = (total - missing) / total if total else 1.0
    matched_df = (pd.concat(matched_rows, ignore_index=True)
                  if matched_rows else pd.DataFrame())

    summary = {
        "total_rows": total, "missing": missing, "coverage": round(coverage, 4),
        "threshold": THRESHOLD_COVERAGE_MIN,
        "pass": coverage >= THRESHOLD_COVERAGE_MIN,
        "missing_codes": len({m["code"] for m in missing_records}),
    }
    return summary, pd.DataFrame(missing_records), matched_df


def check_price_consistency(matched: pd.DataFrame, max_rows: int = 200
                            ) -> Tuple[Dict, pd.DataFrame]:
    if matched is None or matched.empty:
        return ({"n_checked": 0, "price_mismatch": 0, "price_mismatch_rate": 0.0,
                 "chg_out_of_band": 0, "chg_out_of_band_raw": 0,
                 "chg_out_of_band_rate": 0.0, "ex_dividend_excluded": 0,
                 "limit_up_rule_violation": 0,
                 "pass_price": True, "pass_chg_band": True, "pass_limit_up_rule": True,
                 "price_rel_tol": PRICE_REL_TOL, "chg_abs_tol_pp": CHG_ABS_TOL_PP},
                pd.DataFrame())

    df = matched.copy()
    px_zt = pd.to_numeric(df["最新价"], errors="coerce")
    px_d = pd.to_numeric(df["close"], errors="coerce")
    chg_zt = pd.to_numeric(df["涨跌幅"], errors="coerce")
    chg_d = pd.to_numeric(df["change_pct"], errors="coerce")

    valid_px = px_d > 0
    rel_diff = (px_zt - px_d).abs() / px_d.where(valid_px, pd.NA)
    price_mismatch_mask = (rel_diff > PRICE_REL_TOL) & valid_px

    chg_diff = (chg_zt - chg_d).abs()
    chg_band_mask_raw = chg_diff > CHG_ABS_TOL_PP

    # 除权日豁免：名字带 XD/DR/XR，或 daily 跳空 < -8% 而 zt > 8%
    name_col = "名称" if "名称" in df.columns else None
    if name_col is not None:
        ex_div_name = df[name_col].astype(str).str.contains(
            r"^(?:XD|DR|XR)", case=False, na=False, regex=True)
        is_st = df[name_col].astype(str).str.contains("ST", case=False, na=False)
    else:
        ex_div_name = pd.Series(False, index=df.index)
        is_st = pd.Series(False, index=df.index)
    ex_div_jump = (chg_d < -8.0) & (chg_zt > 8.0)
    ex_div_mask = ex_div_name | ex_div_jump
    chg_band_mask = chg_band_mask_raw & ~ex_div_mask

    df["_is_main"] = df["code"].apply(is_main_board)
    rule_main = df["_is_main"] & ~is_st
    rule_st = df["_is_main"] & is_st
    limit_up_rule_violation = (
        (rule_main & ~chg_zt.between(9.5, 10.5)) |
        (rule_st & ~chg_zt.between(4.5, 5.5))
    )

    n = int(valid_px.sum())
    n_pm = int(price_mismatch_mask.sum())
    n_cb_raw = int(chg_band_mask_raw.sum())
    n_cb = int(chg_band_mask.sum())
    n_ex = int(ex_div_mask.sum())
    n_rule = int(limit_up_rule_violation.sum())
    cb_rate = n_cb / n if n else 0.0

    summary = {
        "n_checked": n,
        "price_mismatch": n_pm,
        "price_mismatch_rate": round(n_pm / n, 4) if n else 0.0,
        "chg_out_of_band": n_cb,
        "chg_out_of_band_raw": n_cb_raw,
        "chg_out_of_band_rate": round(cb_rate, 4),
        "ex_dividend_excluded": n_ex,
        "limit_up_rule_violation": n_rule,
        "pass_price": (n == 0) or (n_pm / n <= THRESHOLD_PRICE_MISMATCH_MAX),
        "pass_chg_band": (n == 0) or (cb_rate <= THRESHOLD_CHG_OUT_OF_BAND_RATE),
        "pass_limit_up_rule": n_rule == 0,
        "price_rel_tol": PRICE_REL_TOL,
        "chg_abs_tol_pp": CHG_ABS_TOL_PP,
    }

    bad = df[price_mismatch_mask | chg_band_mask | limit_up_rule_violation].head(max_rows)
    if not bad.empty:
        keep_cols = [c for c in ["code", "date", "名称", "最新价", "close",
                                  "涨跌幅", "change_pct"] if c in bad.columns]
        bad = bad[keep_cols].copy()
        bad["date"] = pd.to_datetime(bad["date"]).dt.strftime("%Y-%m-%d")
    return summary, bad


def check_time_tier(zt_long: pd.DataFrame) -> Dict:
    if zt_long.empty or "首次封板时间" not in zt_long.columns:
        return {"n": 0, "tier_ok_ratio": 1.0, "pass": True,
                "threshold": THRESHOLD_TIER_OK_MIN, "tier_distribution": {}}
    ranks = zt_long["首次封板时间"].apply(time_tier_rank)
    n = len(ranks)
    n_ok = int((ranks <= 3).sum())
    ratio = n_ok / n if n else 1.0
    return {
        "n": n, "tier_ok": n_ok, "tier_ok_ratio": round(ratio, 4),
        "tier_distribution": ranks.value_counts().sort_index().to_dict(),
        "threshold": THRESHOLD_TIER_OK_MIN,
        "pass": ratio >= THRESHOLD_TIER_OK_MIN,
    }


def check_empty_runs(zt_cache_dir: Path) -> Dict:
    sub = Path(zt_cache_dir) / "zt_pool"
    if not sub.exists():
        return {"empty_runs": [], "max_run": 0, "warn": False,
                "threshold": EMPTY_RUN_THRESHOLD}
    files = sorted(sub.glob("*.parquet"))
    if not files:
        return {"empty_runs": [], "max_run": 0, "warn": False,
                "threshold": EMPTY_RUN_THRESHOLD}
    runs: List[Tuple[str, str, int]] = []
    cs: Optional[str] = None
    cl: Optional[str] = None
    cur_len = 0
    for f in files:
        is_empty = f.stat().st_size < 1000
        if is_empty:
            if cs is None:
                cs = f.stem
            cl = f.stem
            cur_len += 1
        else:
            if cur_len >= EMPTY_RUN_THRESHOLD and cs and cl:
                runs.append((cs, cl, cur_len))
            cs = cl = None
            cur_len = 0
    if cur_len >= EMPTY_RUN_THRESHOLD and cs and cl:
        runs.append((cs, cl, cur_len))
    return {
        "empty_runs": [{"start": s, "end": e, "len": n} for s, e, n in runs[:20]],
        "max_run": max((r[2] for r in runs), default=0),
        "threshold": EMPTY_RUN_THRESHOLD,
        "warn": bool(runs),
    }


def check_lookahead_self() -> Dict:
    try:
        from src.utils.validator import LookAheadValidator
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    try:
        LookAheadValidator.scan_feature_names(["f_ret_5", "f_pt_score"], raise_error=False)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main() -> int:
    args = parse_args()
    verify_dir = Path(args.verify_dir)
    verify_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = verify_dir / f"daily_consistency_{ts}.json"
    md_path = verify_dir / f"daily_consistency_{ts}.md"

    print("=" * 70)
    print("日线缓存一致性校验（阶段 1.4）")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"  daily_cache_dir: {args.daily_cache_dir}")
    print(f"  zt_cache_dir:    {args.zt_cache_dir}")
    print(f"  报告输出:        {json_path}")
    print("=" * 70)

    from src.data_fetch.base import DataFetcher

    class _DummyFetcher(DataFetcher):
        def load_stock_list(self, exchange="sh"):
            return pd.DataFrame(columns=["code", "market"])
        def load_batch(self, codes, start_date=None, end_date=None):
            return pd.DataFrame()

    mgr = DailyCacheManager(cache_dir=args.daily_cache_dir,
                            primary_fetcher=_DummyFetcher(),
                            fallback_fetcher=None)
    zt_loader = ZtPoolLoader(cache_dir=args.zt_cache_dir, require_akshare=False)

    print("\n[Step 1] 加载 zt_pool 长表 ...")
    zt_long = load_zt_pool_long(zt_loader, args.start, args.end, args.main_board_only)
    n_codes = zt_long['code'].nunique() if not zt_long.empty else 0
    print(f"  zt_pool 行数: {len(zt_long)} (codes={n_codes})")
    if args.exclude_today and not zt_long.empty:
        today = pd.Timestamp(datetime.now().date())
        before = len(zt_long)
        zt_long = zt_long[zt_long["date"] < today].reset_index(drop=True)
        n_today = before - len(zt_long)
        if n_today:
            print(f"  排除今日 ({today.date()}) {n_today} 行 (TDX 盘后下载前会假阳性)")

    print("\n[Step 2] 覆盖度校验 ...")
    cov_summary, miss_df, matched_df = check_coverage(zt_long, mgr)
    print(f"  total={cov_summary['total_rows']} missing={cov_summary['missing']} "
          f"coverage={cov_summary['coverage']:.4f} -> "
          f"{'PASS' if cov_summary['pass'] else 'FAIL'} (>={THRESHOLD_COVERAGE_MIN:.2%})")
    if not miss_df.empty:
        miss_path = verify_dir / f"missing_daily_{ts}.csv"
        miss_df.head(args.max_mismatch_rows).to_csv(
            miss_path, index=False, encoding="utf-8-sig")
        print(f"  缺失明细前 {min(len(miss_df), args.max_mismatch_rows)} 条: {miss_path}")

    print("\n[Step 3] 价格 / 涨跌幅一致性 ...")
    price_summary, bad_df = check_price_consistency(
        matched_df, max_rows=args.max_mismatch_rows)
    print(f"  n_checked={price_summary['n_checked']} "
          f"price_mismatch={price_summary['price_mismatch']} "
          f"({price_summary['price_mismatch_rate']:.2%}) -> "
          f"{'PASS' if price_summary['pass_price'] else 'FAIL'} "
          f"(<={THRESHOLD_PRICE_MISMATCH_MAX:.2%})")
    print(f"  chg_out_of_band={price_summary['chg_out_of_band']} "
          f"(原始 {price_summary['chg_out_of_band_raw']}, "
          f"除权日豁免 {price_summary['ex_dividend_excluded']}) -> "
          f"{'PASS' if price_summary['pass_chg_band'] else 'FAIL'} "
          f"(<={THRESHOLD_CHG_OUT_OF_BAND_RATE:.1%})")
    print(f"  limit_up_rule_violation={price_summary['limit_up_rule_violation']} -> "
          f"{'PASS' if price_summary['pass_limit_up_rule'] else 'FAIL'} (=0)")
    if not bad_df.empty:
        bad_path = verify_dir / f"mismatch_{ts}.csv"
        bad_df.to_csv(bad_path, index=False, encoding="utf-8-sig")
        print(f"  不一致明细前 {len(bad_df)} 条: {bad_path}")

    print("\n[Step 4] 首次封板时间分布 ...")
    tier_summary = check_time_tier(zt_long)
    print(f"  tier_ok_ratio={tier_summary['tier_ok_ratio']:.2%} -> "
          f"{'PASS' if tier_summary['pass'] else 'FAIL'} "
          f"(>={THRESHOLD_TIER_OK_MIN:.2%})")
    print(f"  tier 分布: {tier_summary.get('tier_distribution')}")

    print("\n[Step 5] zt_pool 空文件连续段 ...")
    empty_summary = check_empty_runs(Path(args.zt_cache_dir))
    if empty_summary["warn"]:
        print(f"  发现 {len(empty_summary['empty_runs'])} 段 >={EMPTY_RUN_THRESHOLD} 个 "
              f"连续空文件 (最长 {empty_summary['max_run']})")
        for r in empty_summary["empty_runs"]:
            print(f"    {r['start']} ~ {r['end']} ({r['len']} 天)")
    else:
        print("  无异常连续空段")

    print("\n[Step 6] 未来函数自检 ...")
    la = check_lookahead_self()
    print(f"  scan_feature_names: {'OK' if la.get('ok') else 'FAIL'}")
    if not la.get("ok"):
        print(f"    error: {la.get('error')}")

    overall_pass = (
        cov_summary["pass"] and price_summary["pass_price"]
        and price_summary["pass_chg_band"]
        and price_summary["pass_limit_up_rule"]
        and tier_summary["pass"] and la.get("ok", False)
    )

    payload = {
        "args": vars(args),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "coverage": cov_summary,
        "price_consistency": price_summary,
        "time_tier": tier_summary,
        "empty_runs": empty_summary,
        "lookahead": la,
        "overall_pass": overall_pass,
        "daily_cache_summary": mgr.cache_summary(),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    md = [
        f"# 日线缓存一致性报告 ({ts})", "",
        f"- 区间: `{args.start} ~ {args.end}`",
        f"- daily_cache_dir: `{args.daily_cache_dir}`",
        f"- zt_cache_dir: `{args.zt_cache_dir}`",
        f"- **总体结论: {'PASS' if overall_pass else 'FAIL'}**", "",
        "## 覆盖度",
        (f"- total_rows={cov_summary['total_rows']}, "
         f"missing={cov_summary['missing']}, "
         f"coverage={cov_summary['coverage']:.4f} -> "
         f"{'PASS' if cov_summary['pass'] else 'FAIL'}"),
        "", "## 价格/涨跌幅",
        f"- n_checked={price_summary['n_checked']}",
        (f"- 价格不一致={price_summary['price_mismatch']} "
         f"({price_summary['price_mismatch_rate']:.2%}) -> "
         f"{'PASS' if price_summary['pass_price'] else 'FAIL'}"),
        (f"- 涨跌幅越界={price_summary['chg_out_of_band']} "
         f"(原始 {price_summary['chg_out_of_band_raw']}, "
         f"除权日豁免 {price_summary['ex_dividend_excluded']}) -> "
         f"{'PASS' if price_summary['pass_chg_band'] else 'FAIL'}"),
        (f"- 涨停规则违反={price_summary['limit_up_rule_violation']} -> "
         f"{'PASS' if price_summary['pass_limit_up_rule'] else 'FAIL'}"),
        "", "## 首次封板时间分布",
        (f"- Tier1-3 占比={tier_summary['tier_ok_ratio']:.2%} -> "
         f"{'PASS' if tier_summary['pass'] else 'FAIL'}"),
        f"- 分布: {tier_summary.get('tier_distribution')}",
        "", "## zt_pool 空文件连续段",
        (f"- 异常段数: {len(empty_summary['empty_runs'])}, "
         f"最长 {empty_summary['max_run']} 天 (仅 WARN)"),
        "", "## 未来函数自检",
        f"- scan_feature_names: {'OK' if la.get('ok') else 'FAIL'}",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"总体结论: {'PASS' if overall_pass else 'FAIL'}")
    print(f"JSON 报告: {json_path}")
    print(f"MD   报告: {md_path}")
    print("=" * 70)

    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
