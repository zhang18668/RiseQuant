"""build_daily_cache.py — 日线数据本地缓存构建脚本（阶段 1.3）

用途
----
基于已经构建好的涨停板池缓存（``ZtPoolLoader``），抽出"曾出现过涨停"的
主板代码作为 universe，再用 TDX 本地 ``.day`` 文件（或 AkShare 兜底）补齐
[start, end] 区间内的日线 OHLCV，落地到 ``<cache_dir>/<code>.parquet``。

约定
----
- 默认 universe 为 ``zt_pool_relevant``：``data/zt_pool_cache`` 中两个池的
  代码并集（仅主板）。
- 默认数据源为 TDX。若 TDX 安装目录不存在，会自动回退到 AkShare。
- 增量优先：已缓存的最新日期之后才发起新请求；``--force-refresh`` 才整覆盖。

策略相关
--------
对应《策略规划》阶段 1.3。完成后运行 ``scripts/verify_daily_cache.py``
执行阶段 1.4 的数据一致性校验。

运行示例
--------
首次构建（仅 zt_pool_relevant，TDX 本地源）::

    cd G:/AI/RiseQuant/limit_up_project
    python scripts/build_daily_cache.py

指定区间 / 数据源 / universe::

    python scripts/build_daily_cache.py --start 20200101 --end 20260521 \\
        --universe zt_pool_relevant --source tdx --tdx-path C:/new_tdx/vipdoc

全市场主板（更耗时也更占空间）::

    python scripts/build_daily_cache.py --universe all --source tdx
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# 把项目根加到 sys.path
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.data.daily_cache import (  # noqa: E402
    DailyCacheManager,
    CacheReport,
)
from src.data.zt_pool_loader import ZtPoolLoader  # noqa: E402
from src.data_fetch.base import DataFetcher  # noqa: E402


DEFAULT_START = "20200101"
DEFAULT_END = datetime.now().strftime("%Y%m%d")
DEFAULT_DAILY_CACHE_DIR = _PROJ_ROOT / "data" / "daily_cache"
DEFAULT_ZT_CACHE_DIR = _PROJ_ROOT / "data" / "zt_pool_cache"
DEFAULT_TDX_PATH = r"C:\new_tdx\vipdoc"
DEFAULT_LOG_DIR = _PROJ_ROOT / "data" / "verify"


# ---------------------------- argparse ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="构建/增量更新日线本地 Parquet 缓存（阶段 1.3）"
    )
    p.add_argument("--start", default=DEFAULT_START, help=f"起始日期 YYYYMMDD（默认 {DEFAULT_START}）")
    p.add_argument("--end", default=DEFAULT_END, help=f"结束日期 YYYYMMDD（默认今天 {DEFAULT_END}）")
    p.add_argument(
        "--universe",
        choices=["zt_pool_relevant", "all"],
        default="zt_pool_relevant",
        help="股票池：zt_pool_relevant（默认，仅涨停池出现过的主板）/ all（全市场主板）",
    )
    p.add_argument("--source", choices=["tdx", "akshare", "auto"], default="auto",
                   help="数据源：tdx 本地 / akshare 在线 / auto（TDX 优先，回退 AkShare）")
    p.add_argument("--tdx-path", default=DEFAULT_TDX_PATH,
                   help=f"通达信 vipdoc 目录（默认 {DEFAULT_TDX_PATH}）")
    p.add_argument("--cache-dir", default=str(DEFAULT_DAILY_CACHE_DIR),
                   help=f"日线缓存目录（默认 {DEFAULT_DAILY_CACHE_DIR}）")
    p.add_argument("--zt-cache-dir", default=str(DEFAULT_ZT_CACHE_DIR),
                   help=f"涨停池缓存目录（默认 {DEFAULT_ZT_CACHE_DIR}）")
    p.add_argument("--force-refresh", action="store_true",
                   help="强制重新拉取（忽略已有缓存）")
    p.add_argument("--no-fallback", action="store_true",
                   help="禁用主源失败后的兜底切换")
    p.add_argument("--max-codes", type=int, default=0,
                   help="仅处理前 N 只（0 表示全部，调试用）")
    p.add_argument("--report-json",
                   default=str(DEFAULT_LOG_DIR / f"build_daily_cache_{datetime.now():%Y%m%d_%H%M%S}.json"),
                   help="批次报告 JSON 输出路径")
    return p.parse_args()


# ---------------------------- fetcher 构造 ----------------------------
def build_fetchers(
    source: str,
    tdx_path: str,
    no_fallback: bool,
) -> tuple[DataFetcher, Optional[DataFetcher]]:
    """根据 --source 决定 primary 和 fallback。失败时给出明确报错。"""

    def _try_tdx() -> Optional[DataFetcher]:
        try:
            from src.data_fetch.tdx_fetcher import TDXFetcher
            f = TDXFetcher(tdx_path)
            if not f.data_path or not Path(f.data_path).exists():
                print(f"[WARN] TDX 路径不存在: {tdx_path}")
                return None
            return f
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] TDX fetcher 不可用: {exc}")
            return None

    def _try_akshare() -> Optional[DataFetcher]:
        try:
            from src.data_fetch.akshare_fetcher import AkshareFetcher
            return AkshareFetcher()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] AkShare fetcher 不可用: {exc}")
            return None

    primary: Optional[DataFetcher] = None
    fallback: Optional[DataFetcher] = None

    if source == "tdx":
        primary = _try_tdx()
        if primary is None:
            raise RuntimeError("source=tdx 但 TDX fetcher 构造失败")
        if not no_fallback:
            fallback = _try_akshare()
    elif source == "akshare":
        primary = _try_akshare()
        if primary is None:
            raise RuntimeError("source=akshare 但 AkShare fetcher 构造失败")
        # akshare 模式默认不再 fallback 到 tdx，避免反复试
    else:  # auto
        primary = _try_tdx() or _try_akshare()
        if primary is None:
            raise RuntimeError("source=auto 但 TDX 与 AkShare 都不可用")
        # 如果 primary 是 tdx，再挂 akshare 兜底
        if not no_fallback and primary.__class__.__name__ == "TDXFetcher":
            fallback = _try_akshare()

    return primary, fallback


# ---------------------------- universe ----------------------------
def resolve_universe(
    args: argparse.Namespace,
    zt_loader: ZtPoolLoader,
    primary: DataFetcher,
) -> List[str]:
    if args.universe == "zt_pool_relevant":
        codes = DailyCacheManager.universe_from_zt_pool(
            zt_loader=zt_loader,
            start_date=args.start,
            end_date=args.end,
            main_board_only=True,
        )
    else:  # all
        from src.data.tdx_loader import TDXDataLoader
        codes_set = set()
        try:
            tdx_loader = TDXDataLoader(args.tdx_path)
            for ex in ("sh", "sz"):
                lst = tdx_loader.load_stock_list(ex)
                if lst is None or lst.empty:
                    continue
                for raw in lst["code"]:
                    c = str(raw).zfill(6)
                    if tdx_loader.is_main_board(c):
                        codes_set.add(c)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 全市场 universe 提取失败：{exc}；退回 zt_pool_relevant")
            return DailyCacheManager.universe_from_zt_pool(
                zt_loader, args.start, args.end, main_board_only=True
            )
        codes = sorted(codes_set)
    if args.max_codes > 0:
        codes = codes[: args.max_codes]
    return codes


# ---------------------------- 打印工具 ----------------------------
def fmt_summary(s: dict) -> str:
    if s.get("count", 0) == 0:
        return f"  [daily_cache] 空 (dir={s.get('cache_dir')})"
    return (
        f"  [daily_cache] {s['count']} 个文件 "
        f"(非空={s['non_empty']} 空={s['empty']}) | "
        f"{s.get('earliest_date')}~{s.get('latest_date')} | "
        f"{s['total_size_mb']} MB"
    )


def summarize_reports(reports: List[CacheReport]) -> dict:
    n = len(reports)
    by_status: dict = {}
    n_added = 0
    elapsed = 0.0
    by_source: dict = {}
    errors: List[dict] = []
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        n_added += r.rows_added
        elapsed += r.elapsed_sec
        if r.source:
            by_source[r.source] = by_source.get(r.source, 0) + 1
        if r.status == "error":
            errors.append({"code": r.code, "error": r.error})
    return {
        "total": n,
        "by_status": by_status,
        "rows_added": n_added,
        "elapsed_sec_sum": round(elapsed, 1),
        "by_source": by_source,
        "errors_head": errors[:20],
        "errors_total": len(errors),
    }


# ---------------------------- 主流程 ----------------------------
def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("日线数据缓存构建（阶段 1.3）")
    print(f"  起始: {args.start}")
    print(f"  结束: {args.end}")
    print(f"  universe: {args.universe}")
    print(f"  source:   {args.source}")
    print(f"  cache_dir:{args.cache_dir}")
    print(f"  强制刷新: {args.force_refresh}")
    print(f"  no_fallback: {args.no_fallback}")
    print("=" * 70)

    # 1) 构造 fetcher
    try:
        primary, fallback = build_fetchers(args.source, args.tdx_path, args.no_fallback)
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] fetcher 构造失败: {exc}")
        return 2

    print(f"\n primary  = {type(primary).__name__}")
    print(f" fallback = {type(fallback).__name__ if fallback else 'None'}")

    # 2) zt_pool loader（不要求联网，仅读已有缓存）
    zt_loader = ZtPoolLoader(cache_dir=args.zt_cache_dir, require_akshare=False)

    # 3) universe
    print("\n抽取 universe ...")
    codes = resolve_universe(args, zt_loader, primary)
    if not codes:
        print("[FATAL] universe 为空。请先运行 build_zt_pool_cache.py，或换 --universe all")
        return 2
    print(f"  共 {len(codes)} 只股票")
    print(f"  前 5: {codes[:5]}")
    print(f"  末 5: {codes[-5:]}")

    # 4) 缓存管理器
    mgr = DailyCacheManager(
        cache_dir=args.cache_dir,
        primary_fetcher=primary,
        fallback_fetcher=fallback,
    )
    print("\n构建前缓存状态：")
    print(fmt_summary(mgr.cache_summary()))

    # 5) 执行
    start_iso = pd.Timestamp(args.start).strftime("%Y-%m-%d")
    end_iso = pd.Timestamp(args.end).strftime("%Y-%m-%d")

    t0 = datetime.now()
    try:
        reports = mgr.update_batch(
            codes=codes,
            start_date=start_iso,
            end_date=end_iso,
            force_refresh=args.force_refresh,
            allow_fallback=not args.no_fallback,
            progress_every=max(20, len(codes) // 50),
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 已写入的 parquet 保留")
        return 130
    elapsed = (datetime.now() - t0).total_seconds()

    # 6) 总结 & 报告
    print("\n" + "=" * 70)
    print(f"完成。本次耗时 {elapsed:.1f} 秒")
    summary = summarize_reports(reports)
    print(f"  by_status: {summary['by_status']}")
    print(f"  by_source: {summary['by_source']}")
    print(f"  rows_added: {summary['rows_added']}")
    print(f"  errors_total: {summary['errors_total']}")
    if summary["errors_total"]:
        print("  errors_head:")
        for e in summary["errors_head"]:
            print(f"    {e['code']}: {e['error']}")
    print("\n构建后缓存状态：")
    print(fmt_summary(mgr.cache_summary()))

    # 报告 JSON
    try:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args),
            "summary": summary,
            "cache_after": mgr.cache_summary(),
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n报告已保存：{report_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 报告写入失败: {exc}")

    print("\n下一步：进入阶段 1.4 → python scripts/verify_daily_cache.py")
    return 0 if summary["errors_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
