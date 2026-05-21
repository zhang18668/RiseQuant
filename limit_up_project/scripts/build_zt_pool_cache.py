"""
build_zt_pool_cache.py — 涨停板池数据缓存构建脚本

用途
----
首次运行时从 2020-01-01 到今日，拉取并落地全部涨停板池数据到本地 Parquet 缓存。
后续每日运行只会增量拉取新交易日（已缓存的会自动跳过）。

支持两个数据池：
- ``zt_pool``          : 当日涨停股池
- ``zt_pool_previous`` : 昨日涨停股池

策略相关
--------
对应《策略规划.md》阶段 1.2。运行成功后即可进入阶段 1.3（日线数据回溯）
和阶段 1.4（数据一致性校验）。

运行
----
首次构建（耗时约 10-20 分钟）::

    cd G:/AI/RiseQuant/limit_up_project
    python scripts/build_zt_pool_cache.py

仅增量更新到今天（每日运行）::

    python scripts/build_zt_pool_cache.py --start 20260520

自定义区间::

    python scripts/build_zt_pool_cache.py --start 20230101 --end 20231231

仅拉一个池子::

    python scripts/build_zt_pool_cache.py --pool zt_pool
    python scripts/build_zt_pool_cache.py --pool zt_pool_previous

依赖
----
需要 ``akshare`` 与 ``pyarrow`` 已安装::

    pip install -U akshare pyarrow
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 把项目根加到 sys.path，方便直接 ``python scripts/xxx.py`` 调用
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.data.zt_pool_loader import (  # noqa: E402
    ZtPoolLoader,
    ZT_POOL_TYPE,
    ZT_PREV_TYPE,
    SUPPORTED_TYPES,
)


DEFAULT_START = "20200101"
DEFAULT_CACHE_DIR = _PROJ_ROOT / "data" / "zt_pool_cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="构建/增量更新涨停板池本地 Parquet 缓存"
    )
    p.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help=f"起始日期 YYYYMMDD（默认 {DEFAULT_START}）"
    )
    p.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y%m%d"),
        help="结束日期 YYYYMMDD（默认今天）"
    )
    p.add_argument(
        "--pool", type=str, choices=["both", *SUPPORTED_TYPES], default="both",
        help="数据池类型：both（默认）/ zt_pool / zt_pool_previous"
    )
    p.add_argument(
        "--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR),
        help=f"缓存根目录（默认 {DEFAULT_CACHE_DIR}）"
    )
    p.add_argument(
        "--request-interval", type=float, default=0.2,
        help="两次请求间最小间隔秒（默认 0.2）"
    )
    p.add_argument(
        "--max-retries", type=int, default=3,
        help="单次请求最大重试次数（默认 3）"
    )
    p.add_argument(
        "--force-refresh", action="store_true",
        help="强制重新拉取（忽略已有缓存）"
    )
    return p.parse_args()


def fmt_summary(s: dict) -> str:
    if s.get("count", 0) == 0:
        return f"  [{s['pool_type']}] 缓存为空"
    return (
        f"  [{s['pool_type']}] {s['count']} 个文件 | "
        f"{s['earliest']}~{s['latest']} | "
        f"{s['total_size_mb']} MB"
    )


def run_one(loader: ZtPoolLoader, pool_type: str, start: str, end: str,
            force: bool) -> dict:
    """拉取单个池子的区间数据，返回统计信息。"""
    print("\n" + "=" * 70)
    print(f"开始拉取池: {pool_type}")
    print(f"区间: {start} ~ {end}")
    print(f"模式: {'强制刷新' if force else '增量（跳过已缓存）'}")
    print("=" * 70)

    t0 = datetime.now()
    df = loader.fetch_range(
        start_date=start,
        end_date=end,
        pool_type=pool_type,
        use_cache=not force,
        save_cache=True,
        skip_errors=True,
        progress_every=50,
    )
    elapsed = (datetime.now() - t0).total_seconds()

    summary = loader.cache_summary(pool_type)
    print(f"\n本次拉取得到 {len(df)} 条记录，耗时 {elapsed:.1f} 秒")
    print(fmt_summary(summary))
    return summary


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("涨停板池缓存构建")
    print(f"  起始: {args.start}")
    print(f"  结束: {args.end}")
    print(f"  池子: {args.pool}")
    print(f"  缓存: {args.cache_dir}")
    print(f"  强制刷新: {args.force_refresh}")
    print("=" * 70)

    loader = ZtPoolLoader(
        cache_dir=args.cache_dir,
        require_akshare=True,
        max_retries=args.max_retries,
        request_interval=args.request_interval,
    )

    # 先打印现有缓存
    print("\n构建前缓存状态：")
    for t in SUPPORTED_TYPES:
        print(fmt_summary(loader.cache_summary(t)))

    # 执行
    types_to_run = SUPPORTED_TYPES if args.pool == "both" else (args.pool,)
    for t in types_to_run:
        try:
            run_one(loader, t, args.start, args.end, args.force_refresh)
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] 在 {t} 拉取过程中被中断，已缓存的数据保留")
            return 130
        except Exception as e:  # noqa: BLE001
            print(f"\n[ERROR] 池 {t} 拉取失败: {type(e).__name__}: {e}")
            return 2

    # 最后再次打印缓存状态
    print("\n" + "=" * 70)
    print("构建完成。最终缓存状态：")
    print("=" * 70)
    for t in SUPPORTED_TYPES:
        print(fmt_summary(loader.cache_summary(t)))

    print("\n下一步：进入阶段 1.3（日线数据回溯）+ 阶段 1.4（数据一致性校验）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
