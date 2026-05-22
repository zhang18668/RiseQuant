"""Build zt_pool_cache from Tushare/Tinyshare historical limit-list data.

This fills the same cache layout that AkShare uses:

data/zt_pool_cache/
  zt_pool/YYYYMMDD.parquet
  zt_pool_previous/YYYYMMDD.parquet

Token is read from ``TINYSHARE_TOKEN`` first, then ``TUSHARE_TOKEN``/``TS_TOKEN``.
Do not hardcode tokens in this file.

Concurrency
-----------
The script can run with multiple worker threads (``--workers``). The shared
``TushareLimitPoolClient`` carries a thread-safe sliding-window rate limiter
(default 120 calls/min), so concurrent workers cannot exceed the provider
quota. This is the practical way to hide HTTP RTT behind the rate-limit
ceiling, bringing total wall time close to the theoretical floor of
``total_calls / rate_limit_per_min`` minutes.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.tushare_limit_pool import (
    TushareLimitPoolClient,
    normalize_limit_list_to_zt_pool,
    normalize_previous_pool,
)
from src.data.zt_pool_loader import (
    ZT_POOL_TYPE,
    ZT_PREV_TYPE,
    assert_zt_pool_cache_quality,
)


DEFAULT_CACHE_DIR = ROOT / "data" / "zt_pool_cache"


def parse_args():
    p = argparse.ArgumentParser(description="Build zt_pool cache from tinyshare/tushare")
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--pool", choices=["both", ZT_POOL_TYPE, ZT_PREV_TYPE], default="both")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument(
        "--workers", type=int, default=4,
        help="并发线程数（默认 4）。共享 rate-limit，不会超过 --rate-limit-per-min。",
    )
    p.add_argument(
        "--sleep", type=float, default=0.0,
        help="每日循环间的额外 sleep（秒）。默认 0，限速由 --rate-limit-per-min 接管。",
    )
    p.add_argument(
        "--rate-limit-per-min", type=int, default=120,
        help="tinyshare/tushare 每分钟最大 API 调用次数（默认 120）",
    )
    p.add_argument("--assert-monthly-non-empty", action="store_true")
    p.add_argument("--min-monthly-non-empty", type=int, default=18)
    return p.parse_args()


def cache_path(cache_dir: Path, pool_type: str, date: str) -> Path:
    path = cache_dir / pool_type / f"{date}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_if_needed(path: Path, df: pd.DataFrame, force_refresh: bool) -> str:
    if path.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(path)
            if not cached.empty:
                return "cached"
        except Exception:  # noqa: BLE001
            pass
    if df is None or df.empty:
        # still touch the file to mark "fetched and empty" so subsequent
        # runs do not waste an API call. parquet for empty frame still
        # records column schema if any, but here we drop a 0-row marker.
        empty = pd.DataFrame()
        empty.to_parquet(path, index=False)
        return "empty_skip"
    df.to_parquet(path, index=False)
    return "written"


_cache_lock = threading.Lock()


def get_or_fetch_limit(
    client: TushareLimitPoolClient,
    date: str,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Thread-safe get-or-fetch for raw ``limit_list_d`` results.

    Note: in rare race cases two threads may both miss the cache and both
    fetch the same date — that's at most one wasted call per boundary date
    and the rate limiter still throttles correctly.
    """
    with _cache_lock:
        cached = cache.get(date)
    if cached is not None:
        return cached
    raw = client.limit_list_d(date, limit_type="U")
    with _cache_lock:
        existing = cache.get(date)
        if existing is None:
            cache[date] = raw
            return raw
        return existing


def process_one_day(
    client: TushareLimitPoolClient,
    date: str,
    prev_date: str | None,
    pools: tuple[str, ...],
    cache_dir: Path,
    force_refresh: bool,
    limit_cache: dict[str, pd.DataFrame],
    extra_sleep: float,
) -> tuple[str, dict[str, str]]:
    statuses: dict[str, str] = {}
    try:
        if ZT_POOL_TYPE in pools:
            raw = get_or_fetch_limit(client, date, limit_cache)
            zt = normalize_limit_list_to_zt_pool(raw)
            statuses[ZT_POOL_TYPE] = write_if_needed(
                cache_path(cache_dir, ZT_POOL_TYPE, date), zt, force_refresh
            )

        if ZT_PREV_TYPE in pools:
            if prev_date is None:
                statuses[ZT_PREV_TYPE] = "empty_skip"
            else:
                prev_raw = get_or_fetch_limit(client, prev_date, limit_cache)
                daily = client.daily(date)
                daily_basic = client.daily_basic(date)
                prev_pool = normalize_previous_pool(date, prev_raw, daily, daily_basic)
                statuses[ZT_PREV_TYPE] = write_if_needed(
                    cache_path(cache_dir, ZT_PREV_TYPE, date), prev_pool, force_refresh
                )

        if extra_sleep > 0:
            time.sleep(extra_sleep)
    except Exception as exc:  # noqa: BLE001
        for pool in pools:
            statuses.setdefault(pool, f"error:{type(exc).__name__}:{exc}")
    return date, statuses


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    client = TushareLimitPoolClient(
        prefer_tinyshare=True,
        rate_limit_per_min=args.rate_limit_per_min,
    )
    print(
        f"rate_limit={args.rate_limit_per_min}/min "
        f"workers={args.workers} sleep_per_day={args.sleep}s"
    )

    dates = client.trade_dates(args.start, args.end)
    pools = (ZT_POOL_TYPE, ZT_PREV_TYPE) if args.pool == "both" else (args.pool,)
    print(f"dates={len(dates)} range={args.start}~{args.end} pools={pools}")

    # Theoretical lower bound — useful for ETA visibility.
    calls_per_day = (1 if ZT_POOL_TYPE in pools else 0) + (3 if ZT_PREV_TYPE in pools else 0)
    # Note: when both pools selected, limit_list_d may be shared across days,
    # so total calls per day is ~3 (limit_list_d + daily + daily_basic).
    if ZT_POOL_TYPE in pools and ZT_PREV_TYPE in pools:
        calls_per_day = 3
    est_calls = len(dates) * calls_per_day
    est_minutes = est_calls / max(args.rate_limit_per_min, 1)
    print(f"estimated calls={est_calls}, lower-bound wall time ~{est_minutes:.1f} min")

    stats: dict[str, dict[str, int]] = {pool: defaultdict(int) for pool in pools}
    limit_cache: dict[str, pd.DataFrame] = {}

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = []
        for i, date in enumerate(dates):
            prev_date = dates[i - 1] if i > 0 else None
            futures.append(
                executor.submit(
                    process_one_day,
                    client,
                    date,
                    prev_date,
                    pools,
                    cache_dir,
                    args.force_refresh,
                    limit_cache,
                    args.sleep,
                )
            )

        done = 0
        for fut in as_completed(futures):
            date, statuses = fut.result()
            for pool, status in statuses.items():
                if status.startswith("error"):
                    stats[pool]["error"] += 1
                else:
                    stats[pool][status] += 1
            done += 1
            if done % 50 == 0 or done == len(dates):
                elapsed = time.monotonic() - t_start
                eta = (elapsed / done) * (len(dates) - done) if done > 0 else 0
                summary = {p: dict(s) for p, s in stats.items()}
                print(
                    f"{done}/{len(dates)} elapsed={elapsed/60:.1f}m "
                    f"eta={eta/60:.1f}m stats={summary}"
                )

    elapsed = time.monotonic() - t_start
    final = {p: dict(s) for p, s in stats.items()}
    print(f"done in {elapsed/60:.1f}m  stats={final}")

    if args.assert_monthly_non_empty:
        for pool in pools:
            quality = assert_zt_pool_cache_quality(
                cache_dir,
                args.start,
                args.end,
                pool_type=pool,
                min_monthly_non_empty=args.min_monthly_non_empty,
            )
            print(f"[{pool}] quality ok non_empty={quality['non_empty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
