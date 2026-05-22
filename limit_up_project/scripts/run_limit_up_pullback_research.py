"""Run the limit-up pullback research sample export.

Example:
    python scripts/run_limit_up_pullback_research.py --source cache --start 2022-01-01 --end 2026-05-21
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.pipeline_loader import load_daily_data
from src.data.zt_pool_loader import (
    ZT_POOL_TYPE,
    ZT_PREV_TYPE,
    ZtPoolLoader,
    assert_zt_pool_cache_quality,
)
from src.feature.market_features import market_proxy_from_daily
from src.strategy.limit_up_pullback import (
    LimitUpPullbackParams,
    build_trade_ledger,
    factor_group_stats,
    merge_zt_pool_fields,
    select_limit_up_pullback,
)
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger

setup_logger(log_level="INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Export limit-up pullback research samples")
    p.add_argument("--source", choices=["cache", "tdx", "auto"], default="cache")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", default="./models/rule_limit_up_pullback_research")
    p.add_argument("--market-source", choices=["akshare", "proxy", "none"], default="proxy")
    p.add_argument("--market-index", default="000001", help="AkShare index code, default 000001 Shanghai Composite")
    p.add_argument("--ignore-market-filter", action="store_true")
    p.add_argument("--ignore-turnover-rate", action="store_true")
    p.add_argument("--ignore-zt-pool", action="store_true")
    p.add_argument("--require-zt-pool-non-empty", action="store_true")
    p.add_argument("--min-zt-pool-non-empty", type=int, default=200)
    p.add_argument("--assert-monthly-zt-pool", action="store_true")
    p.add_argument("--min-monthly-zt-pool-non-empty", type=int, default=18)
    p.add_argument("--wash-turnover-tier-include", default="")
    p.add_argument("--wash-turnover-tier-exclude", default="")
    p.add_argument("--limit-day-vol-ratio5-max", type=float, default=0.0)
    p.add_argument("--limit-return-min", type=float, default=0.095)
    p.add_argument("--confirm-return-max", type=float, default=0.05)
    p.add_argument("--confirm-high-return-max", type=float, default=0.08)
    p.add_argument("--next-high-target", type=float, default=0.03)
    return p.parse_args()


def patch_config_dates(config, start: str, end: str):
    data = config.get_section("data")
    data["start_date"] = start
    data["end_date"] = end
    return config


def _date8(raw: str) -> str:
    return pd.Timestamp(raw).strftime("%Y%m%d")


def load_zt_pool_from_cache(config, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_config = config.get_section("data")
    zt_dir = Path(data_config.get("zt_pool_cache_dir", "./data/zt_pool_cache"))
    if not zt_dir.is_absolute():
        zt_dir = ROOT / zt_dir
    loader = ZtPoolLoader(cache_dir=zt_dir, require_akshare=False)
    start8 = _date8(start)
    end8 = _date8(end)
    zt = loader.load_cached_range(start8, end8, pool_type=ZT_POOL_TYPE)
    zt_prev = loader.load_cached_range(start8, end8, pool_type=ZT_PREV_TYPE)
    logger.info(f"zt_pool cache loaded: limit={len(zt)} rows previous={len(zt_prev)} rows")
    return zt, zt_prev


def validate_zt_pool_cache(args, config) -> None:
    data_config = config.get_section("data")
    zt_dir = Path(data_config.get("zt_pool_cache_dir", "./data/zt_pool_cache"))
    if not zt_dir.is_absolute():
        zt_dir = ROOT / zt_dir
    loader = ZtPoolLoader(cache_dir=zt_dir, require_akshare=False)
    checks = [
        (ZT_POOL_TYPE, loader.cache_summary(ZT_POOL_TYPE)),
        (ZT_PREV_TYPE, loader.cache_summary(ZT_PREV_TYPE)),
    ]
    for pool_type, summary in checks:
        non_empty = 0
        sub = zt_dir / pool_type
        for path in sub.glob("*.parquet"):
            try:
                if not pd.read_parquet(path).empty:
                    non_empty += 1
            except Exception:  # noqa: BLE001
                continue
        if non_empty < args.min_zt_pool_non_empty:
            raise RuntimeError(
                f"{pool_type} cache has only {non_empty} non-empty files "
                f"(total={summary.get('count', 0)}), below --min-zt-pool-non-empty "
                f"{args.min_zt_pool_non_empty}. Run build_zt_pool_cache.py "
                "--refresh-empty-only --assert-monthly-non-empty first."
            )
    if args.assert_monthly_zt_pool:
        for pool_type, _summary in checks:
            assert_zt_pool_cache_quality(
                cache_dir=zt_dir,
                start_date=_date8(args.start),
                end_date=_date8(args.end),
                pool_type=pool_type,
                min_monthly_non_empty=args.min_monthly_zt_pool_non_empty,
            )


def apply_post_filters(samples: pd.DataFrame, args) -> pd.DataFrame:
    out = samples.copy()
    if out.empty:
        return out
    include = {x.strip() for x in args.wash_turnover_tier_include.split(",") if x.strip()}
    exclude = {x.strip() for x in args.wash_turnover_tier_exclude.split(",") if x.strip()}
    if include and "wash_turnover_tier" in out.columns:
        out = out[out["wash_turnover_tier"].astype(str).isin(include)].copy()
    if exclude and "wash_turnover_tier" in out.columns:
        out = out[~out["wash_turnover_tier"].astype(str).isin(exclude)].copy()
    if args.limit_day_vol_ratio5_max and "limit_day_vol_ratio_5" in out.columns:
        out = out[pd.to_numeric(out["limit_day_vol_ratio_5"], errors="coerce") <= args.limit_day_vol_ratio5_max].copy()
    return out


def load_market_index_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("akshare is required for --market-source akshare") from exc

    raw = ak.index_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=_date8(start),
        end_date=_date8(end),
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"AkShare returned no index rows for {symbol} {start}~{end}")

    rename = {
        "日期": "date",
        "收盘": "close",
        "成交量": "volume",
    }
    out = raw.rename(columns=rename).copy()
    required = {"date", "close"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"AkShare index rows missing columns: {sorted(missing)}")
    out["date"] = pd.to_datetime(out["date"])
    if "volume" not in out.columns:
        out["volume"] = pd.NA
    return out[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)


def load_market(args, daily: pd.DataFrame) -> pd.DataFrame | None:
    if args.ignore_market_filter or args.market_source == "none":
        return None
    if args.market_source == "proxy":
        logger.info("using market proxy from daily cache")
        return market_proxy_from_daily(daily)
    logger.info(f"fetching AkShare index {args.market_index}")
    try:
        return load_market_index_akshare(args.market_index, args.start, args.end)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"AkShare index fetch failed, falling back to market proxy: {exc}")
        return market_proxy_from_daily(daily)


def main() -> int:
    args = parse_args()
    config = patch_config_dates(get_config(), args.start, args.end)
    if args.require_zt_pool_non_empty:
        validate_zt_pool_cache(args, config)
    daily = load_daily_data(config, source=args.source)
    daily["date"] = pd.to_datetime(daily["date"])

    zt_pool = zt_prev = None
    if not args.ignore_zt_pool:
        zt_pool, zt_prev = load_zt_pool_from_cache(config, args.start, args.end)
        daily = merge_zt_pool_fields(daily, zt_pool=zt_pool, zt_pool_previous=zt_prev)

    market = load_market(args, daily)
    params = LimitUpPullbackParams(
        limit_return_min=args.limit_return_min,
        confirm_return_max=args.confirm_return_max,
        confirm_high_return_max=args.confirm_high_return_max,
        next_high_target=args.next_high_target,
        require_turnover_rate=not args.ignore_turnover_rate,
        require_market_filter=not args.ignore_market_filter and args.market_source != "none",
    )
    candidate_params = LimitUpPullbackParams(
        limit_return_min=args.limit_return_min,
        confirm_return_max=args.confirm_return_max,
        confirm_high_return_max=args.confirm_high_return_max,
        next_high_target=args.next_high_target,
        require_turnover_rate=False,
        use_market_cap_turnover_tiers=False,
        require_market_filter=False,
    )
    candidates = select_limit_up_pullback(daily, params=candidate_params, market_data=None)
    samples_raw = select_limit_up_pullback(daily, params=params, market_data=market)
    samples = apply_post_filters(samples_raw, args)
    if not candidates.empty:
        selected_keys = set(zip(samples["date"].astype(str), samples["code"].astype(str)))
        candidates["selected_after_factors"] = [
            (str(d), str(c)) in selected_keys
            for d, c in zip(candidates["date"].astype(str), candidates["code"].astype(str))
        ]
    trades = build_trade_ledger(samples, target_return=args.next_high_target)
    stats = factor_group_stats(samples)

    out_dir = Path(args.out) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(out_dir / "samples.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(out_dir / "factor_group_stats.csv", index=False, encoding="utf-8-sig")
    summary = {
        "start": args.start,
        "end": args.end,
        "source": args.source,
        "market_source": args.market_source,
        "market_index": args.market_index,
        "params": params.to_dict(),
        "post_filters": {
            "wash_turnover_tier_include": args.wash_turnover_tier_include,
            "wash_turnover_tier_exclude": args.wash_turnover_tier_exclude,
            "limit_day_vol_ratio5_max": args.limit_day_vol_ratio5_max,
        },
        "num_samples": int(len(samples)),
        "num_samples_before_post_filters": int(len(samples_raw)),
        "num_candidates": int(len(candidates)),
        "num_trades": int(len(trades)),
        "label_3_hit_rate": float(samples["label_3"].mean()) if len(samples) else None,
        "avg_next_high_return": float(samples["next_high_return"].mean()) if len(samples) else None,
        "close_exit_win_rate": float(trades["is_profit"].mean()) if len(trades) else None,
        "avg_close_exit_return_pct": float(trades["return_pct"].mean()) if len(trades) else None,
        "total_close_exit_pnl_per_share": float(trades["gross_pnl"].sum()) if len(trades) else None,
        "outputs": {
            "candidates": str(out_dir / "candidates.csv"),
            "samples": str(out_dir / "samples.csv"),
            "trades": str(out_dir / "trades.csv"),
            "factor_group_stats": str(out_dir / "factor_group_stats.csv"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"samples={summary['num_samples']} hit_rate={summary['label_3_hit_rate']}")
    logger.info(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
