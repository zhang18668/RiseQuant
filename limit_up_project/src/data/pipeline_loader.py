"""Pipeline daily-data loader.

This module centralizes the source selection used by top-level pipelines:

- ``cache``: read only ``data/daily_cache/*.parquet`` and fail with guidance if
  the cache is missing or empty.
- ``tdx``: read directly from local TDX ``vipdoc`` files.
- ``auto``: try cache first, then fall back to TDX.

The cache universe is resolved the same way as ``build_daily_cache.py``:
prefer the landed zt-pool cache, then fall back to scanning existing daily
cache files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.data.daily_cache import DailyCacheManager
from src.data.tdx_loader import TDXDataLoader
from src.data.zt_pool_loader import ZtPoolLoader
from src.data_fetch.base import DataFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _ReadOnlyFetcher(DataFetcher):
    """Dummy fetcher for a DailyCacheManager that is only used for reads."""

    def load_stock_list(self, exchange: str = "sh") -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "market"])

    def load_batch(
        self,
        codes: Iterable[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str | Path, root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _history_start(start_date: str) -> str:
    return (pd.Timestamp(start_date) - relativedelta(months=3)).strftime("%Y-%m-%d")


def resolve_universe_for_cache(
    zt_cache_dir: str | Path,
    daily_cache_dir: str | Path,
    start_date: str,
    end_date: str,
) -> List[str]:
    """Resolve the list of codes to read in cache mode."""
    zt_cache_dir = Path(zt_cache_dir)
    daily_cache_dir = Path(daily_cache_dir)

    if zt_cache_dir.exists():
        try:
            zt_loader = ZtPoolLoader(cache_dir=zt_cache_dir, require_akshare=False)
            codes = DailyCacheManager.universe_from_zt_pool(
                zt_loader=zt_loader,
                start_date=str(start_date).replace("-", ""),
                end_date=str(end_date).replace("-", ""),
                main_board_only=True,
            )
            if codes:
                logger.info("Universe from zt_pool cache: %s codes", len(codes))
                return codes
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve universe from zt_pool cache (%s); "
                "falling back to daily_cache scan.",
                exc,
            )

    if daily_cache_dir.exists():
        codes = sorted(p.stem for p in daily_cache_dir.glob("*.parquet"))
        if codes:
            logger.info("Universe from daily_cache scan: %s codes", len(codes))
            return codes

    return []


def _make_readonly_cache_mgr(cache_dir: str | Path) -> DailyCacheManager:
    return DailyCacheManager(
        cache_dir=cache_dir,
        primary_fetcher=_ReadOnlyFetcher(),
        fallback_fetcher=None,
    )


def _load_from_cache(
    daily_cache_dir: Path,
    zt_cache_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    codes = resolve_universe_for_cache(
        zt_cache_dir=zt_cache_dir,
        daily_cache_dir=daily_cache_dir,
        start_date=start_date,
        end_date=end_date,
    )
    if not codes:
        raise RuntimeError(
            "Unable to resolve cache universe. Run build_zt_pool_cache.py and "
            "build_daily_cache.py first, or use --source tdx/auto."
        )

    mgr = _make_readonly_cache_mgr(daily_cache_dir)
    data = mgr.load_range(
        codes,
        start_date=_history_start(start_date),
        end_date=end_date,
    )
    if data.empty:
        raise RuntimeError(
            f"daily_cache matched {len(codes)} codes but returned no rows from "
            f"{daily_cache_dir}. Rebuild the cache or use --source tdx/auto."
        )
    return data


def _load_from_tdx(tdx_path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    loader = TDXDataLoader(str(tdx_path))
    if loader.data_path is None:
        raise RuntimeError(
            f"TDX vipdoc path not found: {tdx_path}. Update data.tdx_vipdoc "
            "or run with --source cache after building daily_cache."
        )

    codes: List[str] = []
    for exchange in ("sh", "sz"):
        stock_list = loader.load_stock_list(exchange)
        if stock_list is None or stock_list.empty:
            continue
        codes.extend(
            str(code).zfill(6)
            for code in stock_list["code"].tolist()
            if loader.is_main_board(code)
        )

    if not codes:
        raise RuntimeError(f"No main-board codes found from TDX path: {tdx_path}")

    data = loader.load_batch(
        codes=codes,
        start_date=_history_start(start_date),
        end_date=end_date,
    )
    if data.empty:
        raise RuntimeError(f"TDX returned no daily rows from {tdx_path}")
    return data


def load_daily_data(config, source: str = "cache") -> pd.DataFrame:
    """Load daily bars for pipelines using cache, TDX, or cache-then-TDX."""
    source = (source or "cache").lower()
    if source not in {"cache", "tdx", "auto"}:
        raise ValueError("source must be one of: cache, tdx, auto")

    data_config = config.get_section("data")
    start_date = data_config.get("start_date", "2020-01-01")
    end_date = data_config.get("end_date", "2026-12-31")

    root = _project_root()
    daily_cache_dir = _resolve_path(
        data_config.get("daily_cache_dir", "./data/daily_cache"),
        root,
    )
    zt_cache_dir = _resolve_path(
        data_config.get("zt_pool_cache_dir", "./data/zt_pool_cache"),
        root,
    )
    tdx_path = Path(data_config.get("tdx_vipdoc", r"C:\new_tdx\vipdoc"))

    logger.info(
        "Daily data window: %s ~ %s (history start: %s)",
        start_date,
        end_date,
        _history_start(start_date),
    )

    if source in {"cache", "auto"}:
        try:
            logger.info("Loading daily data from cache: %s", daily_cache_dir)
            data = _load_from_cache(daily_cache_dir, zt_cache_dir, start_date, end_date)
            logger.info(
                "daily_cache hit: %s rows x %s codes",
                len(data),
                data["code"].nunique(),
            )
            return data
        except RuntimeError as exc:
            if source == "cache":
                raise
            logger.warning("Cache load failed; falling back to TDX. Reason: %s", exc)

    logger.info("Loading daily data from TDX: %s", tdx_path)
    data = _load_from_tdx(tdx_path, start_date, end_date)
    logger.info("TDX load complete: %s rows x %s codes", len(data), data["code"].nunique())
    return data
