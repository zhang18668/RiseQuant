"""D-008 PipelineDataLoader — 给 run_pipeline / wash_*_pipeline 用的统一加载入口

把"从 daily_cache / TDX / AkShare 之间选择最合适来源"的逻辑集中到一处，
避免每个流水线脚本重复一遍。

source 语义
-----------
- ``cache`` (默认)：仅从 ``data/daily_cache/`` 读 parquet；缺失则 raise 并引导。
- ``tdx``：直接调 ``TDXDataLoader.load_batch``（老行为，全市场主板）。
- ``auto``：cache 优先；为空时回退到 ``tdx``。

universe 选择
-------------
- 优先从 ``data/zt_pool_cache/`` 抽（与 ``scripts/build_daily_cache.py`` 一致）。
- zt_pool 为空时回退到 daily_cache 目录扫描。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.daily_cache import DailyCacheManager
from src.data.zt_pool_loader import ZtPoolLoader
from src.data.tdx_loader import TDXDataLoader
from src.data_fetch.base import DataFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_universe_for_cache(
    zt_cache_dir: Path,
    daily_cache_dir: Path,
    start_date: str,
    end_date: str,
) -> List[str]:
    """决定 cache-first 模式下要读哪些 code。

    优先从 zt_pool 缓存抽 universe（与 build_daily_cache 一致）；
    若 zt_pool 缓存空，则从 daily_cache 目录现存 parquet 列出代码。
    """
    if zt_cache_dir.exists():
        try:
            zt_loader = ZtPoolLoader(cache_dir=str(zt_cache_dir), require_akshare=False)
            codes = DailyCacheManager.universe_from_zt_pool(
                zt_loader=zt_loader,
                start_date=str(start_date).replace("-", ""),
                end_date=str(end_date).replace("-", ""),
                main_board_only=True,
            )
            if codes:
                logger.info(f"universe 来自 zt_pool 缓存: {len(codes)} 只")
                return codes
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"读 zt_pool 缓存失败：{exc}; 回退到目录扫描")

    if daily_cache_dir.exists():
        codes = sorted(p.stem for p in daily_cache_dir.glob("*.parquet"))
        if codes:
            logger.info(f"universe 来自 daily_cache 目录扫描: {len(codes)} 只")
            return codes

    return []


def _make_readonly_cache_mgr(cache_dir: Path) -> DailyCacheManager:
    """只读 DailyCacheManager（用 dummy fetcher，不会发起请求）。"""

    class _DummyFetcher(DataFetcher):
        def load_stock_list(self, exchange: str = "sh") -> pd.DataFrame:
            return pd.DataFrame(columns=["code", "market"])

        def load_batch(self, codes, start_date=None, end_date=None) -> pd.DataFrame:
            return pd.DataFrame()

    return DailyCacheManager(
        cache_dir=cache_dir,
        primary_fetcher=_DummyFetcher(),
        fallback_fetcher=None,
    )


def load_daily_data(config, source: str = "cache") -> pd.DataFrame:
    """加载日线数据，支持 cache / tdx / auto 三种来源策略。

    Parameters
    ----------
    config : Config
        ``src.utils.config.get_config()`` 返回的对象，至少需要 ``data`` section。
    source : str
        参见模块 docstring。

    Returns
    -------
    pd.DataFrame
        date | code | open | high | low | close | volume | turnover | change_pct
    """
    data_config = config.get_section("data")
    start_date = data_config.get("start_date", "2020-01-01")
    end_date = data_config.get("end_date", "2026-12-31")

    from dateutil.relativedelta import relativedelta
    history_start = (
        pd.Timestamp(start_date) - relativedelta(months=3)
    ).strftime("%Y-%m-%d")
    logger.info(
        f"数据加载范围: {start_date} ~ {end_date} (含预读窗 {history_start})"
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    daily_cache_dir = Path(
        data_config.get("daily_cache_dir",
                        str(project_root / "data" / "daily_cache"))
    )
    zt_cache_dir = Path(
        data_config.get("zt_pool_cache_dir",
                        str(project_root / "data" / "zt_pool_cache"))
    )
    tdx_path = data_config.get("tdx_vipdoc", r"C:\new_tdx\vipdoc")

    # -------------- cache / auto --------------
    if source in ("cache", "auto"):
        logger.info(f"[Step 0] 数据源: source={source}, dir={daily_cache_dir}")
        cache_ok = daily_cache_dir.exists() and any(daily_cache_dir.glob("*.parquet"))
        if not cache_ok:
            msg = (
                f"daily_cache 为空: {daily_cache_dir}\n"
                f"  请先运行: python scripts/build_daily_cache.py\n"
                f"  或改用 --source tdx 直接读通达信本地数据"
            )
            if source == "cache":
                raise RuntimeError(msg)
            logger.warning(msg + "\n  → 自动回退到 TDX")
        else:
            codes = resolve_universe_for_cache(
                zt_cache_dir, daily_cache_dir, start_date, end_date
            )
            if not codes:
                if source == "cache":
                    raise RuntimeError(
                        "未能确定 universe（zt_pool / daily_cache 都空）。"
                        "请先运行 build_zt_pool_cache.py + build_daily_cache.py"
                    )
                logger.warning("universe 为空，回退到 TDX 全市场加载")
            else:
                mgr = _make_readonly_cache_mgr(daily_cache_dir)
                daily_data = mgr.load_range(
                    codes, start_date=history_start, end_date=end_date
                )
                if daily_data.empty:
                    if source == "cache":
                        raise RuntimeError(
                            f"daily_cache 命中 {len(codes)} 只代码但读出空表。"
                            f"请检查缓存或重新运行 build_daily_cache.py"
                        )
                    logger.warning("daily_cache 读出空表，回退到 TDX")
                else:
                    logger.info(
                        f"daily_cache 命中: {len(daily_data)} 条日线 × "
                        f"{daily_data['code'].nunique()} 只代码"
                    )
                    return daily_data

    # -------------- tdx / auto fallback --------------
    logger.info(f"[Step 0] 数据源: TDX 本地 ({tdx_path})")
    tdx_loader = TDXDataLoader(tdx_path)
    if tdx_loader.data_path is None:
        raise RuntimeError(
            f"未找到通达信数据目录: {tdx_path}\n"
            f"  检查 config/default.yaml 的 data.tdx_vipdoc 或环境变量 TDX_PATH"
        )

    main_codes: List[str] = []
    for ex in ("sh", "sz"):
        lst = tdx_loader.load_stock_list(ex)
        if lst is None or lst.empty:
            continue
        for c in lst["code"].tolist():
            if tdx_loader.is_main_board(c):
                main_codes.append(c)
    logger.info(f"TDX 主板代码数: {len(main_codes)}")

    daily_data = tdx_loader.load_batch(
        codes=main_codes,
        start_date=history_start,
        end_date=end_date,
    )
    if daily_data.empty:
        raise RuntimeError("TDX 加载结果为空")
    logger.info(
        f"TDX 加载: {len(daily_data)} 条日线 × "
        f"{daily_data['code'].nunique()} 只代码"
    )
    return daily_data
