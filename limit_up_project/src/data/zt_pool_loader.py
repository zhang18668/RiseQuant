"""D-006 涨停板池数据加载器（东方财富 / AkShare）

封装两个 akshare 接口：

- ``stock_zt_pool_em``          — 当日涨停股池
- ``stock_zt_pool_previous_em`` — 昨日涨停股池（含 T-1 涨停今日表现）

提供：单日拉取（含重试）、批量拉取、Parquet 本地缓存、增量更新、
主板过滤、首次封板时间分档、封板强度计算。

缓存布局
--------
::

    <cache_dir>/zt_pool/YYYYMMDD.parquet           # 当日涨停股池
    <cache_dir>/zt_pool_previous/YYYYMMDD.parquet  # 昨日涨停股池

策略相关
--------
《策略规划.md》第 4.2 节信号过滤与第 5.1 节多信号排序所依赖的核心字段
（首次封板时间、封板资金、流通市值）由本模块统一供应。

依赖
----
需要 ``pyarrow`` 才能读写 Parquet。若未安装请运行::

    pip install pyarrow
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional, Union

import pandas as pd

try:
    import akshare as ak
    _HAS_AK = True
except ImportError:  # pragma: no cover
    ak = None  # type: ignore
    _HAS_AK = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_parquet_if_non_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return df if df is not None and not df.empty else pd.DataFrame()


def zt_pool_cache_quality(
    cache_dir: Union[str, Path],
    start_date: str,
    end_date: str,
    pool_type: str = "zt_pool",
    min_monthly_non_empty: int = 18,
) -> dict:
    """Summarize landed zt-pool cache health.

    The important guardrail is monthly non-empty coverage.  A month with many
    empty parquet placeholders usually means a proxy/API failure was cached as
    "valid" data, which can silently shrink multi-year reports to a few weeks.
    """
    cache_dir = Path(cache_dir)
    sub = cache_dir / pool_type
    dates = list(_iter_workdays(start_date, end_date))
    monthly: dict[str, dict] = {}
    total_files = 0
    total_non_empty = 0
    missing = 0

    for d in dates:
        month = d[:6]
        item = monthly.setdefault(month, {"files": 0, "non_empty": 0, "missing": 0})
        path = sub / f"{d}.parquet"
        if not path.exists():
            missing += 1
            item["missing"] += 1
            continue
        total_files += 1
        item["files"] += 1
        if not _read_parquet_if_non_empty(path).empty:
            total_non_empty += 1
            item["non_empty"] += 1

    bad_months = [
        {"month": month, **stats}
        for month, stats in sorted(monthly.items())
        if stats["non_empty"] < min_monthly_non_empty
    ]
    return {
        "pool_type": pool_type,
        "start_date": start_date,
        "end_date": end_date,
        "min_monthly_non_empty": min_monthly_non_empty,
        "expected_workdays": len(dates),
        "files": total_files,
        "non_empty": total_non_empty,
        "missing": missing,
        "monthly": monthly,
        "bad_months": bad_months,
        "ok": not bad_months,
    }


def assert_zt_pool_cache_quality(
    cache_dir: Union[str, Path],
    start_date: str,
    end_date: str,
    pool_type: str = "zt_pool",
    min_monthly_non_empty: int = 18,
) -> dict:
    summary = zt_pool_cache_quality(
        cache_dir=cache_dir,
        start_date=start_date,
        end_date=end_date,
        pool_type=pool_type,
        min_monthly_non_empty=min_monthly_non_empty,
    )
    if not summary["ok"]:
        preview = ", ".join(
            f"{m['month']}={m['non_empty']}" for m in summary["bad_months"][:8]
        )
        raise RuntimeError(
            f"{pool_type} cache quality failed: monthly non-empty files < "
            f"{min_monthly_non_empty}; bad months: {preview}"
        )
    return summary


# ---------------------------- 常量 ----------------------------
ZT_POOL_TYPE = "zt_pool"
ZT_PREV_TYPE = "zt_pool_previous"
SUPPORTED_TYPES = (ZT_POOL_TYPE, ZT_PREV_TYPE)

MAIN_BOARD_PREFIXES = (
    "600", "601", "603", "605",   # 沪市主板
    "000", "001", "002",           # 深市主板
)

# 时间分档边界（HHMMSS 字符串字典序比较）
# Tier 1：集合竞价封板（092500-092959），含 9:25 一字板与少量极早盘封板
# Tier 2：9:30:00 - 10:00:00 开盘后早盘封板
# Tier 3：10:00:01 - 11:30:00 上午中后段封板
# 中午_异常：11:30 - 13:00（午间无交易，理论上不该出现）
# 下午_排除：≥ 13:00:00 → 直接过滤
TIER1_END = "093000"
TIER2_END = "100000"
TIER3_END = "113000"
AFTERNOON_START = "130000"

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # 秒（首次重试间隔；后续指数退避）
DEFAULT_REQUEST_INTERVAL = 0.2  # 批量拉取时两次请求间最小间隔（秒）


# ---------------------------- 工具函数 ----------------------------
def is_main_board(code: Union[str, int]) -> bool:
    """主板判定。沪市 600/601/603/605 + 深市 000/001/002 返回 True；
    创业板 300/301、科创板 688、北交所 8/4 系列返回 False。"""
    code_str = str(code).zfill(6)
    return code_str.startswith(MAIN_BOARD_PREFIXES)


def time_tier(time_str) -> str:
    """首次封板时间 (6 位 HHMMSS) 映射到分档名。"""
    if pd.isna(time_str):
        return "未知"
    t = str(time_str).zfill(6)
    if t < TIER1_END:
        return "Tier1"
    if t < TIER2_END:
        return "Tier2"
    if t <= TIER3_END:
        return "Tier3"
    if t < AFTERNOON_START:
        return "中午_异常"
    return "下午_排除"


_TIER_RANK = {
    "Tier1": 1, "Tier2": 2, "Tier3": 3,
    "中午_异常": 4, "下午_排除": 5, "未知": 9,
}


def time_tier_rank(time_str) -> int:
    """时间分档对应的排序值，数字小 = 排前面。"""
    return _TIER_RANK.get(time_tier(time_str), 9)


def is_acceptable_tier(time_str) -> bool:
    """信号入池判定：仅 Tier 1/2/3 接受，中午异常与下午排除。"""
    return time_tier_rank(time_str) <= 3


def compute_seal_strength(seal_amount, float_mv) -> float:
    """封板强度（百分比）= 封板资金 / 流通市值 × 100。

    任一字段缺失或流通市值 ≤ 0 时返回 NaN。
    """
    if pd.isna(seal_amount) or pd.isna(float_mv):
        return float("nan")
    try:
        fmv = float(float_mv)
        if fmv <= 0:
            return float("nan")
        return float(seal_amount) / fmv * 100.0
    except (TypeError, ValueError):
        return float("nan")


def _iter_workdays(start: str, end: str) -> Iterator[str]:
    """枚举 [start, end] 之间所有工作日（YYYYMMDD），周末跳过。"""
    d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()
    while d <= end_d:
        if d.weekday() < 5:
            yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


# ---------------------------- 主类 ----------------------------
@dataclass
class ZtPoolLoader:
    """涨停板池数据加载器。

    Parameters
    ----------
    cache_dir : str | Path
        缓存根目录。会自动创建 ``zt_pool/`` 和 ``zt_pool_previous/`` 子目录。
    require_akshare : bool
        若为 True 且 akshare 未安装则在初始化时抛 ImportError。否则仅在尝试联网时报错。
    max_retries : int
        单次请求最大重试次数。
    retry_delay : float
        首次重试间隔秒数，后续按线性退避 (delay * attempt)。
    request_interval : float
        批量拉取时两次请求间最小间隔，避免对服务器过度压力。
    """

    cache_dir: Union[str, Path]
    require_akshare: bool = False
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    request_interval: float = DEFAULT_REQUEST_INTERVAL

    # 内部状态
    _last_request_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for t in SUPPORTED_TYPES:
            (self.cache_dir / t).mkdir(exist_ok=True)

        if self.require_akshare and not _HAS_AK:
            raise ImportError(
                "akshare is not installed; cannot fetch online data"
            )

    # ====================== 缓存层 ======================
    def _cache_path(self, pool_type: str, date: str) -> Path:
        if pool_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"unknown pool_type: {pool_type!r}, must be one of {SUPPORTED_TYPES}"
            )
        return self.cache_dir / pool_type / f"{date}.parquet"

    def is_cached(self, pool_type: str, date: str) -> bool:
        """检查某日某类型数据是否已缓存。"""
        return self._cache_path(pool_type, date).exists()

    def is_non_empty_cached(self, pool_type: str, date: str) -> bool:
        """Return True only when the landed parquet exists and has rows."""
        return not _read_parquet_if_non_empty(self._cache_path(pool_type, date)).empty

    def cache_summary(self, pool_type: str = ZT_POOL_TYPE) -> dict:
        """汇总某类型缓存文件统计：数量、日期范围、占用空间。"""
        sub = self.cache_dir / pool_type
        files = sorted(sub.glob("*.parquet"))
        if not files:
            return {"pool_type": pool_type, "count": 0}
        dates = [f.stem for f in files]
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "pool_type": pool_type,
            "count": len(files),
            "earliest": dates[0],
            "latest": dates[-1],
            "total_size_mb": round(total_bytes / 1024 / 1024, 3),
        }

    # ====================== 单日拉取 ======================
    def _api_for(self, pool_type: str):
        if not _HAS_AK:
            raise ImportError("akshare not installed; cannot fetch online")
        if pool_type == ZT_POOL_TYPE:
            return ak.stock_zt_pool_em
        if pool_type == ZT_PREV_TYPE:
            return ak.stock_zt_pool_previous_em
        raise ValueError(f"unknown pool_type: {pool_type}")

    def _throttle(self) -> None:
        """两次联网请求之间的最小间隔。"""
        elapsed = time.time() - self._last_request_ts
        wait = self.request_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.time()

    @staticmethod
    def _normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
        """确保关键文本字段为字符串（防止时间被解析为整数丢失前导 0、代码丢前导 0 等）。"""
        if df.empty:
            return df
        if "代码" in df.columns:
            df["代码"] = df["代码"].astype(str).str.zfill(6)
        for col in ("首次封板时间", "最后封板时间"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.zfill(6)
        return df

    def fetch_one(
        self,
        date: str,
        pool_type: str = ZT_POOL_TYPE,
        use_cache: bool = True,
        save_cache: bool = True,
        refresh_empty_cache: bool = False,
        protect_non_empty_cache: bool = True,
    ) -> pd.DataFrame:
        """拉取单个交易日某类型的涨停板池数据。

        :param date: 交易日 YYYYMMDD
        :param pool_type: ``zt_pool`` 或 ``zt_pool_previous``
        :param use_cache: 是否优先读本地缓存
        :param save_cache: 是否写本地缓存（即使为空数据也会写入空文件作占位，防止反复拉取）
        :return: 原始字段 DataFrame；非交易日 / 无涨停时返回空 DataFrame
        """
        cache_path = self._cache_path(pool_type, date)

        if use_cache and cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if not refresh_empty_cache or not cached.empty:
                return cached

        api = self._api_for(pool_type)
        last_err: Optional[Exception] = None
        df: Optional[pd.DataFrame] = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                df = api(date=date)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    f"fetch {pool_type} {date} attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        if df is None:
            raise RuntimeError(
                f"fetch {pool_type} {date} failed after {self.max_retries} retries: {last_err}"
            )

        if df is None or df.empty:
            df = pd.DataFrame()
        else:
            df = self._normalize_string_columns(df.copy())

        if save_cache:
            # 空 DataFrame 也写入，作为"已确认无数据"的占位（防止反复拉取非交易日）
            if df.empty and protect_non_empty_cache and cache_path.exists():
                cached = _read_parquet_if_non_empty(cache_path)
                if not cached.empty:
                    logger.warning(
                        f"skip empty overwrite for non-empty cache: {pool_type} {date}"
                    )
                    return cached
            df.to_parquet(cache_path, index=False)

        return df

    # ====================== 批量拉取 ======================
    def fetch_range(
        self,
        start_date: str,
        end_date: str,
        pool_type: str = ZT_POOL_TYPE,
        use_cache: bool = True,
        save_cache: bool = True,
        refresh_empty_cache: bool = False,
        protect_non_empty_cache: bool = True,
        skip_errors: bool = True,
        progress_every: int = 50,
    ) -> pd.DataFrame:
        """拉取日期区间数据，自动跳过周末和缓存命中。

        :param skip_errors: True 时单日失败仅记录日志继续；False 则抛出
        :param progress_every: 每处理 N 个日期打印一条进度
        :return: 合并后的长表，含 ``trade_date`` 列（datetime）
        """
        dates = list(_iter_workdays(start_date, end_date))
        logger.info(
            f"fetch_range[{pool_type}] {start_date}~{end_date}: {len(dates)} 工作日"
        )

        frames: List[pd.DataFrame] = []
        n_cached = 0
        n_fetched = 0
        n_empty = 0
        n_error = 0

        for i, date in enumerate(dates, 1):
            from_cache = use_cache and self.is_cached(pool_type, date)
            try:
                df = self.fetch_one(
                    date,
                    pool_type=pool_type,
                    use_cache=use_cache,
                    save_cache=save_cache,
                    refresh_empty_cache=refresh_empty_cache,
                    protect_non_empty_cache=protect_non_empty_cache,
                )
            except Exception as e:  # noqa: BLE001
                n_error += 1
                logger.error(f"fetch {date} failed: {e}")
                if skip_errors:
                    continue
                raise

            if from_cache:
                n_cached += 1
            else:
                n_fetched += 1

            if df.empty:
                n_empty += 1
                continue

            df = df.copy()
            df["trade_date"] = pd.to_datetime(date)
            frames.append(df)

            if i % progress_every == 0:
                logger.info(
                    f"  进度 {i}/{len(dates)}：缓存={n_cached} 新拉={n_fetched} 空={n_empty} 错={n_error}"
                )

        logger.info(
            f"fetch_range[{pool_type}] 完成：缓存={n_cached} 新拉={n_fetched} 空={n_empty} 错={n_error}"
        )

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ====================== 仅从缓存加载 ======================
    def load_cached_range(
        self,
        start_date: str,
        end_date: str,
        pool_type: str = ZT_POOL_TYPE,
    ) -> pd.DataFrame:
        """仅从本地缓存加载区间数据，不联网。缺失日期静默跳过。"""
        dates = list(_iter_workdays(start_date, end_date))
        frames: List[pd.DataFrame] = []
        for date in dates:
            p = self._cache_path(pool_type, date)
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if df.empty:
                continue
            df = df.copy()
            df["trade_date"] = pd.to_datetime(date)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ====================== 数据增强（衍生字段）======================
    @staticmethod
    def enrich(
        df: pd.DataFrame,
        main_board_only: bool = True,
        acceptable_tier_only: bool = False,
    ) -> pd.DataFrame:
        """添加策略所需的衍生字段：

        - ``is_main_board`` : bool，主板标记
        - ``time_tier``     : str ，分档名 (Tier1/Tier2/Tier3/中午_异常/下午_排除/未知)
        - ``time_tier_rank``: int ，分档排序值（1-9，小者优先）
        - ``seal_strength_pct`` : float，封板强度百分比

        :param main_board_only: True 时仅保留主板（建议默认）
        :param acceptable_tier_only: True 时仅保留 Tier 1/2/3（=策略候选池）
        """
        if df.empty:
            return df

        out = df.copy()

        if "代码" in out.columns:
            out["is_main_board"] = out["代码"].apply(is_main_board)
        else:
            out["is_main_board"] = False

        if main_board_only:
            out = out[out["is_main_board"]].copy()

        if "首次封板时间" in out.columns:
            out["time_tier"] = out["首次封板时间"].apply(time_tier)
            out["time_tier_rank"] = out["首次封板时间"].apply(time_tier_rank)
        else:
            out["time_tier"] = "未知"
            out["time_tier_rank"] = 9

        if "封板资金" in out.columns and "流通市值" in out.columns:
            out["seal_strength_pct"] = [
                compute_seal_strength(s, m)
                for s, m in zip(out["封板资金"], out["流通市值"])
            ]
        else:
            out["seal_strength_pct"] = float("nan")

        if acceptable_tier_only:
            out = out[out["time_tier_rank"] <= 3].copy()

        return out
