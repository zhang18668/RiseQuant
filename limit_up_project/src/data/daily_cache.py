"""D-007 日线数据缓存管理器

为"涨停二板主升浪"项目的阶段 1.3（日线数据回溯）提供本地 Parquet 缓存层。
与 ``ZtPoolLoader`` 形成互补：

- ``ZtPoolLoader``（D-006）按交易日落地涨停池 → 提供"哪些股票哪天涨停"。
- ``DailyCacheManager``（本模块）按 ``code`` 落地日线 → 提供"这些股票在 [start, end]
  窗口内的 OHLCV / change_pct"。

设计原则
--------
1. **按代码一个 parquet**：``<cache_dir>/<code>.parquet``，便于增量更新与按需加载，
   避免一次性合并大表带来的内存压力。
2. **标准 schema**：与 ``TDXDataLoader.load_day_file`` 一致 ——
   ``date | code | open | high | low | close | volume | turnover | change_pct``。
3. **数据源可插拔**：构造时注入 ``DataFetcher``（``TDXFetcher`` 默认，``AkshareFetcher``
   兜底），上层无需关心来源。
4. **增量优先**：每个 code 只补 ``last_cached_date`` 之后的新交易日；
   `--force-refresh` 才整覆盖。
5. **空状态友好**：fetcher 没拉到数据时仍写入空占位，避免反复请求。

策略相关
--------
对应《阶段 1.3 日线数据回溯》。验收标准：
- ``zt_pool_relevant`` universe 内每只股票均有 parquet 文件；
- 文件最新日期 ≥ 全局 ``end_date``（或股票退市日期）；
- 一致性校验由 ``scripts/verify_daily_cache.py``（阶段 1.4）单独负责。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

from src.data.zt_pool_loader import (
    ZtPoolLoader,
    SUPPORTED_TYPES as ZT_SUPPORTED_TYPES,
    is_main_board,
)
from src.data_fetch.base import DataFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------- 常量 ----------------------------
STANDARD_COLUMNS: List[str] = [
    "date", "code", "open", "high", "low",
    "close", "volume", "turnover", "change_pct",
]

DEFAULT_EXPAND_DAYS = 20   # universe 提取时左右扩展的交易日数（用于特征前置窗口）
MIN_VALID_BYTES = 200      # 小于此值视为"空占位 parquet"


# ---------------------------- 报告对象 ----------------------------
@dataclass
class CacheReport:
    """单只股票一次 update 的统计结果。"""

    code: str
    status: str = "ok"                   # ok / cached / empty / error
    rows_before: int = 0
    rows_after: int = 0
    rows_added: int = 0
    last_date_before: Optional[str] = None
    last_date_after: Optional[str] = None
    source: Optional[str] = None
    error: Optional[str] = None
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "code": self.code, "status": self.status,
            "rows_before": self.rows_before, "rows_after": self.rows_after,
            "rows_added": self.rows_added,
            "last_date_before": self.last_date_before,
            "last_date_after": self.last_date_after,
            "source": self.source, "error": self.error,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }


# ---------------------------- 工具函数 ----------------------------
def _to_ts(d: Union[str, _date, datetime, pd.Timestamp]) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _ensure_schema(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """确保 DataFrame 满足 STANDARD_COLUMNS，缺失列补 NaN；code 列补齐 6 位。"""
    if df is None or df.empty:
        return _empty_frame()
    out = df.copy()
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["code"] = str(code).zfill(6)
    out["date"] = pd.to_datetime(out["date"])
    out = (
        out[STANDARD_COLUMNS]
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


# ============================================================
# DailyCacheManager
# ============================================================
@dataclass
class DailyCacheManager:
    """按 code 维度的日线 Parquet 缓存。

    Parameters
    ----------
    cache_dir : str | Path
        缓存根目录。会自动 mkdir。
    primary_fetcher : DataFetcher
        主数据源（通常是 ``TDXFetcher``）。
    fallback_fetcher : DataFetcher, optional
        兜底数据源（通常是 ``AkshareFetcher``）。主源返回空时尝试调用。
    """

    cache_dir: Union[str, Path]
    primary_fetcher: DataFetcher
    fallback_fetcher: Optional[DataFetcher] = None

    # 内部状态（dataclass 不参与 init）
    _last_summary: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ====================== 路径与状态 ======================
    def _code_path(self, code: str) -> Path:
        return self.cache_dir / f"{str(code).zfill(6)}.parquet"

    def is_cached(self, code: str) -> bool:
        p = self._code_path(code)
        return p.exists() and p.stat().st_size >= MIN_VALID_BYTES

    def last_cached_date(self, code: str) -> Optional[pd.Timestamp]:
        df = self._read_cache(code)
        if df.empty:
            return None
        return pd.Timestamp(df["date"].max())

    def _read_cache(self, code: str) -> pd.DataFrame:
        p = self._code_path(code)
        if not p.exists():
            return _empty_frame()
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"read cache failed for {code}: {exc}; treating as empty")
            return _empty_frame()
        return _ensure_schema(df, code)

    def _write_cache(self, code: str, df: pd.DataFrame) -> Path:
        p = self._code_path(code)
        df = _ensure_schema(df, code)
        df.to_parquet(p, index=False)
        return p

    # ====================== universe 提取 ======================
    @staticmethod
    def universe_from_zt_pool(
        zt_loader: ZtPoolLoader,
        start_date: str,
        end_date: str,
        pool_types: Sequence[str] = ZT_SUPPORTED_TYPES,
        main_board_only: bool = True,
    ) -> List[str]:
        """从已落地的 ``zt_pool*`` 缓存里抽出 universe（仅读缓存，不联网）。

        :param zt_loader: 已构造的 :class:`ZtPoolLoader`。
        :param start_date, end_date: ``YYYYMMDD``。
        :param pool_types: 默认两个池都扫一遍。
        :param main_board_only: 仅保留沪深主板代码。
        :return: 去重后排序的 6 位 code 列表。
        """
        codes: Set[str] = set()
        for pt in pool_types:
            df = zt_loader.load_cached_range(start_date, end_date, pool_type=pt)
            if df.empty or "代码" not in df.columns:
                continue
            for raw in df["代码"].astype(str):
                c = raw.zfill(6)
                if main_board_only and not is_main_board(c):
                    continue
                codes.add(c)
        result = sorted(codes)
        logger.info(
            f"universe_from_zt_pool: {start_date}~{end_date} → {len(result)} 只 "
            f"(main_board_only={main_board_only}, pools={list(pool_types)})"
        )
        return result

    # ====================== 单 code 更新 ======================
    def update_code(
        self,
        code: str,
        start_date: Union[str, _date, datetime],
        end_date: Union[str, _date, datetime],
        force_refresh: bool = False,
        allow_fallback: bool = True,
    ) -> CacheReport:
        """增量更新单只股票的日线缓存。

        - ``force_refresh=True``：忽略已有缓存，按 [start, end] 全拉。
        - ``force_refresh=False``：仅拉 ``max(last_cached + 1, start)`` 到 ``end``。
        """
        t0 = datetime.now()
        report = CacheReport(code=str(code).zfill(6))

        start_ts = _to_ts(start_date)
        end_ts = _to_ts(end_date)

        existing = _empty_frame() if force_refresh else self._read_cache(code)
        report.rows_before = len(existing)
        if not existing.empty:
            report.last_date_before = _fmt_date(existing["date"].max())

        # 决定增量起点
        if existing.empty:
            fetch_start = start_ts
        else:
            fetch_start = max(start_ts, existing["date"].max() + pd.Timedelta(days=1))

        if fetch_start > end_ts:
            # 已是最新
            report.status = "cached"
            report.rows_after = report.rows_before
            report.last_date_after = report.last_date_before
            report.elapsed_sec = (datetime.now() - t0).total_seconds()
            return report

        fetch_start_str = fetch_start.strftime("%Y-%m-%d")
        fetch_end_str = end_ts.strftime("%Y-%m-%d")

        new_df, src = self._fetch(report.code, fetch_start_str, fetch_end_str, allow_fallback)
        report.source = src

        if new_df is None:
            report.status = "error"
            report.error = "all fetchers failed"
            self._write_cache(report.code, existing)  # 保留旧缓存
            report.rows_after = report.rows_before
            report.last_date_after = report.last_date_before
            report.elapsed_sec = (datetime.now() - t0).total_seconds()
            return report

        # 合并与去重
        if existing.empty:
            merged = new_df
        elif new_df.empty:
            merged = existing
        else:
            merged = pd.concat([existing, new_df], ignore_index=True)
        merged = _ensure_schema(merged, report.code)
        # 限制到目标区间（防止 fetcher 越界）
        merged = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)].reset_index(drop=True)

        self._write_cache(report.code, merged)

        report.rows_after = len(merged)
        report.rows_added = max(0, report.rows_after - report.rows_before)
        if not merged.empty:
            report.last_date_after = _fmt_date(merged["date"].max())
        if merged.empty:
            report.status = "empty"
        elif report.rows_added == 0:
            report.status = "cached"
        else:
            report.status = "ok"

        report.elapsed_sec = (datetime.now() - t0).total_seconds()
        return report

    def _fetch(
        self,
        code: str,
        start_str: str,
        end_str: str,
        allow_fallback: bool,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """按 primary → fallback 顺序尝试拉取。返回 (df, source_name)；都失败返回 (None, None)。"""
        for src_name, fetcher in self._iter_sources(allow_fallback):
            try:
                df = fetcher.load_batch([code], start_date=start_str, end_date=end_str)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{src_name}] load_batch({code}) raise: {exc}")
                continue
            if df is None or df.empty:
                # 主源空 → 尝试下一个
                logger.debug(f"[{src_name}] empty for {code} {start_str}~{end_str}")
                continue
            return _ensure_schema(df, code), src_name
        return None, None

    def _iter_sources(self, allow_fallback: bool):
        yield "primary", self.primary_fetcher
        if allow_fallback and self.fallback_fetcher is not None:
            yield "fallback", self.fallback_fetcher

    # ====================== 批量更新 ======================
    def update_batch(
        self,
        codes: Sequence[str],
        start_date: Union[str, _date, datetime],
        end_date: Union[str, _date, datetime],
        force_refresh: bool = False,
        allow_fallback: bool = True,
        progress_every: int = 100,
    ) -> List[CacheReport]:
        """对一组 code 批量调用 ``update_code``，自动打印进度。"""
        reports: List[CacheReport] = []
        n = len(codes)
        n_ok = n_cached = n_empty = n_err = 0

        logger.info(
            f"update_batch: {n} 只股票 {start_date}~{end_date} "
            f"force_refresh={force_refresh} fallback={'on' if allow_fallback else 'off'}"
        )

        for i, code in enumerate(codes, 1):
            r = self.update_code(
                code=code,
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
                allow_fallback=allow_fallback,
            )
            reports.append(r)
            if r.status == "ok":
                n_ok += 1
            elif r.status == "cached":
                n_cached += 1
            elif r.status == "empty":
                n_empty += 1
            else:
                n_err += 1
                logger.warning(f"[{code}] {r.status}: {r.error}")

            if i % progress_every == 0:
                logger.info(
                    f"  进度 {i}/{n}：ok={n_ok} cached={n_cached} empty={n_empty} err={n_err}"
                )

        logger.info(
            f"update_batch 完成：ok={n_ok} cached={n_cached} empty={n_empty} err={n_err}"
        )
        return reports

    # ====================== 加载层 ======================
    def load_code(
        self,
        code: str,
        start_date: Optional[Union[str, _date, datetime]] = None,
        end_date: Optional[Union[str, _date, datetime]] = None,
    ) -> pd.DataFrame:
        """从缓存读取单只股票的日线，可选区间过滤。"""
        df = self._read_cache(code)
        if df.empty:
            return df
        if start_date is not None:
            df = df[df["date"] >= _to_ts(start_date)]
        if end_date is not None:
            df = df[df["date"] <= _to_ts(end_date)]
        return df.reset_index(drop=True)

    def load_range(
        self,
        codes: Iterable[str],
        start_date: Optional[Union[str, _date, datetime]] = None,
        end_date: Optional[Union[str, _date, datetime]] = None,
    ) -> pd.DataFrame:
        """批量读取并合并多只股票的日线。缺失缓存的 code 自动跳过。"""
        frames: List[pd.DataFrame] = []
        for code in codes:
            df = self.load_code(code, start_date=start_date, end_date=end_date)
            if not df.empty:
                frames.append(df)
        if not frames:
            return _empty_frame()
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["code", "date"]).reset_index(drop=True)
        return out

    # ====================== 摘要 ======================
    def cache_summary(self) -> dict:
        """统计缓存目录的整体状态。"""
        files = sorted(self.cache_dir.glob("*.parquet"))
        if not files:
            return {"count": 0, "cache_dir": str(self.cache_dir)}
        total_bytes = sum(f.stat().st_size for f in files)
        n_non_empty = sum(1 for f in files if f.stat().st_size >= MIN_VALID_BYTES)
        # 抽查若干文件估算日期范围（避免逐文件读太慢）
        sample = files[:: max(1, len(files) // 50)]
        dmin: Optional[pd.Timestamp] = None
        dmax: Optional[pd.Timestamp] = None
        total_rows = 0
        for f in sample:
            try:
                df = pd.read_parquet(f, columns=["date"])
            except Exception:  # noqa: BLE001
                continue
            if df.empty:
                continue
            total_rows += len(df)
            d_lo, d_hi = df["date"].min(), df["date"].max()
            dmin = d_lo if dmin is None or d_lo < dmin else dmin
            dmax = d_hi if dmax is None or d_hi > dmax else dmax
        return {
            "cache_dir": str(self.cache_dir),
            "count": len(files),
            "non_empty": n_non_empty,
            "empty": len(files) - n_non_empty,
            "total_size_mb": round(total_bytes / 1024 / 1024, 3),
            "sampled_rows": int(total_rows),
            "earliest_date": _fmt_date(dmin) if dmin is not None else None,
            "latest_date": _fmt_date(dmax) if dmax is not None else None,
            "sampled_files": len(sample),
        }

    # ====================== 反向便利：通过 zt_pool 直接更新 ======================
    def update_universe_from_zt_pool(
        self,
        zt_loader: ZtPoolLoader,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        allow_fallback: bool = True,
        main_board_only: bool = True,
    ) -> List[CacheReport]:
        """一键：从 zt_pool 缓存抽 universe，批量更新日线。"""
        codes = self.universe_from_zt_pool(
            zt_loader=zt_loader,
            start_date=start_date,
            end_date=end_date,
            main_board_only=main_board_only,
        )
        if not codes:
            logger.warning("universe 为空，没有可更新的股票。")
            return []
        # daily 的 start/end 用 ISO 字符串，方便 fetcher
        start_iso = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_iso = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        return self.update_batch(
            codes=codes,
            start_date=start_iso,
            end_date=end_iso,
            force_refresh=force_refresh,
            allow_fallback=allow_fallback,
        )
