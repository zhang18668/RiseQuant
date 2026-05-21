"""DailyCacheManager 单元测试 (D-007)

覆盖：
- 首次写入：mock fetcher → parquet 落地 → cache_summary 反映
- 增量更新：第二次调用不再请求 (status=cached)
- force_refresh：强制重新拉取
- universe_from_zt_pool：从 mock 涨停池缓存里抽 universe
- load_range：跨多 code 合并
- fallback：primary 空时切到 fallback
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.data.daily_cache import DailyCacheManager, CacheReport, STANDARD_COLUMNS
from src.data.zt_pool_loader import ZtPoolLoader
from src.data_fetch.base import DataFetcher


# ============================================================
# 测试工具
# ============================================================
class MockFetcher(DataFetcher):
    """记录调用次数 + 返回预设 DataFrame 的 fetcher。"""

    def __init__(self, frame_by_code: Optional[dict] = None,
                 stock_list: Optional[pd.DataFrame] = None):
        self.frame_by_code = frame_by_code or {}
        self.stock_list = stock_list if stock_list is not None else pd.DataFrame(columns=["code", "market"])
        self.calls: List[tuple] = []

    def load_stock_list(self, exchange="sh") -> pd.DataFrame:
        return self.stock_list

    def load_batch(self, codes: Iterable[str],
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        codes = list(codes)
        self.calls.append((tuple(codes), start_date, end_date))
        frames = []
        for c in codes:
            df = self.frame_by_code.get(str(c).zfill(6))
            if df is None or df.empty:
                continue
            sub = df.copy()
            if start_date is not None:
                sub = sub[sub["date"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                sub = sub[sub["date"] <= pd.Timestamp(end_date)]
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        return pd.concat(frames, ignore_index=True)


def _make_daily(code: str, start: str, n: int, base: float = 10.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(int(code) + n)
    close = base + np.cumsum(rng.normal(0.0, 0.1, size=n))
    df = pd.DataFrame({
        "date": dates,
        "code": str(code).zfill(6),
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": rng.integers(1e5, 1e7, size=n).astype(float),
        "turnover": rng.uniform(1e7, 1e9, size=n),
        "change_pct": np.concatenate([[np.nan], np.diff(close) / close[:-1] * 100.0]),
    })
    return df


# ============================================================
# 1. 首次写入 + 增量
# ============================================================
def test_first_write_and_incremental(tmp_path: Path):
    pytest.importorskip("pyarrow")
    code = "600000"
    primary = MockFetcher(frame_by_code={
        code: _make_daily(code, "2024-01-02", 5),
    })
    mgr = DailyCacheManager(cache_dir=tmp_path, primary_fetcher=primary)

    r1 = mgr.update_code(code, "2024-01-01", "2024-01-08")
    assert r1.status == "ok"
    assert r1.rows_before == 0
    assert r1.rows_after == 5
    assert r1.rows_added == 5

    # 第二次：fetcher 没有新数据 → cached
    r2 = mgr.update_code(code, "2024-01-01", "2024-01-08")
    assert r2.status in ("cached", "empty")
    assert r2.rows_after == 5
    assert r2.rows_added == 0

    # 缓存读取
    df = mgr.load_code(code)
    assert len(df) == 5
    assert list(df.columns) == STANDARD_COLUMNS
    assert df["code"].nunique() == 1 and df["code"].iloc[0] == code


# ============================================================
# 2. 真正的增量（更长的 end_date 拉到新数据）
# ============================================================
def test_truly_incremental_adds_new_rows(tmp_path: Path):
    pytest.importorskip("pyarrow")
    code = "600000"
    primary = MockFetcher(frame_by_code={
        code: _make_daily(code, "2024-01-02", 10),  # 拥有 10 个交易日
    })
    mgr = DailyCacheManager(cache_dir=tmp_path, primary_fetcher=primary)

    # 第一次只拉前 5 天
    mgr.update_code(code, "2024-01-01", "2024-01-08")
    assert primary.calls[-1][1] == "2024-01-01"
    cached1 = mgr.load_code(code)
    assert len(cached1) == 5

    # 第二次拉到 1-15 → 应该增量补 5 条
    r = mgr.update_code(code, "2024-01-01", "2024-01-15")
    assert r.status == "ok"
    assert r.rows_before == 5
    assert r.rows_after == 10
    assert r.rows_added == 5
    # 第二次请求应当是从已缓存最末日 + 1 开始
    last_call_start = primary.calls[-1][1]
    assert last_call_start >= "2024-01-09"


# ============================================================
# 3. force_refresh
# ============================================================
def test_force_refresh_replaces_existing(tmp_path: Path):
    pytest.importorskip("pyarrow")
    code = "000001"
    full = _make_daily(code, "2024-01-02", 5)
    primary = MockFetcher(frame_by_code={code: full})
    mgr = DailyCacheManager(cache_dir=tmp_path, primary_fetcher=primary)

    mgr.update_code(code, "2024-01-01", "2024-01-08")
    assert mgr.load_code(code).shape[0] == 5

    # 修改 mock 数据后强制刷新
    new_frame = _make_daily(code, "2024-01-02", 3)
    primary.frame_by_code[code] = new_frame
    r = mgr.update_code(code, "2024-01-01", "2024-01-08", force_refresh=True)
    assert r.status == "ok"
    assert mgr.load_code(code).shape[0] == 3


# ============================================================
# 4. fallback：primary 空 → fallback 救场
# ============================================================
def test_fallback_when_primary_empty(tmp_path: Path):
    pytest.importorskip("pyarrow")
    code = "000002"
    primary = MockFetcher(frame_by_code={})   # 主源啥都没有
    fallback = MockFetcher(frame_by_code={code: _make_daily(code, "2024-01-02", 4)})
    mgr = DailyCacheManager(
        cache_dir=tmp_path,
        primary_fetcher=primary,
        fallback_fetcher=fallback,
    )
    r = mgr.update_code(code, "2024-01-01", "2024-01-08")
    assert r.status == "ok"
    assert r.source == "fallback"
    assert r.rows_after == 4

    # 关闭 fallback 时应该报错状态
    code2 = "000003"
    r2 = mgr.update_code(code2, "2024-01-01", "2024-01-08", allow_fallback=False)
    assert r2.status in ("error", "empty")


# ============================================================
# 5. load_range：跨 code 合并
# ============================================================
def test_load_range_merges_codes(tmp_path: Path):
    pytest.importorskip("pyarrow")
    primary = MockFetcher(frame_by_code={
        "600000": _make_daily("600000", "2024-01-02", 5),
        "000001": _make_daily("000001", "2024-01-02", 5),
    })
    mgr = DailyCacheManager(cache_dir=tmp_path, primary_fetcher=primary)
    mgr.update_batch(["600000", "000001"], "2024-01-01", "2024-01-08")
    df = mgr.load_range(["600000", "000001"], "2024-01-02", "2024-01-15")
    assert df["code"].nunique() == 2
    assert len(df) == 10
    # 排序：先按 code 再按 date
    assert (df.sort_values(["code", "date"]).reset_index(drop=True)
            .equals(df.reset_index(drop=True)))


# ============================================================
# 6. universe_from_zt_pool（用真实 ZtPoolLoader + parquet 文件）
# ============================================================
def test_universe_from_zt_pool(tmp_path: Path):
    pytest.importorskip("pyarrow")
    zt_dir = tmp_path / "zt_cache"
    (zt_dir / "zt_pool").mkdir(parents=True)
    (zt_dir / "zt_pool_previous").mkdir(parents=True)

    # 造两个交易日的 zt_pool parquet
    df1 = pd.DataFrame({
        "代码": ["600000", "300999", "002600"],
        "名称": ["a", "b", "c"],
        "涨跌幅": [10.0, 20.0, 10.0],
        "最新价": [10.0, 5.0, 12.0],
        "首次封板时间": ["093500", "100500", "094500"],
    })
    df1.to_parquet(zt_dir / "zt_pool" / "20240102.parquet", index=False)

    df2 = pd.DataFrame({
        "代码": ["600519", "688001"],
        "名称": ["d", "e"],
        "涨跌幅": [10.0, 20.0],
        "最新价": [1500.0, 80.0],
        "首次封板时间": ["093000", "094000"],
    })
    df2.to_parquet(zt_dir / "zt_pool" / "20240103.parquet", index=False)

    zt_loader = ZtPoolLoader(cache_dir=zt_dir, require_akshare=False)
    codes = DailyCacheManager.universe_from_zt_pool(
        zt_loader=zt_loader, start_date="20240101", end_date="20240110"
    )
    # 主板：600000、600519、002600；剔除 300999、688001
    assert codes == ["002600", "600000", "600519"]


# ============================================================
# 7. cache_summary
# ============================================================
def test_cache_summary(tmp_path: Path):
    pytest.importorskip("pyarrow")
    primary = MockFetcher(frame_by_code={
        "600000": _make_daily("600000", "2024-01-02", 5),
        "000001": _make_daily("000001", "2024-01-02", 3),
    })
    mgr = DailyCacheManager(cache_dir=tmp_path, primary_fetcher=primary)
    mgr.update_batch(["600000", "000001"], "2024-01-01", "2024-01-08")

    s = mgr.cache_summary()
    assert s["count"] == 2
    assert s["non_empty"] == 2
    assert s["earliest_date"] is not None
    assert s["latest_date"] is not None
