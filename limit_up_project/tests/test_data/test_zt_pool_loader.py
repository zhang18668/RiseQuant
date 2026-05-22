from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.zt_pool_loader import (
    ZT_POOL_TYPE,
    ZtPoolLoader,
    assert_zt_pool_cache_quality,
    zt_pool_cache_quality,
)


def test_fetch_one_does_not_overwrite_non_empty_cache_with_empty_api(monkeypatch, tmp_path: Path):
    pytest.importorskip("pyarrow")
    loader = ZtPoolLoader(cache_dir=tmp_path, require_akshare=False)
    cache_path = tmp_path / ZT_POOL_TYPE / "20240102.parquet"
    original = pd.DataFrame({"代码": ["600000"], "名称": ["A"]})
    original.to_parquet(cache_path, index=False)

    monkeypatch.setattr(loader, "_api_for", lambda pool_type: (lambda date: pd.DataFrame()))

    result = loader.fetch_one(
        "20240102",
        pool_type=ZT_POOL_TYPE,
        use_cache=False,
        save_cache=True,
        protect_non_empty_cache=True,
    )

    landed = pd.read_parquet(cache_path)
    assert result.equals(original)
    assert landed.equals(original)


def test_refresh_empty_cache_refetches_placeholder(monkeypatch, tmp_path: Path):
    pytest.importorskip("pyarrow")
    loader = ZtPoolLoader(cache_dir=tmp_path, require_akshare=False)
    cache_path = tmp_path / ZT_POOL_TYPE / "20240102.parquet"
    pd.DataFrame().to_parquet(cache_path, index=False)
    fetched = pd.DataFrame({"代码": ["600001"], "名称": ["B"]})

    monkeypatch.setattr(loader, "_api_for", lambda pool_type: (lambda date: fetched))

    result = loader.fetch_one(
        "20240102",
        pool_type=ZT_POOL_TYPE,
        use_cache=True,
        refresh_empty_cache=True,
        save_cache=True,
    )

    assert result.equals(fetched)
    assert pd.read_parquet(cache_path).equals(fetched)


def test_zt_pool_cache_quality_monthly_assertion(tmp_path: Path):
    pytest.importorskip("pyarrow")
    loader = ZtPoolLoader(cache_dir=tmp_path, require_akshare=False)
    workdays = pd.bdate_range("2024-01-01", "2024-01-31")
    for d in workdays[:18]:
        path = tmp_path / ZT_POOL_TYPE / f"{d:%Y%m%d}.parquet"
        pd.DataFrame({"代码": ["600000"]}).to_parquet(path, index=False)
    for d in workdays[18:]:
        path = tmp_path / ZT_POOL_TYPE / f"{d:%Y%m%d}.parquet"
        pd.DataFrame().to_parquet(path, index=False)

    summary = assert_zt_pool_cache_quality(
        tmp_path,
        "20240101",
        "20240131",
        pool_type=ZT_POOL_TYPE,
        min_monthly_non_empty=18,
    )

    assert summary["ok"] is True
    assert summary["non_empty"] == 18

    bad = zt_pool_cache_quality(
        tmp_path,
        "20240101",
        "20240131",
        pool_type=ZT_POOL_TYPE,
        min_monthly_non_empty=19,
    )
    assert bad["ok"] is False
    assert bad["bad_months"][0]["month"] == "202401"
