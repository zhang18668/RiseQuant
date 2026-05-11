"""DS-003 SectorStockSplitter 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.dataset.sector_split import SectorStockSplitter


def _make_df():
    return pd.DataFrame({
        "code": [f"00000{i % 5}" for i in range(50)],
        "sector": [f"sec_{i % 4}" for i in range(50)],
        "event_date": pd.bdate_range("2023-01-02", periods=50),
        "f_x": list(range(50)),
        "label_combined": [i % 3 for i in range(50)],
    })


class TestSectorStockSplitter:
    def test_split_by_time(self):
        sp = SectorStockSplitter(test_ratio=0.2, valid_ratio=0.1)
        out = sp.split_by_time(_make_df(), "event_date")
        total = len(out["train"]) + len(out["valid"]) + len(out["test"])
        assert total == 50

    def test_split_by_sector_no_overlap(self):
        sp = SectorStockSplitter(test_ratio=0.5, valid_ratio=0.0)
        out = sp.split_by_sector(_make_df(), "sector")
        train_sec = set(out["train"]["sector"])
        test_sec = set(out["test"]["sector"])
        assert train_sec.isdisjoint(test_sec)

    def test_split_by_stock_no_overlap(self):
        sp = SectorStockSplitter(test_ratio=0.4, valid_ratio=0.0)
        out = sp.split_by_stock(_make_df(), "code")
        train_codes = set(out["train"]["code"])
        test_codes = set(out["test"]["code"])
        assert train_codes.isdisjoint(test_codes)

    def test_split_by_sector_and_stock(self):
        sp = SectorStockSplitter(test_ratio=0.3, valid_ratio=0.0)
        out = sp.split_by_sector_and_stock(_make_df(), "sector", "code")
        assert "train" in out and "test" in out

    def test_cross_validate_by_sector(self):
        sp = SectorStockSplitter()
        folds = list(sp.cross_validate_by_sector(_make_df(), "sector", n_splits=4))
        assert len(folds) == 4
        for f in folds:
            assert "train" in f and "test" in f

    def test_cross_validate_by_stock(self):
        sp = SectorStockSplitter()
        folds = list(sp.cross_validate_by_stock(_make_df(), "code", n_splits=5))
        assert len(folds) == 5

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            SectorStockSplitter(test_ratio=0.6, valid_ratio=0.5)
