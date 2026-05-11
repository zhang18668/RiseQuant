"""U-004 DataValidator 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.validator import DataValidator, ValidationError


class TestDataValidator:
    def test_required_columns_ok(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ok, missing = DataValidator.check_required_columns(df, ["a", "b"])
        assert ok and missing == []

    def test_required_columns_missing(self):
        df = pd.DataFrame({"a": [1]})
        ok, missing = DataValidator.check_required_columns(df, ["a", "b"])
        assert not ok and missing == ["b"]

    def test_required_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValidationError):
            DataValidator.check_required_columns(df, ["x"], raise_error=True)

    def test_check_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        has_dup, n = DataValidator.check_duplicates(df, subset=["a"])
        assert has_dup and n == 1

    def test_missing_ratio(self):
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        ratios = DataValidator.check_missing_ratio(df)
        assert ratios["a"] == pytest.approx(0.5)

    def test_validate_daily_data_happy(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=5),
            "code": ["000001"] * 5,
            "open": [10.0] * 5,
            "high": [10.5] * 5,
            "low": [9.5] * 5,
            "close": [10.0] * 5,
            "volume": [1000] * 5,
        })
        res = DataValidator.validate_daily_data(df)
        assert res["is_valid"]

    def test_validate_daily_data_bad_high_low(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=2),
            "code": ["000001"] * 2,
            "open": [10.0, 10.0],
            "high": [9.0, 10.0],   # high < low for first row
            "low": [10.0, 9.0],
            "close": [10.0, 10.0],
            "volume": [1000, 1000],
        })
        res = DataValidator.validate_daily_data(df)
        assert not res["is_valid"]
