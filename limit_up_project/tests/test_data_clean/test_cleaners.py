"""Tests for DataCleaner + DataValidatorPro."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_clean import DataCleaner, DataValidatorPro


def _make_df(n_codes=3, n_days=30, has_st=False):
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for k in range(n_codes):
        code = f"00000{k}"
        for d in dates:
            rows.append({
                "date": d, "code": code,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0 + 0.01 * (d.day),
                "volume": 1e6,
                "is_st": has_st and k == 0,
            })
    return pd.DataFrame(rows)


def test_basic_clean_keeps_rows():
    df = _make_df()
    cleaned, rep = DataCleaner().clean(df)
    assert rep.n_output > 0
    assert "change_pct" in cleaned.columns
    assert cleaned["date"].dtype.kind == "M"


def test_st_dropped():
    df = _make_df(n_codes=3, n_days=10, has_st=True)
    cleaned, rep = DataCleaner(exclude_st=True).clean(df)
    assert rep.n_dropped_st > 0
    assert (cleaned["code"] != "000000").all()


def test_invalid_ohlc_dropped():
    df = _make_df()
    df.loc[0, "close"] = 0
    df.loc[1, "volume"] = np.nan
    cleaned, rep = DataCleaner().clean(df)
    assert rep.n_dropped_invalid_ohlc >= 1


def test_dup_dropped():
    df = _make_df()
    dup = df.head(1).copy()
    df = pd.concat([df, dup], ignore_index=True)
    cleaned, rep = DataCleaner().clean(df)
    assert rep.n_dropped_dup >= 1


def test_change_pct_recomputed():
    df = _make_df()
    cleaned, rep = DataCleaner().clean(df)
    # change_pct first row per code 应为 NaN
    first_each = cleaned.groupby("code").head(1)
    assert first_each["change_pct"].isna().all()


def test_empty_input():
    cleaned, rep = DataCleaner().clean(pd.DataFrame())
    assert cleaned.empty
    assert rep.n_input == 0


def test_missing_required_raises():
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=3)})
    with pytest.raises(ValueError):
        DataCleaner().clean(df)


def test_validator_ok():
    df = _make_df(n_codes=60, n_days=300)
    cleaned, _ = DataCleaner().clean(df)
    rep = DataValidatorPro(min_codes=10, min_total_rows=100).validate(cleaned)
    assert rep["ok"] is True
    assert rep["n_codes"] >= 60


def test_validator_fails_too_few_codes():
    df = _make_df(n_codes=2, n_days=5)
    cleaned, _ = DataCleaner().clean(df)
    rep = DataValidatorPro(min_codes=10, min_total_rows=100).validate(cleaned)
    assert rep["ok"] is False
