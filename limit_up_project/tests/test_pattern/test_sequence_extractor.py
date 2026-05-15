"""单测: SequenceExtractor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pattern.sequence_extractor import SequenceExtractor, LOOKBACK, N_CHANNELS


def _make_daily(code: str, n: int, base: float = 10.0):
    dates = pd.bdate_range("2024-01-02", periods=n)
    closes = [base + 0.05 * i for i in range(n)]
    df = pd.DataFrame({
        "date": dates,
        "code": code,
        "open":  [c * 0.99 for c in closes],
        "high":  [c * 1.02 for c in closes],
        "low":   [c * 0.98 for c in closes],
        "close": closes,
        "volume": [1_000_000.0 + i * 1000 for i in range(n)],
    })
    return df


def test_shape_is_lookback_x_channels():
    df = _make_daily("A", n=60)
    first_d = df["date"].iloc[20]
    pot_d = df["date"].iloc[40]
    seq = SequenceExtractor().extract(df, first_d, pot_d)
    assert seq is not None
    assert seq.shape == (LOOKBACK, N_CHANNELS)
    assert seq.dtype == np.float32


def test_no_padding_when_full_history():
    df = _make_daily("A", n=80)
    first_d = df["date"].iloc[30]
    pot_d = df["date"].iloc[50]
    seq = SequenceExtractor().extract(df, first_d, pot_d)
    assert seq is not None
    # is_padded 通道全 0
    assert np.all(seq[:, 9] == 0.0)


def test_padding_when_insufficient_history():
    # pot_idx 距 0 不足 20 日 -> 应 padding
    df = _make_daily("A", n=20)
    first_d = df["date"].iloc[2]
    pot_d = df["date"].iloc[10]   # 距开头只有 11 根 K 线 (idx 0..10)
    seq = SequenceExtractor().extract(df, first_d, pot_d)
    assert seq is not None
    # 应该 padding 9 行 (20 - 11)
    assert np.sum(seq[:, 9] == 1.0) == 9
    assert np.sum(seq[:, 9] == 0.0) == 11
    # padding 部分其他通道全 0
    assert np.all(seq[:9, :9] == 0.0)


def test_close_relative_to_first_close():
    df = _make_daily("A", n=60)
    first_d = df["date"].iloc[10]
    pot_d = df["date"].iloc[30]
    first_close = float(df["close"].iloc[10])
    pot_close = float(df["close"].iloc[30])
    seq = SequenceExtractor().extract(df, first_d, pot_d)
    # 最后一根 (pot 当日) 的 channel 0 = pot_close / first_close - 1
    expected = pot_close / first_close - 1.0
    assert seq[-1, 0] == pytest.approx(expected, abs=1e-5)


def test_days_from_first_normalization():
    df = _make_daily("A", n=60)
    first_d = df["date"].iloc[10]
    pot_d = df["date"].iloc[25]   # pot 距 first 15 天
    seq = SequenceExtractor(days_norm=30.0).extract(df, first_d, pot_d)
    # 最后一根的 channel 7 = 15 / 30 = 0.5
    assert seq[-1, 7] == pytest.approx(0.5, abs=1e-5)


def test_volume_centered_around_zero():
    df = _make_daily("A", n=60)
    first_d = df["date"].iloc[20]
    pot_d = df["date"].iloc[35]
    seq = SequenceExtractor().extract(df, first_d, pot_d)
    # volume 通道 (3) 的窗口平均应该接近 0 (因为是 v/baseline - 1 形式, 量持续上涨所以会略大于 0)
    # 至少不应该全是大数
    assert np.abs(seq[-5:, 3]).max() < 5.0


def test_missing_first_date_returns_none():
    df = _make_daily("A", n=30)
    seq = SequenceExtractor().extract(df, pd.Timestamp("2099-01-01"), df["date"].iloc[10])
    assert seq is None


def test_missing_pot_date_returns_none():
    df = _make_daily("A", n=30)
    seq = SequenceExtractor().extract(df, df["date"].iloc[5], pd.Timestamp("2099-01-01"))
    assert seq is None


def test_pot_before_first_returns_none():
    df = _make_daily("A", n=30)
    seq = SequenceExtractor().extract(df, df["date"].iloc[10], df["date"].iloc[5])
    assert seq is None


def test_empty_df_returns_none():
    seq = SequenceExtractor().extract(pd.DataFrame(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    assert seq is None


def test_extract_batch():
    df1 = _make_daily("A", n=60).assign(code="A")
    df2 = _make_daily("B", n=60, base=20.0).assign(code="B")
    daily_by_code = {"A": df1, "B": df2}
    first_a = df1["date"].iloc[10]
    pot_a = df1["date"].iloc[30]
    first_b = df2["date"].iloc[15]
    pot_b = df2["date"].iloc[35]
    records = pd.DataFrame([
        {"code": "A", "first_date": first_a, "potential_date": pot_a, "sample_id": "s1"},
        {"code": "B", "first_date": first_b, "potential_date": pot_b, "sample_id": "s2"},
        {"code": "C", "first_date": first_a, "potential_date": pot_a, "sample_id": "s3"},  # 缺 code
    ])
    out = SequenceExtractor().extract_batch(daily_by_code, records)
    assert "s1" in out and "s2" in out
    assert "s3" not in out
    assert out["s1"].shape == (LOOKBACK, N_CHANNELS)


def test_no_nan_no_inf_in_output():
    df = _make_daily("A", n=60)
    seq = SequenceExtractor().extract(df, df["date"].iloc[20], df["date"].iloc[40])
    assert np.isfinite(seq).all()
