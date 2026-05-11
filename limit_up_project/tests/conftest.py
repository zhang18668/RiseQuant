"""pytest 全局夹具。"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保 ``import src.xxx`` 可以从项目根目录解析
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def calendar_dates():
    """从 2023-01-02 (周一) 开始连续 30 个交易日 (跨月)。"""
    return pd.bdate_range("2023-01-02", periods=30)


def _build_stock(code: str, dates, base: float, change_pcts) -> pd.DataFrame:
    closes = [base]
    for cp in change_pcts:
        closes.append(closes[-1] * (1 + cp / 100.0))
    closes = closes[1:]  # 与 change_pcts 长度一致
    df = pd.DataFrame({
        "date": dates,
        "code": code,
        "close": closes,
    })
    df["open"] = df["close"] * 0.99
    df["high"] = df["close"] * 1.01
    df["low"] = df["close"] * 0.98
    df["volume"] = 1_000_000.0
    df["turnover"] = df["close"] * df["volume"]
    df["change_pct"] = change_pcts
    df["is_st"] = False
    return df


@pytest.fixture
def sample_daily_data(calendar_dates) -> pd.DataFrame:
    """三只股票的合成日线数据，足够覆盖事件检测/特征/标签。"""
    n = len(calendar_dates)
    rng = np.random.default_rng(42)
    frames = []
    # 股票 000001: 第 5 天首板, 第 7 天二板, 之后涨幅 20% (完美标的)
    cp_a = rng.normal(0, 0.6, n).round(2)
    cp_a[5] = 9.95
    cp_a[7] = 9.95
    for i in range(8, 13):
        cp_a[i] = 3.0  # 主升浪
    frames.append(_build_stock("000001", calendar_dates, 10.0, cp_a))

    # 股票 000002: 第 10 天首板, 第 12 天二板, 主升浪不足
    cp_b = rng.normal(0, 0.5, n).round(2)
    cp_b[10] = 9.95
    cp_b[12] = 9.95
    for i in range(13, 18):
        cp_b[i] = 0.5  # 涨幅不到 15%
    frames.append(_build_stock("000002", calendar_dates, 12.0, cp_b))

    # 股票 000003: 仅有首板, 5 个交易日内未二板
    cp_c = rng.normal(0, 0.4, n).round(2)
    cp_c[15] = 9.95
    frames.append(_build_stock("000003", calendar_dates, 8.0, cp_c))

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def stock_a_data(sample_daily_data) -> pd.DataFrame:
    return sample_daily_data[sample_daily_data["code"] == "000001"].reset_index(drop=True)
