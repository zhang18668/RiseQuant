from __future__ import annotations

import pandas as pd
import pytest

from src.data.tushare_limit_pool import (
    normalize_limit_list_to_zt_pool,
    normalize_previous_pool,
)


def test_normalize_limit_list_keeps_market_value_in_yuan():
    raw = pd.DataFrame(
        {
            "trade_date": ["20200108"],
            "ts_code": ["000510.SZ"],
            "industry": ["化工"],
            "name": ["新金路"],
            "close": [4.72],
            "pct_chg": [10.02],
            "amount": [95_274_787],
            "float_mv": [2_593_542_000],
            "total_mv": [2_875_340_000],
            "turnover_ratio": [3.68],
            "fd_amount": [45_011_260_480],
            "first_time": ["93003"],
            "last_time": ["93003"],
            "open_times": [0],
            "up_stat": ["1/1"],
            "limit_times": [1],
            "limit": ["U"],
        }
    )

    out = normalize_limit_list_to_zt_pool(raw)

    assert out.loc[0, "代码"] == "000510"
    assert out.loc[0, "流通市值"] == pytest.approx(2_593_542_000)
    assert out.loc[0, "首次封板时间"] == "093003"


def test_normalize_previous_pool_uses_daily_basic_circ_mv_as_yuan():
    prev_limit = pd.DataFrame(
        {
            "ts_code": ["000700.SZ"],
            "industry": ["汽车"],
            "name": ["模塑科技"],
            "close": [4.65],
            "pct_chg": [9.9],
            "amount": [100_000_000],
            "float_mv": [3_335_032_000],
            "total_mv": [3_846_395_000],
            "turnover_ratio": [3.0],
            "fd_amount": [1_000_000],
            "first_time": ["093003"],
            "last_time": ["093003"],
            "open_times": [0],
            "up_stat": ["1/1"],
            "limit_times": [1],
            "limit": ["U"],
        }
    )
    daily = pd.DataFrame(
        {
            "ts_code": ["000700.SZ"],
            "trade_date": ["20200108"],
            "open": [4.90],
            "high": [5.12],
            "low": [4.98],
            "close": [5.12],
            "pre_close": [4.65],
            "pct_chg": [10.1075],
            "amount": [148_988.469],
        }
    )
    daily_basic = pd.DataFrame(
        {
            "ts_code": ["000700.SZ"],
            "trade_date": ["20200108"],
            "turnover_rate": [4.0659],
            "circ_mv": [367_212.1],
            "total_mv": [423_517.1],
        }
    )

    out = normalize_previous_pool("20200108", prev_limit, daily, daily_basic)

    assert out.loc[0, "代码"] == "000700"
    assert out.loc[0, "流通市值"] == pytest.approx(3_672_121_000)
    assert out.loc[0, "换手率"] == pytest.approx(4.0659)
