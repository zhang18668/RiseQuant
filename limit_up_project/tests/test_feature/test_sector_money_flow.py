"""F-003/F-004 sentiment & money flow 测试。"""
from __future__ import annotations

import pandas as pd

from src.feature.money_flow import MoneyFlowFeatures
from src.feature.sector_sentiment import SectorSentimentFeatures


def test_market_sentiment(sample_daily_data):
    out = SectorSentimentFeatures().market_sentiment(sample_daily_data)
    assert "f_mkt_up_ratio" in out.columns
    assert "f_mkt_limit_up_count" in out.columns
    # 比例在 [0, 1]
    assert (out["f_mkt_up_ratio"] >= 0).all() and (out["f_mkt_up_ratio"] <= 1).all()


def test_rolling_limit_up_count(sample_daily_data):
    senti = SectorSentimentFeatures().market_sentiment(sample_daily_data)
    rolled = SectorSentimentFeatures().rolling_limit_up_count(senti, windows=[3])
    assert "f_mkt_lu_ma_3" in rolled.columns


def test_money_flow(sample_daily_data):
    out = MoneyFlowFeatures().rolling_net_amount(sample_daily_data, windows=[3])
    assert "f_mf_net_3" in out.columns
