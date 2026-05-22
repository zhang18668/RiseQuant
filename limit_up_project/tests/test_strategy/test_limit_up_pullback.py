from __future__ import annotations

import pandas as pd
import pytest

from src.strategy.limit_up_pullback import (
    LimitUpPullbackParams,
    build_trade_ledger,
    select_limit_up_pullback,
)


def make_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    rows = [
        # base day
        ("600000", dates[0], 10.0, 10.2, 9.8, 10.0, 1000, 2.0),
        # T-2 limit-up: close/high > previous close * 1.095
        ("600000", dates[1], 10.5, 11.0, 10.4, 11.0, 1000, 4.0),
        # T-1 wash: higher volume, upper/lower shadows, turnover 5
        ("600000", dates[2], 10.8, 11.4, 10.5, 10.9, 1500, 5.0),
        # T confirm: close above wash close/open, moderate return/high return
        ("600000", dates[3], 11.0, 11.6, 10.9, 11.2, 1200, 4.0),
        ("600000", dates[4], 11.2, 11.6, 11.0, 11.1, 1000, 4.0),
        ("600000", dates[5], 11.1, 11.2, 11.0, 11.1, 1000, 4.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["code", "date", "open", "high", "low", "close", "volume", "turnover_rate"],
    )


def make_market() -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-25", periods=9)
    closes = [2950, 2960, 2970, 2980, 2990, 3000, 3010, 3020, 3030]
    return pd.DataFrame({"date": dates, "close": closes, "volume": 1_000_000})


def test_select_limit_up_pullback_matches_tdx_formula_offsets():
    signals = select_limit_up_pullback(make_frame(), LimitUpPullbackParams(), market_data=make_market())

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["date"] == pd.Timestamp("2024-01-04")
    assert row["limit_date"] == pd.Timestamp("2024-01-02")
    assert row["wash_date"] == pd.Timestamp("2024-01-03")
    assert row["code"] == "600000"
    assert bool(row["label_3"]) is True


def test_select_limit_up_pullback_requires_turnover_when_configured():
    df = make_frame().drop(columns=["turnover_rate"])

    with pytest.raises(ValueError, match="turnover_rate"):
        select_limit_up_pullback(df, LimitUpPullbackParams(require_turnover_rate=True), market_data=make_market())

    signals = select_limit_up_pullback(
        df,
        LimitUpPullbackParams(require_turnover_rate=False),
        market_data=make_market(),
    )
    assert len(signals) == 1


def test_select_limit_up_pullback_filters_confirm_limit_like_day():
    df = make_frame()
    df.loc[df["date"] == pd.Timestamp("2024-01-04"), "high"] = 12.2

    signals = select_limit_up_pullback(df, LimitUpPullbackParams(), market_data=make_market())

    assert signals.empty


def test_select_limit_up_pullback_uses_market_cap_turnover_tiers():
    df = make_frame()
    df["float_market_cap"] = 4_000_000_000.0
    # Small-cap bucket needs 5%-20%, so 4% should be rejected.
    df.loc[df["date"] == pd.Timestamp("2024-01-03"), "turnover_rate"] = 4.0

    signals = select_limit_up_pullback(df, LimitUpPullbackParams(), market_data=make_market())

    assert signals.empty


def test_select_limit_up_pullback_requires_market_when_configured():
    with pytest.raises(ValueError, match="market_data"):
        select_limit_up_pullback(make_frame(), LimitUpPullbackParams(require_market_filter=True))


def test_build_trade_ledger_has_buy_sell_and_pnl():
    signals = select_limit_up_pullback(make_frame(), LimitUpPullbackParams(), market_data=make_market())

    trades = build_trade_ledger(signals)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["buy_date"] == pd.Timestamp("2024-01-04")
    assert row["sell_date"] == pd.Timestamp("2024-01-05")
    assert row["code"] == "600000"
    assert row["buy_price"] == pytest.approx(11.2)
    assert row["sell_price"] == pytest.approx(11.1)
    assert row["gross_pnl"] == pytest.approx(-0.1)
    assert row["return_pct"] == pytest.approx((11.1 / 11.2 - 1.0) * 100.0)
    assert bool(row["target_hit"]) is True
