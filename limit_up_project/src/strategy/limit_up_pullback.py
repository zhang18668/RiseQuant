"""Limit-up pullback selector.

Rule shape:

T-2: main-board limit-up day.
T-1: volume expansion, intraday shadows, turnover checked by float-market-cap tier.
T: close stands above T-1 close/open, while the day is still below chase levels.

The generated signal date is T.  The first research label uses the T close as
the tail-entry proxy and checks whether T+1 high is more than 3% above it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.zt_pool_loader import compute_seal_strength, is_main_board, time_tier, time_tier_rank


@dataclass(frozen=True)
class MarketCapTurnoverTier:
    """Turnover bounds for a float-market-cap bucket.

    Market cap values are expected in yuan, matching AkShare/Eastmoney zt-pool
    fields such as ``流通市值``.
    """

    upper: float | None
    turnover_min: float
    turnover_max: float
    name: str


DEFAULT_MARKET_CAP_TURNOVER_TIERS: tuple[MarketCapTurnoverTier, ...] = (
    MarketCapTurnoverTier(5_000_000_000, 5.0, 20.0, "lt_50e"),
    MarketCapTurnoverTier(10_000_000_000, 4.0, 15.0, "50e_100e"),
    MarketCapTurnoverTier(30_000_000_000, 3.0, 10.0, "100e_300e"),
    MarketCapTurnoverTier(None, 1.5, 8.0, "gte_300e"),
)


@dataclass(frozen=True)
class LimitUpPullbackParams:
    limit_return_min: float = 0.095
    confirm_return_max: float = 0.05
    confirm_high_return_max: float = 0.08
    turnover_min: float = 3.0
    turnover_max: float = 10.0
    require_turnover_rate: bool = True
    use_market_cap_turnover_tiers: bool = True
    require_shadow: bool = True
    volume_ratio_min: float = 1.0
    limit_gap_days: int = 2
    close_eq_high_tol: float = 0.001
    next_high_target: float = 0.03
    require_market_filter: bool = True
    market_ma_window: int = 5
    market_ret_min: float = -0.015

    def to_dict(self) -> dict:
        return asdict(self)


def _turnover_rate_col(df: pd.DataFrame) -> str | None:
    for col in ("turnover_rate", "换手率", "turnoverRatio"):
        if col in df.columns:
            return col
    return None


def _market_cap_col(df: pd.DataFrame) -> str | None:
    for col in ("float_market_cap", "流通市值", "market_value", "float_mv"):
        if col in df.columns:
            return col
    return None


def _stock_name_col(df: pd.DataFrame) -> str | None:
    for col in ("name", "名称", "stock_name"):
        if col in df.columns:
            return col
    return None


def _amount_col(df: pd.DataFrame) -> str | None:
    for col in ("amount", "turnover", "成交额"):
        if col in df.columns:
            return col
    return None


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _score(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    today_strength = frame["close"] / prev_close - 1.0
    vol_ratio = frame["volume"].shift(1) / frame["volume"].shift(2).replace(0, np.nan)
    return (today_strength.fillna(0.0) * 100.0 + vol_ratio.fillna(1.0)).astype(float)


def _turnover_tier_bounds(
    market_cap: pd.Series,
    tiers: Sequence[MarketCapTurnoverTier] = DEFAULT_MARKET_CAP_TURNOVER_TIERS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    cap = pd.to_numeric(market_cap, errors="coerce")
    min_s = pd.Series(np.nan, index=cap.index, dtype=float)
    max_s = pd.Series(np.nan, index=cap.index, dtype=float)
    name_s = pd.Series(pd.NA, index=cap.index, dtype="object")
    lower = -np.inf
    for tier in tiers:
        if tier.upper is None:
            mask = cap >= lower
        else:
            mask = (cap >= lower) & (cap < tier.upper)
        min_s.loc[mask] = tier.turnover_min
        max_s.loc[mask] = tier.turnover_max
        name_s.loc[mask] = tier.name
        lower = tier.upper if tier.upper is not None else lower
    return min_s, max_s, name_s


def _market_filter_by_date(market_data: pd.DataFrame, params: LimitUpPullbackParams) -> pd.DataFrame:
    required = {"date", "close"}
    missing = required - set(market_data.columns)
    if missing:
        raise ValueError(f"market_data missing required columns: {sorted(missing)}")

    out = market_data.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="coerce")
    out["market_ma"] = close.rolling(params.market_ma_window, min_periods=params.market_ma_window).mean()
    out["market_return"] = close.pct_change(fill_method=None)
    out["market_filter_ok"] = (
        (close > out["market_ma"])
        & (out["market_return"] > params.market_ret_min)
    )
    return out[["date", "market_ma", "market_return", "market_filter_ok"]]


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date", "code", "score", "limit_date", "wash_date",
            "wash_turnover_rate", "wash_turnover_tier", "wash_turnover_min",
            "wash_turnover_max", "wash_float_market_cap", "confirm_return",
            "confirm_high_return", "volume_ratio", "buy_price",
            "next_date", "next_high_return", "next_close_return",
            "next_low_return", "next_open_return", "label_3",
            "limit_day_return", "limit_day_open_return", "limit_day_amount",
            "limit_day_turnover_rate", "limit_day_vol_ratio_5",
            "limit_day_vol_ratio_10", "limit_day_vol_ratio_20",
            "limit_day_amp", "is_yizi", "market_filter_ok",
            "market_return", "market_ma",
        ]
    )


def select_limit_up_pullback(
    daily: pd.DataFrame,
    params: LimitUpPullbackParams | None = None,
    codes: Iterable[str] | None = None,
    market_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return signal rows with diagnostic factors and T+1 labels.

    ``daily`` may already contain AkShare zt-pool columns such as ``流通市值``,
    ``换手率``, ``首次封板时间`` and ``封板资金``.  If those columns are absent,
    the selector still works when turnover filtering is disabled; otherwise it
    raises a clear error.
    """
    params = params or LimitUpPullbackParams()
    if daily is None or daily.empty:
        return _empty_signal_frame()

    required = {"date", "code", "open", "high", "low", "close", "volume"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily missing required columns: {sorted(missing)}")

    tr_col = _turnover_rate_col(daily)
    if params.require_turnover_rate and tr_col is None:
        raise ValueError(
            "turnover_rate/换手率 column is required by this strategy. "
            "Merge AkShare zt_pool_previous data, pass require_turnover_rate=False, "
            "or enrich daily_cache first."
        )

    if params.require_market_filter and (market_data is None or market_data.empty):
        raise ValueError(
            "market_data is required when require_market_filter=True. "
            "Pass Shanghai index data or set require_market_filter=False."
        )

    market_by_date = None
    if market_data is not None and not market_data.empty:
        market_by_date = _market_filter_by_date(market_data, params).set_index("date")

    cap_col = _market_cap_col(daily)
    name_col = _stock_name_col(daily)
    amount_col = _amount_col(daily)
    use_codes = {str(c).zfill(6) for c in codes} if codes else None
    frames = []
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["code"] = data["code"].astype(str).str.zfill(6)
    if use_codes is not None:
        data = data[data["code"].isin(use_codes)]

    for code, g in data.groupby("code", sort=False):
        if not is_main_board(code):
            continue
        g = g.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        prev_close = g["close"].shift(1)
        limit_day_return = g["close"] / prev_close - 1.0
        limit_up = (
            (limit_day_return > params.limit_return_min)
            & (g["close"] >= g["high"] * (1.0 - params.close_eq_high_tol))
        )

        k = int(params.limit_gap_days)
        limit_day = limit_up.shift(k).fillna(False)
        wash_idx = k - 1

        wash_volume = g["volume"].shift(wash_idx)
        limit_volume = g["volume"].shift(k)
        wash_open = g["open"].shift(wash_idx)
        wash_high = g["high"].shift(wash_idx)
        wash_low = g["low"].shift(wash_idx)
        wash_close = g["close"].shift(wash_idx)

        volume_ratio = wash_volume / limit_volume.replace(0, np.nan)
        volume_ok = volume_ratio > params.volume_ratio_min
        if params.require_shadow:
            shadow_ok = (
                (wash_high > wash_close)
                & (wash_high > wash_open)
                & (wash_low < wash_open)
                & (wash_low < wash_close)
            )
        else:
            shadow_ok = pd.Series(True, index=g.index)

        if tr_col is not None:
            turnover = pd.to_numeric(g[tr_col].shift(wash_idx), errors="coerce")
            if params.use_market_cap_turnover_tiers and cap_col is not None:
                wash_cap = pd.to_numeric(g[cap_col].shift(wash_idx), errors="coerce")
                tier_min, tier_max, tier_name = _turnover_tier_bounds(wash_cap)
                fixed_ok = turnover.between(params.turnover_min, params.turnover_max, inclusive="both")
                tier_ok = turnover.between(tier_min, tier_max, inclusive="both")
                turnover_ok = tier_ok.where(wash_cap.notna(), fixed_ok)
                tier_min = tier_min.fillna(params.turnover_min)
                tier_max = tier_max.fillna(params.turnover_max)
                tier_name = tier_name.fillna("fixed")
            else:
                wash_cap = pd.Series(np.nan, index=g.index)
                tier_min = pd.Series(params.turnover_min, index=g.index, dtype=float)
                tier_max = pd.Series(params.turnover_max, index=g.index, dtype=float)
                tier_name = pd.Series("fixed", index=g.index, dtype="object")
                turnover_ok = turnover.between(params.turnover_min, params.turnover_max, inclusive="both")
        else:
            turnover = pd.Series(np.nan, index=g.index)
            wash_cap = pd.Series(np.nan, index=g.index)
            tier_min = pd.Series(np.nan, index=g.index)
            tier_max = pd.Series(np.nan, index=g.index)
            tier_name = pd.Series(pd.NA, index=g.index, dtype="object")
            turnover_ok = pd.Series(True, index=g.index)

        confirm_return = g["close"] / prev_close - 1.0
        confirm_high_return = g["high"] / prev_close - 1.0
        stand_ok = (
            (g["close"] > wash_close)
            & (g["close"] > wash_open)
            & (confirm_return < params.confirm_return_max)
            & (confirm_high_return < params.confirm_high_return_max)
        )

        market_ok = pd.Series(True, index=g.index)
        market_return = pd.Series(np.nan, index=g.index)
        market_ma = pd.Series(np.nan, index=g.index)
        if market_by_date is not None:
            joined = g[["date"]].join(market_by_date, on="date")
            market_return = joined["market_return"]
            market_ma = joined["market_ma"]
            market_ok = joined["market_filter_ok"].fillna(False)
            if not params.require_market_filter:
                market_ok = pd.Series(True, index=g.index)

        mask = limit_day & volume_ok & shadow_ok & turnover_ok & stand_ok & market_ok
        if not mask.any():
            continue

        next_open = g["open"].shift(-1)
        next_high = g["high"].shift(-1)
        next_low = g["low"].shift(-1)
        next_close = g["close"].shift(-1)
        buy_price = g["close"]

        out_cols = ["date", "code", "open", "high", "low", "close", "volume"]
        if name_col is not None:
            out_cols.append(name_col)
        out = g.loc[mask, out_cols].copy()
        if name_col is not None and name_col != "name":
            out = out.rename(columns={name_col: "name"})

        out["score"] = _score(g).loc[mask].values
        out["limit_date"] = g["date"].shift(k).loc[mask].values
        out["wash_date"] = g["date"].shift(wash_idx).loc[mask].values
        out["wash_turnover_rate"] = turnover.loc[mask].values
        out["wash_turnover_tier"] = tier_name.loc[mask].values
        out["wash_turnover_min"] = tier_min.loc[mask].values
        out["wash_turnover_max"] = tier_max.loc[mask].values
        out["wash_float_market_cap"] = wash_cap.loc[mask].values
        out["confirm_return"] = confirm_return.loc[mask].values
        out["confirm_high_return"] = confirm_high_return.loc[mask].values
        out["volume_ratio"] = volume_ratio.loc[mask].values
        out["buy_price"] = buy_price.loc[mask].values
        out["next_date"] = g["date"].shift(-1).loc[mask].values
        out["next_high_return"] = (next_high / buy_price - 1.0).loc[mask].values
        out["next_close_return"] = (next_close / buy_price - 1.0).loc[mask].values
        out["next_low_return"] = (next_low / buy_price - 1.0).loc[mask].values
        out["next_open_return"] = (next_open / buy_price - 1.0).loc[mask].values
        out["label_3"] = out["next_high_return"] > params.next_high_target

        limit_open = g["open"].shift(k)
        limit_high = g["high"].shift(k)
        limit_low = g["low"].shift(k)
        limit_close = g["close"].shift(k)
        out["limit_day_return"] = limit_day_return.shift(k).loc[mask].values
        out["limit_day_open_return"] = (limit_open / g["close"].shift(k + 1) - 1.0).loc[mask].values
        if amount_col is not None:
            out["limit_day_amount"] = g[amount_col].shift(k).loc[mask].values
        else:
            out["limit_day_amount"] = np.nan
        if tr_col is not None:
            out["limit_day_turnover_rate"] = g[tr_col].shift(k).loc[mask].values
        else:
            out["limit_day_turnover_rate"] = np.nan
        out["limit_day_vol_ratio_5"] = _safe_ratio(g["volume"], g["volume"].rolling(5).mean()).shift(k).loc[mask].values
        out["limit_day_vol_ratio_10"] = _safe_ratio(g["volume"], g["volume"].rolling(10).mean()).shift(k).loc[mask].values
        out["limit_day_vol_ratio_20"] = _safe_ratio(g["volume"], g["volume"].rolling(20).mean()).shift(k).loc[mask].values
        out["limit_day_amp"] = (limit_high / limit_low - 1.0).loc[mask].values
        out["is_yizi"] = ((limit_open == limit_high) & (limit_low == limit_high)).loc[mask].values
        out["market_filter_ok"] = market_ok.loc[mask].values
        out["market_return"] = market_return.loc[mask].values
        out["market_ma"] = market_ma.loc[mask].values
        for extra_col in [c for c in g.columns if c.startswith("limit_pool_") or c == "prev_limit_time"]:
            out[extra_col] = g[extra_col].loc[mask].values

        frames.append(out)

    if not frames:
        return _empty_signal_frame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "score"], ascending=[True, False])


def normalize_zt_pool(pool: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    """Normalize AkShare zt-pool/previous-pool fields for joining with daily bars."""
    if pool is None or pool.empty:
        return pd.DataFrame(columns=["date", "code"])
    out = pool.copy()
    if date_col not in out.columns:
        if "date" not in out.columns:
            raise ValueError(f"zt pool missing date column: {date_col!r}")
        date_col = "date"
    rename = {
        date_col: "date",
        "代码": "code",
        "名称": "name",
        "换手率": "turnover_rate",
        "流通市值": "float_market_cap",
        "总市值": "total_market_cap",
        "成交额": "amount",
        "封板资金": "seal_amount",
        "首次封板时间": "first_limit_time",
        "最后封板时间": "last_limit_time",
        "炸板次数": "open_board_count",
        "昨日封板时间": "prev_limit_time",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "code" not in out.columns:
        return pd.DataFrame(columns=["date", "code"])
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str).str.zfill(6)
    if "first_limit_time" in out.columns:
        out["first_limit_time_tier"] = out["first_limit_time"].apply(time_tier)
        out["first_limit_time_tier_rank"] = out["first_limit_time"].apply(time_tier_rank)
    if "seal_amount" in out.columns and "float_market_cap" in out.columns:
        out["seal_strength_pct"] = [
            compute_seal_strength(seal, cap)
            for seal, cap in zip(out["seal_amount"], out["float_market_cap"])
        ]
    keep = [
        "date", "code", "name", "turnover_rate", "float_market_cap",
        "total_market_cap", "amount", "seal_amount", "first_limit_time",
        "last_limit_time", "open_board_count", "prev_limit_time",
        "first_limit_time_tier", "first_limit_time_tier_rank", "seal_strength_pct",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def merge_zt_pool_fields(
    daily: pd.DataFrame,
    zt_pool: pd.DataFrame | None = None,
    zt_pool_previous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge AkShare zt-pool fields into daily bars by date/code.

    ``zt_pool`` enriches T-2 limit-day factors; ``zt_pool_previous`` enriches
    the next day after a limit-up, which is exactly T-1 in this rule.
    """
    if daily is None or daily.empty:
        return daily
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str).str.zfill(6)

    if zt_pool is not None and not zt_pool.empty:
        limit_pool = normalize_zt_pool(zt_pool)
        limit_cols = [c for c in limit_pool.columns if c not in {"date", "code"}]
        limit_pool = limit_pool.rename(columns={c: f"limit_pool_{c}" for c in limit_cols})
        out = out.merge(limit_pool, on=["date", "code"], how="left")

    if zt_pool_previous is not None and not zt_pool_previous.empty:
        prev_pool = normalize_zt_pool(zt_pool_previous)
        use = [
            "date", "code", "name", "turnover_rate", "float_market_cap",
            "total_market_cap", "amount", "prev_limit_time",
        ]
        prev_pool = prev_pool[[c for c in use if c in prev_pool.columns]]
        out = out.merge(prev_pool, on=["date", "code"], how="left", suffixes=("", "_zt_prev"))

        for col in ("turnover_rate", "float_market_cap", "total_market_cap", "amount", "name"):
            alt = f"{col}_zt_prev"
            if alt in out.columns:
                if col in out.columns:
                    out[col] = out[col].combine_first(out[alt])
                    out = out.drop(columns=[alt])
                else:
                    out = out.rename(columns={alt: col})

    return out


def factor_group_stats(samples: pd.DataFrame, target_col: str = "label_3") -> pd.DataFrame:
    """Build compact factor-bucket stats for the first research pass."""
    if samples is None or samples.empty:
        return pd.DataFrame(columns=["factor", "bucket", "count", "hit_rate", "avg_next_high_return"])

    rows = []

    def add_group(factor: str, bucket: pd.Series) -> None:
        work = samples.copy()
        work["_bucket"] = bucket.astype("object")
        grouped = work.dropna(subset=["_bucket"]).groupby("_bucket", dropna=False)
        for key, g in grouped:
            rows.append(
                {
                    "factor": factor,
                    "bucket": str(key),
                    "count": int(len(g)),
                    "hit_rate": float(g[target_col].mean()) if target_col in g else np.nan,
                    "avg_next_high_return": float(g["next_high_return"].mean()),
                    "avg_next_close_return": float(g["next_close_return"].mean()),
                    "avg_next_low_return": float(g["next_low_return"].mean()),
                }
            )

    add_group("is_yizi", samples["is_yizi"].map({True: "yizi", False: "non_yizi"}))
    add_group("wash_turnover_tier", samples["wash_turnover_tier"])
    add_group(
        "limit_day_vol_ratio_5",
        pd.cut(samples["limit_day_vol_ratio_5"], [-np.inf, 1.0, 1.5, 2.5, 4.0, np.inf]),
    )
    add_group(
        "volume_ratio",
        pd.cut(samples["volume_ratio"], [1.0, 1.5, 2.0, 3.0, 5.0, np.inf]),
    )
    if "limit_pool_first_limit_time_tier" in samples.columns:
        add_group("first_limit_time_tier", samples["limit_pool_first_limit_time_tier"])
    if "limit_pool_open_board_count" in samples.columns:
        add_group("open_board_count", samples["limit_pool_open_board_count"].fillna("unknown"))
    return pd.DataFrame(rows).sort_values(["factor", "bucket"]).reset_index(drop=True)


def build_trade_ledger(samples: pd.DataFrame, target_return: float = 0.03) -> pd.DataFrame:
    """Convert research samples into an explicit buy/sell ledger.

    The first executable research assumption is:
    - buy at T close, recorded as ``buy_date`` / ``buy_price``;
    - sell at T+1 close, recorded as ``sell_date`` / ``sell_price``;
    - also record whether T+1 high touched the target-return price.
    """
    columns = [
        "code", "name", "buy_date", "buy_price", "sell_date", "sell_price",
        "shares", "gross_pnl", "return_pct", "is_profit",
        "target_price", "target_hit", "target_high_return_pct",
        "next_open_return_pct", "next_low_return_pct",
        "limit_date", "wash_date",
    ]
    if samples is None or samples.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    out["code"] = samples["code"].astype(str).str.zfill(6)
    out["name"] = samples["name"] if "name" in samples.columns else pd.NA
    out["buy_date"] = pd.to_datetime(samples["date"])
    out["buy_price"] = pd.to_numeric(samples["buy_price"], errors="coerce")
    out["sell_date"] = pd.to_datetime(samples["next_date"])
    out["sell_price"] = out["buy_price"] * (1.0 + pd.to_numeric(samples["next_close_return"], errors="coerce"))
    out["shares"] = 1
    out["gross_pnl"] = out["sell_price"] - out["buy_price"]
    out["return_pct"] = (out["sell_price"] / out["buy_price"] - 1.0) * 100.0
    out["is_profit"] = out["gross_pnl"] > 0
    out["target_price"] = out["buy_price"] * (1.0 + target_return)
    out["target_hit"] = pd.to_numeric(samples["next_high_return"], errors="coerce") > target_return
    out["target_high_return_pct"] = pd.to_numeric(samples["next_high_return"], errors="coerce") * 100.0
    out["next_open_return_pct"] = pd.to_numeric(samples["next_open_return"], errors="coerce") * 100.0
    out["next_low_return_pct"] = pd.to_numeric(samples["next_low_return"], errors="coerce") * 100.0
    out["limit_date"] = pd.to_datetime(samples["limit_date"])
    out["wash_date"] = pd.to_datetime(samples["wash_date"])
    return out[columns].sort_values(["buy_date", "code"]).reset_index(drop=True)
