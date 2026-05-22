"""Tushare/Tinyshare limit-up pool adapter.

The project stores zt-pool cache in the AkShare/Eastmoney column shape.  This
module fetches Tushare Pro compatible APIs and normalizes them into that shape
so existing research code can keep reading ``data/zt_pool_cache`` unchanged.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class _RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Allows up to ``max_calls`` invocations within ``window_seconds`` seconds.
    When the cap is reached, ``acquire()`` blocks until the oldest call
    falls outside the window.
    """

    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.max_calls = int(max_calls)
        self.window = float(window_seconds)
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0] >= self.window:
                    self._events.popleft()
                if len(self._events) < self.max_calls:
                    self._events.append(now)
                    return
                sleep_for = self.window - (now - self._events[0]) + 0.01
            time.sleep(max(sleep_for, 0.01))


LIMIT_LIST_FIELDS = ",".join(
    [
        "trade_date",
        "ts_code",
        "industry",
        "name",
        "close",
        "pct_chg",
        "amount",
        "float_mv",
        "total_mv",
        "turnover_ratio",
        "fd_amount",
        "first_time",
        "last_time",
        "open_times",
        "up_stat",
        "limit_times",
        "limit",
    ]
)

DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,amount"
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,circ_mv,total_mv"


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _code(value: Any) -> str:
    raw = str(value or "")
    if "." in raw:
        raw = raw.split(".", 1)[0]
    return raw.zfill(6)


def _time(value: Any) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).replace(":", "").replace(".", "").strip()
    if not raw:
        return ""
    return raw.zfill(6)[:6]


def _wan_to_yuan(series: pd.Series) -> pd.Series:
    # Tushare market value fields are conventionally in 10k yuan.
    return _as_number(series) * 10_000.0


def _amount_to_yuan(series: pd.Series) -> pd.Series:
    values = _as_number(series)
    median = values.abs().median()
    if pd.isna(median):
        return values
    # Tushare amount fields commonly use 1k yuan; if the magnitude is already
    # exchange-like yuan, leave it untouched.
    return values * 1_000.0 if median < 100_000_000 else values


def _first_series(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(index=frame.index, dtype=float)


def _token_from_config() -> str | None:
    try:
        from src.utils.config import get_config

        data = get_config().get_section("data")
    except Exception:  # noqa: BLE001
        return None
    for key in (
        "tinyshare_token",
        "tushare_token",
        "ts_token",
        "tushare_pro_token",
    ):
        token = data.get(key)
        if token:
            return str(token)
    return None


@dataclass
class TushareLimitPoolClient:
    """Thin wrapper that supports either tinyshare or official tushare.

    A built-in sliding-window rate limiter caps total API calls per minute so
    that batch backfills cannot exceed the provider's quota.  Default is
    ``rate_limit_per_min=120`` which matches the user's tinyshare plan.
    """

    token: str | None = None
    prefer_tinyshare: bool = True
    rate_limit_per_min: int = 120
    rate_limit_window_seconds: float = 60.0
    _limiter: _RateLimiter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.token = (
            self.token
            or os.environ.get("TINYSHARE_TOKEN")
            or os.environ.get("TUSHARE_TOKEN")
            or os.environ.get("TS_TOKEN")
            or os.environ.get("TUSHARE_PRO_TOKEN")
            or _token_from_config()
        )
        if not self.token:
            raise RuntimeError(
                "missing token: set TINYSHARE_TOKEN or TUSHARE_TOKEN/TS_TOKEN"
            )
        self.ts = self._import_sdk()
        self.ts.set_token(self.token)
        self.pro = self.ts.pro_api()
        self._limiter = _RateLimiter(
            max_calls=self.rate_limit_per_min,
            window_seconds=self.rate_limit_window_seconds,
        )

    def _throttled(self):
        if self._limiter is not None:
            self._limiter.acquire()

    def _import_sdk(self):
        if self.prefer_tinyshare and os.environ.get("TINYSHARE_TOKEN"):
            import tinyshare as ts  # type: ignore

            return ts
        try:
            import tinyshare as ts  # type: ignore

            return ts
        except Exception:  # noqa: BLE001
            import tushare as ts  # type: ignore

            return ts

    def limit_list_d(self, trade_date: str, limit_type: str = "U") -> pd.DataFrame:
        self._throttled()
        return self.pro.limit_list_d(
            trade_date=trade_date,
            limit_type=limit_type,
            fields=LIMIT_LIST_FIELDS,
        )

    def daily(self, trade_date: str) -> pd.DataFrame:
        self._throttled()
        return self.pro.daily(trade_date=trade_date, fields=DAILY_FIELDS)

    def daily_basic(self, trade_date: str) -> pd.DataFrame:
        self._throttled()
        return self.pro.daily_basic(trade_date=trade_date, fields=DAILY_BASIC_FIELDS)

    def trade_dates(self, start_date: str, end_date: str) -> list[str]:
        try:
            self._throttled()
            cal = self.pro.trade_cal(
                exchange="SSE",
                start_date=start_date,
                end_date=end_date,
                is_open="1",
                fields="cal_date,is_open",
            )
            if cal is not None and not cal.empty:
                return sorted(cal["cal_date"].astype(str).tolist())
        except Exception:  # noqa: BLE001
            pass
        return [d.strftime("%Y%m%d") for d in pd.bdate_range(start_date, end_date)]


def normalize_limit_list_to_zt_pool(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["代码"] = df["ts_code"].map(_code)
    out["名称"] = df.get("name", pd.NA)
    out["涨跌幅"] = _as_number(df.get("pct_chg", pd.NA))
    out["最新价"] = _as_number(df.get("close", pd.NA))
    out["成交额"] = _amount_to_yuan(df.get("amount", pd.Series(index=df.index, dtype=float)))
    # limit_list_d returns market value in yuan in tinyshare/tushare-compatible
    # responses, while daily_basic uses 10k yuan. Keep this path in yuan.
    out["流通市值"] = _as_number(df.get("float_mv", pd.Series(index=df.index, dtype=float)))
    out["总市值"] = _as_number(df.get("total_mv", pd.Series(index=df.index, dtype=float)))
    out["换手率"] = _as_number(df.get("turnover_ratio", pd.NA))
    out["封板资金"] = _as_number(df.get("fd_amount", pd.NA))
    out["首次封板时间"] = df.get("first_time", pd.Series(index=df.index)).map(_time)
    out["最后封板时间"] = df.get("last_time", pd.Series(index=df.index)).map(_time)
    out["炸板次数"] = _as_number(df.get("open_times", pd.NA)).fillna(0).astype(int)
    out["涨停统计"] = df.get("up_stat", pd.NA)
    out["连板数"] = _as_number(df.get("limit_times", pd.NA)).fillna(1).astype(int)
    out["所属行业"] = df.get("industry", pd.NA)
    out.insert(0, "序号", range(1, len(out) + 1))
    return out


def normalize_previous_pool(
    current_date: str,
    prev_limit: pd.DataFrame,
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
) -> pd.DataFrame:
    if prev_limit is None or prev_limit.empty:
        return pd.DataFrame()
    base = prev_limit.copy()
    base["code"] = base["ts_code"].map(_code)

    quote = daily.copy() if daily is not None else pd.DataFrame()
    if not quote.empty:
        quote["code"] = quote["ts_code"].map(_code)
    basic = daily_basic.copy() if daily_basic is not None else pd.DataFrame()
    if not basic.empty:
        basic["code"] = basic["ts_code"].map(_code)

    merged = base.merge(
        quote[["code", "close", "pre_close", "high", "low", "pct_chg", "amount"]]
        if not quote.empty else pd.DataFrame(columns=["code"]),
        on="code",
        how="left",
        suffixes=("", "_today"),
    )
    if not basic.empty and "circ_mv" in basic.columns and "float_mv" not in basic.columns:
        basic["float_mv"] = basic["circ_mv"]
    merged = merged.merge(
        basic[[c for c in ["code", "turnover_rate", "float_mv", "total_mv"] if c in basic.columns]]
        if not basic.empty else pd.DataFrame(columns=["code"]),
        on="code",
        how="left",
    )
    for col in [
        "pct_chg_today", "close_today", "close", "pre_close", "amount_today",
        "amount", "float_mv_today", "float_mv", "total_mv_today", "total_mv",
        "turnover_rate", "high", "low",
    ]:
        if col not in merged.columns:
            merged[col] = pd.NA

    out = pd.DataFrame()
    out["代码"] = merged["code"]
    out["名称"] = merged.get("name", pd.NA)
    out["涨跌幅"] = _as_number(merged.get("pct_chg_today", merged.get("pct_chg", pd.NA)))
    out["最新价"] = _as_number(merged.get("close_today", merged.get("close", pd.NA)))
    pre_close = _as_number(merged.get("pre_close", pd.NA))
    out["涨停价"] = pre_close * 1.10
    out["成交额"] = _amount_to_yuan(_first_series(merged, ["amount_today", "amount_y", "amount_x", "amount"]))
    out["流通市值"] = _wan_to_yuan(_first_series(merged, ["float_mv_y", "float_mv_today", "float_mv"]))
    out["总市值"] = _wan_to_yuan(_first_series(merged, ["total_mv_y", "total_mv_today", "total_mv"]))
    out["换手率"] = _as_number(_first_series(merged, ["turnover_rate_today", "turnover_rate_y", "turnover_rate"]))
    out["涨速"] = pd.NA
    high = _as_number(merged.get("high", pd.NA))
    low = _as_number(merged.get("low", pd.NA))
    out["振幅"] = (high - low) / pre_close.replace(0, pd.NA) * 100.0
    out["昨日封板时间"] = merged.get("last_time", pd.Series(index=merged.index)).map(_time)
    out["昨日连板数"] = _as_number(merged.get("limit_times", pd.NA)).fillna(1).astype(int)
    out["涨停统计"] = merged.get("up_stat", pd.NA)
    out["所属行业"] = merged.get("industry", pd.NA)
    out.insert(0, "序号", range(1, len(out) + 1))
    return out
