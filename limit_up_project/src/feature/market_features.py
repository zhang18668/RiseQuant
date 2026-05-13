"""F-008 大盘环境特征

输入: 大盘指数日线 (date, close, volume) — 比如上证综指 999999.
输出: 在每一个事件日的"大盘环境快照", 横切到所有事件样本.

**反未来函数约定**: 所有计算都仅用 `date <= event_date` 的数据.

如果没有真实的大盘指数, 可以传入"全市场日均价" / "全市场成交量之和"
作为代理 — 见 :func:`market_proxy_from_daily`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


def market_proxy_from_daily(daily_data: pd.DataFrame) -> pd.DataFrame:
    """从所有股票的日线里合成大盘代理:
        close: 当日所有股票 close 的中位数 (避免极端值影响)
        volume: 当日所有股票 volume 之和
    返回 DataFrame [date, close, volume].
    """
    if daily_data.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = daily_data[["date", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    agg = df.groupby("date").agg(
        close=("close", "median"),
        volume=("volume", "sum"),
    ).reset_index()
    return agg.sort_values("date").reset_index(drop=True)


@dataclass
class MarketFeatures:
    """大盘环境特征."""

    ma_periods: tuple = (5, 10, 20, 60)
    ret_periods: tuple = (1, 5, 10, 20)
    vol_ma_periods: tuple = (5, 20)

    # ------------------------------------------------------------------
    def calculate_at_event(
        self,
        event_date,
        market_data: pd.DataFrame,
    ) -> pd.Series:
        """单事件日的大盘特征.

        Parameters
        ----------
        event_date : datetime / str
        market_data : DataFrame
            列 ``date, close, volume``.
        """
        if market_data is None or market_data.empty:
            return self._empty(event_date)
        df = market_data.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        event_date = pd.Timestamp(event_date)
        idx_arr = df.index[df["date"] == event_date]
        if len(idx_arr) == 0:
            # 找最近的早于 event_date 的交易日
            idx_arr = df.index[df["date"] <= event_date]
            if len(idx_arr) == 0:
                return self._empty(event_date)
            event_idx = int(idx_arr[-1])
        else:
            event_idx = int(idx_arr[0])

        # 严格切片到当日及之前
        close = df["close"].iloc[: event_idx + 1].astype(float)
        volume = df["volume"].iloc[: event_idx + 1].astype(float)

        if len(close) < max(max(self.ma_periods), max(self.ret_periods),
                             max(self.vol_ma_periods)) + 1:
            return self._empty(event_date)

        base = float(close.iloc[-1])
        out: Dict[str, float] = {"event_date": event_date}

        # 大盘点位 + N 日收益
        out["f_mkt_close"] = base
        for n in self.ret_periods:
            if event_idx - n >= 0:
                prev = float(close.iloc[-(n + 1)])
                out[f"f_mkt_ret_{n}"] = _safe_div(base - prev, prev)
            else:
                out[f"f_mkt_ret_{n}"] = float("nan")

        # 大盘均线 + 价格位置
        ma_vals = {}
        for n in self.ma_periods:
            ma = float(close.iloc[-n:].mean()) if len(close) >= n else float("nan")
            ma_vals[n] = ma
            out[f"f_mkt_ma{n}"] = ma
            out[f"f_mkt_close_over_ma{n}"] = _safe_div(base - ma, ma) if pd.notna(ma) else float("nan")
        # 多头排列
        ordered = [ma_vals[n] for n in self.ma_periods if not pd.isna(ma_vals[n])]
        out["f_mkt_ma_bullish"] = 1.0 if (
            len(ordered) == len(self.ma_periods)
            and all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1))
        ) else 0.0

        # 大盘趋势强度 (近 20 日线性回归斜率 / 价格)
        if len(close) >= 20:
            y = close.iloc[-20:].to_numpy(dtype=float)
            x = np.arange(20, dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            out["f_mkt_slope_20"] = _safe_div(slope, float(np.mean(y)))
        else:
            out["f_mkt_slope_20"] = float("nan")

        # 大盘波动 (近 20 日 std)
        rets = close.pct_change(fill_method=None).iloc[-20:]
        out["f_mkt_vol_20"] = float(rets.std()) if rets.notna().sum() >= 5 else float("nan")

        # 大盘成交量
        out["f_mkt_volume"] = float(volume.iloc[-1])
        for n in self.vol_ma_periods:
            vma = float(volume.iloc[-n:].mean()) if len(volume) >= n else float("nan")
            out[f"f_mkt_vol_ma{n}"] = vma
            out[f"f_mkt_vol_ratio_{n}"] = _safe_div(float(volume.iloc[-1]), vma)

        # 量价配合
        if len(close) >= 5:
            c_pct = close.pct_change(fill_method=None).iloc[-5:]
            v_pct = volume.pct_change(fill_method=None).iloc[-5:]
            if c_pct.std() and v_pct.std():
                out["f_mkt_vp_corr_5"] = float(c_pct.corr(v_pct))
            else:
                out["f_mkt_vp_corr_5"] = 0.0
        else:
            out["f_mkt_vp_corr_5"] = float("nan")

        # 距离 60 日高/低位置
        if len(close) >= 60:
            hi60 = float(close.iloc[-60:].max())
            lo60 = float(close.iloc[-60:].min())
            rng = hi60 - lo60
            out["f_mkt_pos_60"] = _safe_div(base - lo60, rng) if rng > 0 else float("nan")
            out["f_mkt_dist_to_hi60"] = _safe_div(hi60 - base, base)
        else:
            out["f_mkt_pos_60"] = float("nan")
            out["f_mkt_dist_to_hi60"] = float("nan")

        return pd.Series(out)

    # ------------------------------------------------------------------
    def batch_for_events(
        self,
        events: pd.DataFrame,
        market_data: pd.DataFrame,
        event_date_col: str = "event_date",
    ) -> pd.DataFrame:
        """对一批事件日批量算大盘特征 (按日期去重后只算一次, 再 join 回事件)."""
        if events.empty:
            return pd.DataFrame()
        unique_dates = pd.to_datetime(events[event_date_col]).drop_duplicates().sort_values()
        rows = [self.calculate_at_event(d, market_data) for d in unique_dates]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["event_date"] = pd.to_datetime(df["event_date"])
        return df

    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        names = ["event_date", "f_mkt_close"]
        for n in self.ret_periods:
            names.append(f"f_mkt_ret_{n}")
        for n in self.ma_periods:
            names += [f"f_mkt_ma{n}", f"f_mkt_close_over_ma{n}"]
        names += [
            "f_mkt_ma_bullish", "f_mkt_slope_20", "f_mkt_vol_20",
            "f_mkt_volume",
        ]
        for n in self.vol_ma_periods:
            names += [f"f_mkt_vol_ma{n}", f"f_mkt_vol_ratio_{n}"]
        names += ["f_mkt_vp_corr_5", "f_mkt_pos_60", "f_mkt_dist_to_hi60"]
        return names

    def _empty(self, event_date) -> pd.Series:
        d = {n: float("nan") for n in self.feature_names()}
        d["event_date"] = pd.Timestamp(event_date)
        return pd.Series(d)
