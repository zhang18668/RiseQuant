"""F-007 经典技术指标因子

提供 MA / EMA / MACD / RSI / BOLL / KDJ / ATR 等指标, 以及它们在 event_date
当日切片下的事件特征.

**反未来函数约定**: 所有指标计算的窗口均 `[event_idx + 1 - n : event_idx + 1]`
(含当日及之前), 绝不使用 `event_idx + 1` 及之后的数据.

接口与 :class:`PreTrendFeatures` 保持一致, 主入口:
    TechnicalIndicators().calculate_at_event(code, event_date, daily_data) -> pd.Series
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


# ----------------------------------------------------------------------
# 1) 纯静态指标计算器 (输入 Series, 输出 Series)
# ----------------------------------------------------------------------
class Indicators:
    """无副作用的技术指标工具集. 全部基于 pandas, 不依赖 talib."""

    # ---- 均线 -------------------------------------------------------
    @staticmethod
    def sma(close: pd.Series, n: int) -> pd.Series:
        return close.rolling(window=n, min_periods=n).mean()

    @staticmethod
    def ema(close: pd.Series, n: int) -> pd.Series:
        return close.ewm(span=n, adjust=False, min_periods=n).mean()

    # ---- MACD ------------------------------------------------------
    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        ema_fast = Indicators.ema(close, fast)
        ema_slow = Indicators.ema(close, slow)
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = (dif - dea) * 2
        return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})

    # ---- RSI -------------------------------------------------------
    @staticmethod
    def rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        # Wilder's smoothing
        avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        rs = avg_up / (avg_down + EPS)
        return 100.0 - 100.0 / (1.0 + rs)

    # ---- BOLL -----------------------------------------------------
    @staticmethod
    def boll(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
        mid = close.rolling(window=n, min_periods=n).mean()
        std = close.rolling(window=n, min_periods=n).std(ddof=0)
        up = mid + k * std
        dn = mid - k * std
        return pd.DataFrame({"mid": mid, "up": up, "dn": dn})

    # ---- KDJ ------------------------------------------------------
    @staticmethod
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
    ) -> pd.DataFrame:
        ll = low.rolling(window=n, min_periods=n).min()
        hh = high.rolling(window=n, min_periods=n).max()
        rsv = 100 * (close - ll) / (hh - ll + EPS)
        # 通达信 KDJ: K/D 用 SMA(rsv, m1) 的简化 (实际是带权 EMA 1/m1)
        k = rsv.ewm(alpha=1.0 / m1, adjust=False, min_periods=m1).mean()
        d = k.ewm(alpha=1.0 / m2, adjust=False, min_periods=m2).mean()
        j = 3 * k - 2 * d
        return pd.DataFrame({"k": k, "d": d, "j": j})

    # ---- ATR ------------------------------------------------------
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        n: int = 14,
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ----------------------------------------------------------------------
# 2) 事件层特征 — 在 event_date 当日的指标快照与衍生形态
# ----------------------------------------------------------------------
@dataclass
class TechnicalIndicators:
    """经典技术指标 + 衍生形态特征 (事件层)."""

    ma_periods: tuple = (5, 10, 20, 60)
    ema_periods: tuple = (12, 26)
    macd_params: tuple = (12, 26, 9)
    rsi_periods: tuple = (6, 14)
    boll_n: int = 20
    boll_k: float = 2.0
    kdj_params: tuple = (9, 3, 3)
    atr_n: int = 14

    # ------------------------------------------------------------------
    def calculate_at_event(
        self,
        code: str,
        event_date,
        daily_data: pd.DataFrame,
    ) -> pd.Series:
        df = daily_data[daily_data["code"] == code].sort_values("date").reset_index(drop=True)
        event_date = pd.Timestamp(event_date)
        idx_arr = df.index[df["date"] == event_date]
        if len(idx_arr) == 0:
            return self._empty(code, event_date)
        event_idx = int(idx_arr[0])

        # 切片到当日及之前, 防未来
        close_full = df["close"]
        high_full = df["high"]
        low_full = df["low"]
        # 仅传入 [:event_idx+1] 给指标计算
        close = close_full.iloc[: event_idx + 1]
        high = high_full.iloc[: event_idx + 1]
        low = low_full.iloc[: event_idx + 1]

        if len(close) < max(self.ma_periods + self.ema_periods +
                             self.rsi_periods + (self.boll_n, self.kdj_params[0], self.atr_n)):
            return self._empty(code, event_date)

        out: Dict[str, float] = {"code": code, "event_date": event_date}
        base_close = float(close.iloc[-1])

        # MA + EMA --------------------------------------------------
        ma_vals: Dict[int, float] = {}
        for n in self.ma_periods:
            v = Indicators.sma(close, n).iloc[-1]
            ma_vals[n] = float(v) if pd.notna(v) else float("nan")
            out[f"f_ti_ma{n}"] = ma_vals[n]
            out[f"f_ti_close_over_ma{n}"] = _safe_div(base_close - ma_vals[n], ma_vals[n])
        # 多头排列 (MA5 > MA10 > MA20 > MA60)
        ordered = [ma_vals[n] for n in self.ma_periods if not pd.isna(ma_vals[n])]
        bull = len(ordered) == len(self.ma_periods) and all(
            ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)
        )
        out["f_ti_ma_bullish"] = 1.0 if bull else 0.0

        ema_vals: Dict[int, float] = {}
        for n in self.ema_periods:
            v = Indicators.ema(close, n).iloc[-1]
            ema_vals[n] = float(v) if pd.notna(v) else float("nan")
            out[f"f_ti_ema{n}"] = ema_vals[n]

        # MACD ------------------------------------------------------
        macd = Indicators.macd(close, *self.macd_params)
        out["f_ti_macd_dif"] = float(macd["dif"].iloc[-1]) if pd.notna(macd["dif"].iloc[-1]) else float("nan")
        out["f_ti_macd_dea"] = float(macd["dea"].iloc[-1]) if pd.notna(macd["dea"].iloc[-1]) else float("nan")
        out["f_ti_macd_hist"] = float(macd["hist"].iloc[-1]) if pd.notna(macd["hist"].iloc[-1]) else float("nan")
        # 金叉/死叉 (前一日 vs 今日)
        if len(macd) >= 2 and pd.notna(macd["dif"].iloc[-2]) and pd.notna(macd["dea"].iloc[-2]):
            prev_diff = macd["dif"].iloc[-2] - macd["dea"].iloc[-2]
            curr_diff = macd["dif"].iloc[-1] - macd["dea"].iloc[-1]
            out["f_ti_macd_golden_x"] = 1.0 if (prev_diff < 0 and curr_diff > 0) else 0.0
            out["f_ti_macd_dead_x"]   = 1.0 if (prev_diff > 0 and curr_diff < 0) else 0.0
        else:
            out["f_ti_macd_golden_x"] = 0.0
            out["f_ti_macd_dead_x"] = 0.0
        # MACD 柱状方向 (近 3 日是否放大)
        last3 = macd["hist"].iloc[-3:].dropna().tolist()
        out["f_ti_macd_hist_up3"] = 1.0 if (len(last3) == 3 and last3[2] > last3[1] > last3[0]) else 0.0

        # RSI -------------------------------------------------------
        for n in self.rsi_periods:
            v = Indicators.rsi(close, n).iloc[-1]
            out[f"f_ti_rsi{n}"] = float(v) if pd.notna(v) else float("nan")
            # 超买超卖
            out[f"f_ti_rsi{n}_overbought"] = 1.0 if (pd.notna(v) and v > 70) else 0.0
            out[f"f_ti_rsi{n}_oversold"]   = 1.0 if (pd.notna(v) and v < 30) else 0.0

        # BOLL ------------------------------------------------------
        boll = Indicators.boll(close, self.boll_n, self.boll_k)
        b_mid = float(boll["mid"].iloc[-1]) if pd.notna(boll["mid"].iloc[-1]) else float("nan")
        b_up  = float(boll["up"].iloc[-1])  if pd.notna(boll["up"].iloc[-1])  else float("nan")
        b_dn  = float(boll["dn"].iloc[-1])  if pd.notna(boll["dn"].iloc[-1])  else float("nan")
        out["f_ti_boll_mid"] = b_mid
        out["f_ti_boll_up"]  = b_up
        out["f_ti_boll_dn"]  = b_dn
        # 价格在通道内的位置 (0=下轨, 1=上轨)
        rng = b_up - b_dn if (pd.notna(b_up) and pd.notna(b_dn) and (b_up - b_dn) > 0) else float("nan")
        out["f_ti_boll_pos"] = _safe_div(base_close - b_dn, rng) if pd.notna(rng) else float("nan")
        # 带宽 (上下轨距离 / 中轨)
        out["f_ti_boll_width"] = _safe_div(b_up - b_dn, b_mid) if pd.notna(b_mid) else float("nan")
        # 是否突破上轨
        out["f_ti_boll_breakup"] = 1.0 if (pd.notna(b_up) and base_close > b_up) else 0.0

        # KDJ -------------------------------------------------------
        kdj = Indicators.kdj(high, low, close, *self.kdj_params)
        out["f_ti_kdj_k"] = float(kdj["k"].iloc[-1]) if pd.notna(kdj["k"].iloc[-1]) else float("nan")
        out["f_ti_kdj_d"] = float(kdj["d"].iloc[-1]) if pd.notna(kdj["d"].iloc[-1]) else float("nan")
        out["f_ti_kdj_j"] = float(kdj["j"].iloc[-1]) if pd.notna(kdj["j"].iloc[-1]) else float("nan")
        if len(kdj) >= 2 and pd.notna(kdj["k"].iloc[-2]) and pd.notna(kdj["d"].iloc[-2]):
            prev = kdj["k"].iloc[-2] - kdj["d"].iloc[-2]
            curr = kdj["k"].iloc[-1] - kdj["d"].iloc[-1]
            out["f_ti_kdj_golden_x"] = 1.0 if (prev < 0 and curr > 0) else 0.0
        else:
            out["f_ti_kdj_golden_x"] = 0.0

        # ATR -------------------------------------------------------
        atr = Indicators.atr(high, low, close, self.atr_n).iloc[-1]
        out["f_ti_atr"] = float(atr) if pd.notna(atr) else float("nan")
        out["f_ti_atr_pct"] = _safe_div(atr, base_close) if pd.notna(atr) else float("nan")

        return pd.Series(out)

    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        names = ["code", "event_date"]
        for n in self.ma_periods:
            names += [f"f_ti_ma{n}", f"f_ti_close_over_ma{n}"]
        names.append("f_ti_ma_bullish")
        for n in self.ema_periods:
            names.append(f"f_ti_ema{n}")
        names += [
            "f_ti_macd_dif", "f_ti_macd_dea", "f_ti_macd_hist",
            "f_ti_macd_golden_x", "f_ti_macd_dead_x", "f_ti_macd_hist_up3",
        ]
        for n in self.rsi_periods:
            names += [f"f_ti_rsi{n}", f"f_ti_rsi{n}_overbought", f"f_ti_rsi{n}_oversold"]
        names += [
            "f_ti_boll_mid", "f_ti_boll_up", "f_ti_boll_dn",
            "f_ti_boll_pos", "f_ti_boll_width", "f_ti_boll_breakup",
            "f_ti_kdj_k", "f_ti_kdj_d", "f_ti_kdj_j", "f_ti_kdj_golden_x",
            "f_ti_atr", "f_ti_atr_pct",
        ]
        return names

    def _empty(self, code: str, event_date) -> pd.Series:
        data = {n: float("nan") for n in self.feature_names()}
        data["code"] = code
        data["event_date"] = event_date
        return pd.Series(data)

    def to_dict(self, feats, drop_meta: bool = True) -> Dict[str, float]:
        d = dict(feats) if isinstance(feats, dict) else feats.to_dict()
        if drop_meta:
            for k in ("code", "event_date"):
                d.pop(k, None)
        return d
