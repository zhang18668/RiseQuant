"""F-005 涨停前走势特征 (核心)

在涨停事件发生当日，回看 1 个月 / 2 个月窗口内的趋势、量价、波动、
支撑压力等特征。最终再做一个综合评分，用于直观判断"涨停前是否
处于良好启动形态"。

输入要求
----------
单只股票按日期升序排列的 ``DataFrame``，至少包含
``date, open, high, low, close, volume``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and not pd.isna(b) else float("nan")


@dataclass
class PreTrendFeatures:
    """涨停前走势特征计算器。"""

    window_1m: int = 20
    window_2m: int = 40

    # ------------------------------------------------------------------
    def calculate_at_event(
        self,
        code: str,
        event_date,
        daily_data: pd.DataFrame,
    ) -> pd.Series:
        """计算 ``event_date`` 当日的涨停前走势特征。

        若历史数据不足 ``window_1m``，返回全 NaN 的特征 Series。
        """
        df = daily_data[daily_data["code"] == code].sort_values("date").reset_index(drop=True)
        event_date = pd.Timestamp(event_date)
        idx_arr = df.index[df["date"] == event_date]
        if len(idx_arr) == 0:
            return self._empty_features(code, event_date)
        event_idx = int(idx_arr[0])

        if event_idx < self.window_1m:
            return self._empty_features(code, event_date)

        feats: Dict[str, float] = {"code": code, "event_date": event_date}
        feats.update(self.calc_return_features(df["close"], event_idx))
        feats.update(self.calc_ma_alignment(df["close"], event_idx))
        feats.update(self.calc_trend_slope(df["close"], event_idx))
        feats.update(self.calc_volume_price_features(df["volume"], df["close"], event_idx))
        feats.update(self.calc_volume_convergence(df["volume"], event_idx))
        feats.update(self.calc_volatility_features(df["high"], df["low"], df["close"], event_idx))
        feats.update(self.calc_consolidation_days(df["high"], df["low"], event_idx))
        feats.update(self.calc_support_resistance(df["close"], df["high"], df["low"], event_idx))
        feats.update(self.calc_comprehensive_score(feats))
        return pd.Series(feats)

    # ------------------------------------------------------------------
    # 5.5.1 趋势类
    # ------------------------------------------------------------------
    def calc_return_features(self, close: pd.Series, event_idx: int) -> Dict[str, float]:
        base = float(close.iloc[event_idx])
        ret: Dict[str, float] = {}
        for n in (1, 5, self.window_1m, self.window_2m):
            if event_idx - n >= 0:
                prev = float(close.iloc[event_idx - n])
                ret[f"f_pt_ret_{n}"] = _safe_div(base - prev, prev)
            else:
                ret[f"f_pt_ret_{n}"] = float("nan")
        # 1 个月内最大涨幅
        win = close.iloc[max(0, event_idx - self.window_1m) : event_idx]
        if not win.empty:
            ret["f_pt_max_ret_1m"] = (float(win.max()) - base) / base
            ret["f_pt_min_ret_1m"] = (float(win.min()) - base) / base
        else:
            ret["f_pt_max_ret_1m"] = float("nan")
            ret["f_pt_min_ret_1m"] = float("nan")
        return ret

    def calc_ma_alignment(self, close: pd.Series, event_idx: int) -> Dict[str, float]:
        """均线多头排列：MA5 > MA10 > MA20 > MA40 时为 1。"""
        out: Dict[str, float] = {}
        mas: Dict[int, float] = {}
        for n in (5, 10, 20, self.window_2m):
            if event_idx + 1 - n >= 0:
                mas[n] = float(close.iloc[event_idx + 1 - n : event_idx + 1].mean())
            else:
                mas[n] = float("nan")
            out[f"f_pt_ma_{n}"] = mas[n]

        bull = (
            not any(pd.isna(v) for v in mas.values())
            and mas[5] > mas[10] > mas[20] > mas[self.window_2m]
        )
        out["f_pt_ma_bullish"] = 1.0 if bull else 0.0
        # 收盘价与 20 日均线偏离
        out["f_pt_close_to_ma20"] = (
            (float(close.iloc[event_idx]) / mas[20] - 1.0)
            if not pd.isna(mas[20])
            else float("nan")
        )
        return out

    def calc_trend_slope(self, close: pd.Series, event_idx: int) -> Dict[str, float]:
        """对窗口内做线性回归得到斜率（除以均价归一化）。"""
        out: Dict[str, float] = {}
        for n, name in ((self.window_1m, "1m"), (self.window_2m, "2m")):
            if event_idx + 1 - n < 0:
                out[f"f_pt_slope_{name}"] = float("nan")
                continue
            y = close.iloc[event_idx + 1 - n : event_idx + 1].to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            mean = float(np.mean(y))
            out[f"f_pt_slope_{name}"] = slope / mean if mean else float("nan")
        return out

    # ------------------------------------------------------------------
    # 5.5.2 量价配合
    # ------------------------------------------------------------------
    def calc_volume_price_features(
        self,
        volume: pd.Series,
        close: pd.Series,
        event_idx: int,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if event_idx + 1 - self.window_1m < 0:
            return {
                "f_pt_vp_corr_1m": float("nan"),
                "f_pt_vol_change_1m": float("nan"),
            }
        v = volume.iloc[event_idx + 1 - self.window_1m : event_idx + 1]
        c = close.iloc[event_idx + 1 - self.window_1m : event_idx + 1]
        v_pct = v.pct_change(fill_method=None)
        c_pct = c.pct_change(fill_method=None)
        if v_pct.std() > 0 and c_pct.std() > 0:
            out["f_pt_vp_corr_1m"] = float(v_pct.corr(c_pct))
        else:
            out["f_pt_vp_corr_1m"] = 0.0
        # 平均量能变化
        out["f_pt_vol_change_1m"] = float(v_pct.mean())
        return out

    def calc_volume_convergence(self, volume: pd.Series, event_idx: int) -> Dict[str, float]:
        """缩量洗盘程度：最近 5 日均量 / 1 个月均量。"""
        if event_idx + 1 - self.window_1m < 0:
            return {"f_pt_vol_conv": float("nan")}
        recent = float(volume.iloc[event_idx + 1 - 5 : event_idx + 1].mean())
        full = float(volume.iloc[event_idx + 1 - self.window_1m : event_idx + 1].mean())
        return {"f_pt_vol_conv": _safe_div(recent, full) if full else float("nan")}

    # ------------------------------------------------------------------
    # 5.5.3 波动类
    # ------------------------------------------------------------------
    def calc_volatility_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        event_idx: int,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for n, name in ((self.window_1m, "1m"), (self.window_2m, "2m")):
            if event_idx + 1 - n < 0:
                out[f"f_pt_amp_{name}"] = float("nan")
                out[f"f_pt_vol_{name}"] = float("nan")
                continue
            h = high.iloc[event_idx + 1 - n : event_idx + 1]
            l = low.iloc[event_idx + 1 - n : event_idx + 1]
            c = close.iloc[event_idx + 1 - n : event_idx + 1]
            out[f"f_pt_amp_{name}"] = (float(h.max()) - float(l.min())) / max(float(l.min()), EPS)
            out[f"f_pt_vol_{name}"] = float(c.pct_change(fill_method=None).std())
        return out

    def calc_consolidation_days(
        self,
        high: pd.Series,
        low: pd.Series,
        event_idx: int,
    ) -> Dict[str, float]:
        """横盘整理天数：最近 N 日中振幅小于 3% 的天数。"""
        if event_idx + 1 - self.window_1m < 0:
            return {"f_pt_consol_days": float("nan")}
        h = high.iloc[event_idx + 1 - self.window_1m : event_idx + 1]
        l = low.iloc[event_idx + 1 - self.window_1m : event_idx + 1]
        amp = (h.values - l.values) / np.maximum(l.values, EPS)
        return {"f_pt_consol_days": float(np.sum(amp < 0.03))}

    # ------------------------------------------------------------------
    # 5.5.4 支撑压力
    # ------------------------------------------------------------------
    def calc_support_resistance(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        event_idx: int,
    ) -> Dict[str, float]:
        """支撑/压力位特征。"""
        out: Dict[str, float] = {}
        if event_idx + 1 - self.window_2m < 0:
            return {
                "f_pt_pos_in_range_2m": float("nan"),
                "f_pt_dist_to_high_2m": float("nan"),
                "f_pt_dist_to_low_2m": float("nan"),
            }
        h = high.iloc[event_idx + 1 - self.window_2m : event_idx + 1]
        l = low.iloc[event_idx + 1 - self.window_2m : event_idx + 1]
        c = float(close.iloc[event_idx])
        hi = float(h.max())
        lo = float(l.min())
        rng = max(hi - lo, EPS)
        out["f_pt_pos_in_range_2m"] = (c - lo) / rng
        out["f_pt_dist_to_high_2m"] = (hi - c) / max(c, EPS)
        out["f_pt_dist_to_low_2m"] = (c - lo) / max(c, EPS)
        return out

    # ------------------------------------------------------------------
    # 5.5.5 综合评分
    # ------------------------------------------------------------------
    def calc_comprehensive_score(self, feats: Dict[str, float]) -> Dict[str, float]:
        """对四个维度各给一个 0~1 的分数，最终取均值作为综合分数。"""
        # 趋势分：基于 1m 收益与均线排列
        ret_1m = feats.get(f"f_pt_ret_{self.window_1m}", np.nan)
        ma_bull = feats.get("f_pt_ma_bullish", np.nan)
        trend_score = _clip01(_norm_ret(ret_1m) * 0.7 + (ma_bull if not pd.isna(ma_bull) else 0.0) * 0.3)

        # 量能分：缩量洗盘越明显得分越高，0.5~0.9 区间最优
        vol_conv = feats.get("f_pt_vol_conv", np.nan)
        if pd.isna(vol_conv):
            vol_score = float("nan")
        else:
            vol_score = _clip01(1.0 - abs(vol_conv - 0.7))

        # 形态分：横盘天数越多 + 距 2m 高位越近
        consol = feats.get("f_pt_consol_days", np.nan)
        dist_high = feats.get("f_pt_dist_to_high_2m", np.nan)
        if pd.isna(consol) or pd.isna(dist_high):
            shape_score = float("nan")
        else:
            shape_score = _clip01(0.5 * (consol / max(self.window_1m, 1)) + 0.5 * (1.0 - min(dist_high, 1.0)))

        scores = [s for s in (trend_score, vol_score, shape_score) if not pd.isna(s)]
        comp = float(np.mean(scores)) if scores else float("nan")

        return {
            "f_pt_score_trend": trend_score,
            "f_pt_score_volume": vol_score,
            "f_pt_score_shape": shape_score,
            "f_pt_score_comp": comp,
        }

    # ------------------------------------------------------------------
    def to_series(self, feats) -> pd.Series:
        if isinstance(feats, pd.Series):
            return feats
        return pd.Series(feats)

    def to_dict(self, feats, drop_meta: bool = True) -> Dict[str, float]:
        """转成 ``dict``. ``drop_meta=True`` 时剔除 ``code`` / ``event_date`` 等元数据."""
        if isinstance(feats, dict):
            d = dict(feats)
        else:
            d = feats.to_dict()
        if drop_meta:
            for k in ("code", "event_date"):
                d.pop(k, None)
        return d

    # ------------------------------------------------------------------
    def _empty_features(self, code: str, event_date) -> pd.Series:
        cols = self._feature_names()
        data: Dict[str, float] = {c: float("nan") for c in cols}
        data["code"] = code
        data["event_date"] = event_date
        return pd.Series(data)

    def _feature_names(self):
        return [
            "code", "event_date",
            f"f_pt_ret_1", f"f_pt_ret_5",
            f"f_pt_ret_{self.window_1m}", f"f_pt_ret_{self.window_2m}",
            "f_pt_max_ret_1m", "f_pt_min_ret_1m",
            "f_pt_ma_5", "f_pt_ma_10", "f_pt_ma_20",
            f"f_pt_ma_{self.window_2m}", "f_pt_ma_bullish", "f_pt_close_to_ma20",
            "f_pt_slope_1m", "f_pt_slope_2m",
            "f_pt_vp_corr_1m", "f_pt_vol_change_1m",
            "f_pt_vol_conv",
            "f_pt_amp_1m", "f_pt_vol_1m", "f_pt_amp_2m", "f_pt_vol_2m",
            "f_pt_consol_days",
            "f_pt_pos_in_range_2m", "f_pt_dist_to_high_2m", "f_pt_dist_to_low_2m",
            "f_pt_score_trend", "f_pt_score_volume",
            "f_pt_score_shape", "f_pt_score_comp",
        ]


# ----------------------------------------------------------------------
def _clip01(x: float) -> float:
    if pd.isna(x):
        return float("nan")
    return float(min(max(x, 0.0), 1.0))


def _norm_ret(x: float) -> float:
    """把累计收益率压到 [0,1]：0% -> 0.5, +50% -> 1.0, -50% -> 0.0。"""
    if pd.isna(x):
        return float("nan")
    return _clip01(0.5 + x)
