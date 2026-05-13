"""F-009 震荡洗盘期特征

针对 "首板 → 震荡 → 第二涨停" 策略, 在震荡期的某一天 (potential_date) 上,
回看 [first_date, potential_date] 区间, 提取:

1) **价格相对首板的位置** — 相对首板 close/open 的偏离
2) **均线粘合发散** — MA5/10/20 之间的相对距离 (粘合 -> 即将爆发)
3) **MACD 收敛** — DIF/DEA 是否粘合在零轴附近
4) **RSI 中性** — 是否在 40-60 区间洗盘
5) **量能特征** — 缩量程度 (相对首板)、连续缩量天数
6) **形态特征** — 横盘天数、振幅收敛、最大回撤
7) **首板/前期高低锚点** — 距首板 close 的距离、距 wash 区间最高最低

**反未来函数约定**: 所有切片都在 ``[first_idx, potential_idx]``, 绝不偷看
potential_idx 之后的数据 (尤其不能用未来的 second_date).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.feature.technical_indicators import Indicators

EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


@dataclass
class WashFeatures:
    """震荡洗盘期特征 (在震荡区间某天 potential_date 计算)."""

    ma_periods: tuple = (5, 10, 20)
    macd_params: tuple = (12, 26, 9)
    rsi_n: int = 14

    # ------------------------------------------------------------------
    def calculate(
        self,
        stock_daily: pd.DataFrame,
        first_date,
        potential_date,
    ) -> pd.Series:
        """对单只股票, 给定 first_date 与 potential_date, 算震荡期特征.

        Parameters
        ----------
        stock_daily : DataFrame
            单只票按 date 升序排列的日线 (date, open, high, low, close, volume, change_pct).
        first_date : 首板日期
        potential_date : 潜伏候选日 (in [first_date+1, second_date-1])
        """
        df = stock_daily.sort_values("date").reset_index(drop=True)
        first_date = pd.Timestamp(first_date)
        potential_date = pd.Timestamp(potential_date)

        # 定位下标
        first_arr = df.index[df["date"] == first_date]
        pot_arr = df.index[df["date"] == potential_date]
        if len(first_arr) == 0 or len(pot_arr) == 0:
            return self._empty(first_date, potential_date)
        first_idx = int(first_arr[0])
        pot_idx = int(pot_arr[0])
        if pot_idx <= first_idx:
            return self._empty(first_date, potential_date)

        first_close = float(df["close"].iloc[first_idx])
        first_open  = float(df["open"].iloc[first_idx])
        first_high  = float(df["high"].iloc[first_idx])
        first_volume = float(df["volume"].iloc[first_idx])

        # 切片到当日及之前 (含 first_idx, 含 pot_idx)
        win_close  = df["close"].iloc[: pot_idx + 1]
        win_high   = df["high"].iloc[: pot_idx + 1]
        win_low    = df["low"].iloc[: pot_idx + 1]
        win_volume = df["volume"].iloc[: pot_idx + 1]

        # 震荡子区间 (first_idx+1 .. pot_idx, 含当日)
        wash_close  = df["close"].iloc[first_idx + 1: pot_idx + 1]
        wash_high   = df["high"].iloc[first_idx + 1: pot_idx + 1]
        wash_low    = df["low"].iloc[first_idx + 1: pot_idx + 1]
        wash_volume = df["volume"].iloc[first_idx + 1: pot_idx + 1]

        base_close = float(df["close"].iloc[pot_idx])

        out: Dict[str, float] = {
            "code": str(df["code"].iloc[pot_idx]) if "code" in df.columns else "",
            "first_date": first_date,
            "potential_date": potential_date,
            "gap_days_so_far": int(pot_idx - first_idx),
        }

        # --- 1) 价格相对首板的位置 ----------------------------------
        out["f_w_close_to_fc"]    = _safe_div(base_close - first_close, first_close)
        out["f_w_close_to_fo"]    = _safe_div(base_close - first_open,  first_open)
        out["f_w_close_to_fh"]    = _safe_div(base_close - first_high,  first_high)
        out["f_w_open_vs_fc"]     = _safe_div(float(df["open"].iloc[pot_idx]) - first_close, first_close)
        # 是否曾跌破首板 open (止损位)
        if not wash_low.empty:
            out["f_w_broke_stop"]  = 1.0 if float(wash_low.min()) < first_open else 0.0
            out["f_w_max_drawdown_from_fc"] = _safe_div(float(wash_low.min()) - first_close, first_close)
            out["f_w_max_high_over_fc"]     = _safe_div(float(wash_high.max()) - first_close, first_close)
        else:
            out["f_w_broke_stop"] = float("nan")
            out["f_w_max_drawdown_from_fc"] = float("nan")
            out["f_w_max_high_over_fc"] = float("nan")

        # --- 2) 均线粘合发散 (MA5/10/20) ----------------------------
        mas: Dict[int, float] = {}
        for n in self.ma_periods:
            v = Indicators.sma(win_close, n).iloc[-1]
            mas[n] = float(v) if pd.notna(v) else float("nan")
            out[f"f_w_ma{n}"] = mas[n]
        valid_mas = [v for v in mas.values() if pd.notna(v)]
        if len(valid_mas) == len(self.ma_periods) and base_close > 0:
            # 均线间相对距离的标准差 (越小越粘合)
            rel = np.array(valid_mas, dtype=float) / base_close
            out["f_w_ma_cohesion"] = float(np.std(rel))
            # 多头排列 (MA5 > MA10 > MA20)
            ordered = [mas[n] for n in self.ma_periods]
            out["f_w_ma_bullish"] = 1.0 if all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)) else 0.0
            # close 在所有均线之上
            out["f_w_close_above_all_ma"] = 1.0 if all(base_close > v for v in valid_mas) else 0.0
        else:
            out["f_w_ma_cohesion"] = float("nan")
            out["f_w_ma_bullish"] = float("nan")
            out["f_w_close_above_all_ma"] = float("nan")

        # --- 3) MACD 收敛 -------------------------------------------
        macd = Indicators.macd(win_close, *self.macd_params)
        out["f_w_macd_dif"]  = float(macd["dif"].iloc[-1]) if pd.notna(macd["dif"].iloc[-1]) else float("nan")
        out["f_w_macd_dea"]  = float(macd["dea"].iloc[-1]) if pd.notna(macd["dea"].iloc[-1]) else float("nan")
        out["f_w_macd_hist"] = float(macd["hist"].iloc[-1]) if pd.notna(macd["hist"].iloc[-1]) else float("nan")
        # 金叉信号 (今日 dif > dea, 昨日 dif <= dea)
        if len(macd) >= 2:
            prev = macd["dif"].iloc[-2] - macd["dea"].iloc[-2]
            curr = macd["dif"].iloc[-1] - macd["dea"].iloc[-1]
            out["f_w_macd_golden_x"] = 1.0 if (pd.notna(prev) and pd.notna(curr) and prev <= 0 < curr) else 0.0
        else:
            out["f_w_macd_golden_x"] = 0.0
        # 柱状收敛后翻红
        last3 = macd["hist"].iloc[-3:].dropna().tolist()
        out["f_w_macd_hist_up3"] = 1.0 if (len(last3) == 3 and last3[2] > last3[1] > last3[0]) else 0.0

        # --- 4) RSI 中性区 -----------------------------------------
        rsi = Indicators.rsi(win_close, self.rsi_n).iloc[-1]
        out["f_w_rsi"] = float(rsi) if pd.notna(rsi) else float("nan")
        out["f_w_rsi_neutral"] = 1.0 if (pd.notna(rsi) and 40 <= rsi <= 60) else 0.0

        # --- 5) 量能特征 -------------------------------------------
        if not wash_volume.empty and first_volume > 0:
            avg_wash_vol = float(wash_volume.mean())
            out["f_w_vol_shrink"] = _safe_div(avg_wash_vol, first_volume)  # <1 表示缩量
            # 连续缩量天数 (close 较前一日下跌或持平的天数)
            recent5 = wash_volume.iloc[-5:].to_numpy(dtype=float)
            if len(recent5) >= 2:
                shrink_streak = 0
                for k in range(len(recent5) - 1, 0, -1):
                    if recent5[k] < recent5[k - 1]:
                        shrink_streak += 1
                    else:
                        break
                out["f_w_vol_shrink_streak"] = float(shrink_streak)
            else:
                out["f_w_vol_shrink_streak"] = float("nan")
            # 最近 3 日均量 / 震荡区间均量
            if len(wash_volume) >= 3:
                out["f_w_recent_vol_ratio"] = _safe_div(
                    float(wash_volume.iloc[-3:].mean()), avg_wash_vol
                )
            else:
                out["f_w_recent_vol_ratio"] = float("nan")
        else:
            out["f_w_vol_shrink"] = float("nan")
            out["f_w_vol_shrink_streak"] = float("nan")
            out["f_w_recent_vol_ratio"] = float("nan")

        # --- 6) 形态特征 -------------------------------------------
        if not wash_high.empty:
            amp = (wash_high.values - wash_low.values) / np.maximum(wash_low.values, EPS)
            out["f_w_consol_days"] = float(np.sum(amp < 0.03))   # 振幅 <3% 的天数
            out["f_w_amp_mean"] = float(np.mean(amp))
            out["f_w_amp_recent5"] = float(np.mean(amp[-5:])) if len(amp) >= 5 else float("nan")
            # 振幅收敛: 最近 5 日振幅均值 / 前期均值
            if len(amp) >= 10:
                out["f_w_amp_convergence"] = _safe_div(
                    float(np.mean(amp[-5:])), float(np.mean(amp[:-5]))
                )
            else:
                out["f_w_amp_convergence"] = float("nan")
        else:
            out["f_w_consol_days"] = float("nan")
            out["f_w_amp_mean"] = float("nan")
            out["f_w_amp_recent5"] = float("nan")
            out["f_w_amp_convergence"] = float("nan")

        # --- 7) 当日动作 -------------------------------------------
        # 今日相对昨日的涨跌幅
        if pot_idx >= 1:
            prev_close = float(df["close"].iloc[pot_idx - 1])
            out["f_w_today_ret"] = _safe_div(base_close - prev_close, prev_close)
        else:
            out["f_w_today_ret"] = float("nan")
        # 今日振幅
        today_high = float(df["high"].iloc[pot_idx])
        today_low  = float(df["low"].iloc[pot_idx])
        out["f_w_today_amp"] = _safe_div(today_high - today_low, today_low)
        # 今日量比 (相对震荡区间均量)
        if not wash_volume.empty:
            avg = float(wash_volume.mean())
            out["f_w_today_vol_ratio"] = _safe_div(float(df["volume"].iloc[pot_idx]), avg)
        else:
            out["f_w_today_vol_ratio"] = float("nan")

        return pd.Series(out)

    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        names = ["code", "first_date", "potential_date", "gap_days_so_far"]
        names += [
            "f_w_close_to_fc", "f_w_close_to_fo", "f_w_close_to_fh", "f_w_open_vs_fc",
            "f_w_broke_stop", "f_w_max_drawdown_from_fc", "f_w_max_high_over_fc",
        ]
        for n in self.ma_periods:
            names.append(f"f_w_ma{n}")
        names += [
            "f_w_ma_cohesion", "f_w_ma_bullish", "f_w_close_above_all_ma",
            "f_w_macd_dif", "f_w_macd_dea", "f_w_macd_hist",
            "f_w_macd_golden_x", "f_w_macd_hist_up3",
            "f_w_rsi", "f_w_rsi_neutral",
            "f_w_vol_shrink", "f_w_vol_shrink_streak", "f_w_recent_vol_ratio",
            "f_w_consol_days", "f_w_amp_mean", "f_w_amp_recent5", "f_w_amp_convergence",
            "f_w_today_ret", "f_w_today_amp", "f_w_today_vol_ratio",
        ]
        return names

    def _empty(self, first_date, potential_date) -> pd.Series:
        d = {n: float("nan") for n in self.feature_names()}
        d["first_date"] = pd.Timestamp(first_date)
        d["potential_date"] = pd.Timestamp(potential_date)
        return pd.Series(d)
