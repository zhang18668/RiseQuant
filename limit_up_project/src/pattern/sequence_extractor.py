"""P-001 形态序列提取器 (Pattern-Cluster v2)

把 (code, potential_date, first_date) 切成一段固定长度 20、10 通道的归一化序列,
供 DTW + KMedoids 聚类与 PatternRouter 路由使用.

10 个通道
---------
0: (close / first_close) - 1     — 相对首板 close 的累计偏移
1: (high - low) / first_close    — 振幅 (相对首板)
2: (close - open) / first_close  — 实体 (相对首板)
3: volume / 前 20 日均量 - 1     — 量比偏离 (中心化)
4: (close - ma5) / ma5
5: (close - ma10) / ma10
6: (close - ma20) / ma20
7: (i_days_from_first / 30)       — 距首板天数 (固定除以 30)
8: cum_change_from_first         — 距首板 close 的累计涨跌幅
9: is_padded                     — 0/1, 标记此根 K 线是否是 padding 来的

输出
----
``np.ndarray`` shape ``(20, 10)``, dtype=float32.
缺数据时 (e.g. 找不到 first_date / potential_date) 返回 ``None``.
当回看窗口不足 20 日时, **前面 padding 0**, 同时通道 9 (`is_padded`) 标 1.0.

注意
----
* 切片严格 ``[potential_idx - 19, potential_idx]``, 不含未来.
* MA 用前 20 日均量做基准时, 如果 first_idx 之前不足 20 日, 用可用部分计算 (避免大量 NaN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


N_CHANNELS = 10
LOOKBACK = 20


@dataclass
class SequenceExtractor:
    """从单只股票日线切 20 日回看窗口."""

    lookback: int = LOOKBACK
    vol_baseline_days: int = 20    # 量比基准: 前 20 日均量
    days_norm: float = 30.0        # 距首板天数的归一化分母 (Q9 倾向: 固定 30)

    # ------------------------------------------------------------------
    def extract(
        self,
        stock_daily: pd.DataFrame,
        first_date,
        potential_date,
    ) -> Optional[np.ndarray]:
        """切 [potential_date - 19, potential_date] 的 20 日回看窗口.

        Parameters
        ----------
        stock_daily : DataFrame
            单只股票按 date 升序排列的日线, 至少含
            ``date, open, high, low, close, volume``.
        first_date : 首板日期 (Timestamp / str)
        potential_date : 候选日 (>= first_date)

        Returns
        -------
        np.ndarray shape (lookback, N_CHANNELS) dtype=float32, 缺数据返回 None.
        """
        if stock_daily is None or stock_daily.empty:
            return None
        df = stock_daily.sort_values("date").reset_index(drop=True)
        first_date = pd.Timestamp(first_date)
        potential_date = pd.Timestamp(potential_date)

        first_arr = df.index[df["date"] == first_date]
        pot_arr = df.index[df["date"] == potential_date]
        if len(first_arr) == 0 or len(pot_arr) == 0:
            return None
        first_idx = int(first_arr[0])
        pot_idx = int(pot_arr[0])
        if pot_idx < first_idx:
            return None

        first_close = float(df["close"].iloc[first_idx])
        if first_close <= 0:
            return None

        # ----- 前 20 日均量基准 (用 first 之前 + first 当日 的 20 日) -----
        vol_base_end = first_idx + 1   # exclusive, 但既然算"前 20 日"含 first 也可接受
        vol_base_start = max(0, vol_base_end - self.vol_baseline_days)
        vol_base = float(df["volume"].iloc[vol_base_start:vol_base_end].mean())
        if not (vol_base > 0):
            vol_base = float("nan")

        # ----- 切窗口 [pot_idx - lookback + 1, pot_idx] -----
        win_start_ideal = pot_idx - self.lookback + 1
        win_start = max(0, win_start_ideal)
        win = df.iloc[win_start: pot_idx + 1].reset_index(drop=True)
        pad_count = self.lookback - len(win)  # >= 0

        # ----- 计算 MA5/10/20 (用 win 之前可用的全部历史) -----
        # 为了让 win 中第 0 根也有 MA, 我们用 df 整段算 MA 然后切窗口对应位置
        full_close = df["close"].astype(float)
        ma5_series  = full_close.rolling(5,  min_periods=1).mean()
        ma10_series = full_close.rolling(10, min_periods=1).mean()
        ma20_series = full_close.rolling(20, min_periods=1).mean()

        # ----- 构造 (lookback, N_CHANNELS) -----
        arr = np.zeros((self.lookback, N_CHANNELS), dtype=np.float32)
        # padding 部分: 全 0, 但 is_padded = 1
        if pad_count > 0:
            arr[:pad_count, 9] = 1.0

        for k in range(len(win)):
            row_pos = pad_count + k      # 在 arr 中的行号
            df_pos = win_start + k       # 在 df 中的下标
            o = float(win["open"].iloc[k])
            h = float(win["high"].iloc[k])
            l = float(win["low"].iloc[k])
            c = float(win["close"].iloc[k])
            v = float(win["volume"].iloc[k])
            ma5  = float(ma5_series.iloc[df_pos])
            ma10 = float(ma10_series.iloc[df_pos])
            ma20 = float(ma20_series.iloc[df_pos])

            arr[row_pos, 0] = (c / first_close) - 1.0
            arr[row_pos, 1] = (h - l) / first_close
            arr[row_pos, 2] = (c - o) / first_close
            arr[row_pos, 3] = (v / vol_base - 1.0) if np.isfinite(vol_base) else 0.0
            arr[row_pos, 4] = (c - ma5)  / ma5  if ma5  > 0 else 0.0
            arr[row_pos, 5] = (c - ma10) / ma10 if ma10 > 0 else 0.0
            arr[row_pos, 6] = (c - ma20) / ma20 if ma20 > 0 else 0.0
            days_from_first = df_pos - first_idx
            arr[row_pos, 7] = days_from_first / self.days_norm
            arr[row_pos, 8] = (c / first_close) - 1.0  # 累计涨跌幅 (= channel 0, 但语义清楚)
            arr[row_pos, 9] = 0.0  # 非 padded

        # 安全: 把 NaN/Inf 替换为 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    # ------------------------------------------------------------------
    def extract_batch(
        self,
        daily_by_code: Dict[str, pd.DataFrame],
        records: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """批量切. records 至少含 ``code, first_date, potential_date``.

        Returns
        -------
        dict {key -> ndarray}, key = sample_id (如果有) 否则 (code, potential_date) 元组.
        缺数据样本会被跳过.
        """
        out: Dict[str, np.ndarray] = {}
        for _, r in records.iterrows():
            code = str(r["code"])
            sub = daily_by_code.get(code)
            if sub is None:
                continue
            seq = self.extract(sub, r["first_date"], r["potential_date"])
            if seq is None:
                continue
            key = str(r["sample_id"]) if "sample_id" in r and pd.notna(r.get("sample_id")) else (
                f"{code}_{pd.Timestamp(r['first_date']).strftime('%Y%m%d')}_"
                f"{pd.Timestamp(r['potential_date']).strftime('%Y%m%d')}"
            )
            out[key] = seq
        return out
