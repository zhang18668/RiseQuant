"""DS-004 震荡潜伏样本构造

把 (first → wash → second) 三元组展开成"滚动潜伏样本":

对每个事件三元组 (code, first_date, second_date):
  对震荡区间 [first_date+1, second_date-1] 的 *每一天 t* 生成一个样本:
    - 特征日 = t
    - label_pre_K = 1 if (second_date - t) <= ``positive_window`` else 0
    - label_to_second_days = second_date - t (回归用)
    - stop_loss_price = first_open

对没有第二涨停的首板 (negative pool):
  也生成滚动样本, 在 first 后 [min_gap, max_gap] 区间的每一天 t 都给 label=0
  这部分负样本来自 :func:`build_negative_from_failed_first`.

调用入口
--------
::

    builder = WashSampleBuilder()
    samples = builder.build(daily_data, events=wash_events,
                            failed_firsts=failed_first_events)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.event.wash_second_detector import WashSecondDetector


SAMPLE_COLS = [
    "sample_id", "code",
    "first_date", "potential_date",
    "second_date",                   # 可能为 NaT (负样本)
    "gap_so_far",                    # potential - first
    "days_to_second",                # second - potential, 负样本 = -1
    "label_pre",                     # 0/1, 是否在第二涨停前 K 日内
    "sample_weight",                 # B3: 训练样本权重 (正样本按 1/days_to_second)
    "stop_loss_price",
    "first_open", "first_close",
]


@dataclass
class WashSampleBuilder:
    """滚动潜伏样本构造器."""

    positive_window: int = 5      # 距 second_date <= K 日为正样本 (放宽到 5, 给模型更多潜伏机会)
    min_gap_for_neg: int = 3      # 负样本起始 (与 detector 的 min_gap 一致)
    max_gap_for_neg: int = 30     # 负样本结束 (与 detector 的 max_gap 一致)
    threshold: float = 9.9
    cooldown_days: int = 3        # 与 detector 的 cooldown_days 一致

    # ------------------------------------------------------------------
    def build(
        self,
        daily_data: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        include_negatives: bool = True,
    ) -> pd.DataFrame:
        """生成滚动潜伏样本.

        Parameters
        ----------
        daily_data : DataFrame
            列 ``date, code, open, high, low, close, volume, change_pct``.
        events : DataFrame, optional
            :meth:`WashSecondDetector.detect` 的输出. 不传时内部自动计算.
        include_negatives : bool
            是否生成"未来 N 日内未爆第二涨停"的负样本.
        """
        if daily_data.empty:
            return pd.DataFrame(columns=SAMPLE_COLS)

        if events is None:
            det = WashSecondDetector(
                threshold=self.threshold,
                cooldown_days=self.cooldown_days,
                min_gap=self.min_gap_for_neg,
                max_gap=self.max_gap_for_neg,
            )
            events = det.detect(daily_data)

        df = daily_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 索引: code -> sorted dates
        code_to_dates = {c: g["date"].tolist() for c, g in df.groupby("code")}

        rows: List[dict] = []

        # 正样本/中性样本: 来自 (first, second) 对的震荡区间
        events_by_code = {c: g for c, g in events.groupby("code")} if not events.empty else {}
        positive_keys = set()  # (code, first_date, t) 防止与负样本重复

        for code, ev_g in events_by_code.items():
            dates = code_to_dates.get(code, [])
            if not dates:
                continue
            date_to_pos = {d: i for i, d in enumerate(dates)}
            for _, ev in ev_g.iterrows():
                first_d = pd.Timestamp(ev["first_date"])
                second_d = pd.Timestamp(ev["second_date"])
                first_idx = date_to_pos.get(first_d)
                second_idx = date_to_pos.get(second_d)
                if first_idx is None or second_idx is None:
                    continue
                # 滚动遍历震荡区间内每一天作为潜伏候选
                for t_idx in range(first_idx + self.min_gap_for_neg, second_idx):
                    t = dates[t_idx]
                    days_to_second = second_idx - t_idx
                    label = 1 if days_to_second <= self.positive_window else 0
                    # B3: 样本权重 — 正样本按 1/days_to_second 加权 (距 second 越近权重越大)
                    # 距 1 天: weight=1.0; 2 天: 0.5; 3 天: 0.33; 4 天: 0.25; 5 天: 0.2
                    if label == 1:
                        weight = 1.0 / max(days_to_second, 1)
                    else:
                        # 震荡区间内的"中性"负样本 (距 second 较远但不属于失败首板)
                        # 给较小权重, 避免淹没失败首板的真负样本
                        weight = 0.5
                    rows.append({
                        "sample_id":      f"{code}_{first_d.strftime('%Y%m%d')}_{t.strftime('%Y%m%d')}",
                        "code":           code,
                        "first_date":     first_d,
                        "potential_date": t,
                        "second_date":    second_d,
                        "gap_so_far":     t_idx - first_idx,
                        "days_to_second": days_to_second,
                        "label_pre":      label,
                        "sample_weight":  weight,
                        "stop_loss_price": float(ev["stop_loss_price"]),
                        "first_open":     float(ev["first_open"]),
                        "first_close":    float(ev["first_close"]),
                    })
                    positive_keys.add((code, first_d, t))

        # 负样本: 所有"严格首板但 max_gap 内未出第二涨停"的首板, 滚动负样本
        if include_negatives:
            # 找出"失败首板": 严格首板 (前 cooldown_days 无涨停), 但在 [min_gap, max_gap] 内
            # 没有创新高涨停 (即 detector 未输出该 first_date)
            success_first = set()
            if not events.empty:
                for _, ev in events.iterrows():
                    success_first.add((ev["code"], pd.Timestamp(ev["first_date"])))

            failed_firsts = self._find_failed_firsts(df, success_first)
            for (code, first_d, first_open) in failed_firsts:
                dates = code_to_dates.get(code, [])
                date_to_pos = {d: i for i, d in enumerate(dates)}
                first_idx = date_to_pos.get(first_d)
                if first_idx is None:
                    continue
                end_idx = min(len(dates) - 1, first_idx + self.max_gap_for_neg)
                for t_idx in range(first_idx + self.min_gap_for_neg, end_idx + 1):
                    t = dates[t_idx]
                    if (code, first_d, t) in positive_keys:
                        continue
                    rows.append({
                        "sample_id":      f"{code}_{first_d.strftime('%Y%m%d')}_{t.strftime('%Y%m%d')}_neg",
                        "code":           code,
                        "first_date":     first_d,
                        "potential_date": t,
                        "second_date":    pd.NaT,
                        "gap_so_far":     t_idx - first_idx,
                        "days_to_second": -1,
                        "label_pre":      0,
                        "sample_weight":  1.0,   # B3: 失败首板的负样本是真负样本, 权重 1.0
                        "stop_loss_price": first_open,
                        "first_open":     first_open,
                        "first_close":    float("nan"),
                    })

        if not rows:
            return pd.DataFrame(columns=SAMPLE_COLS)
        return pd.DataFrame(rows)[SAMPLE_COLS].sort_values(
            ["code", "first_date", "potential_date"]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    def _find_failed_firsts(
        self,
        df: pd.DataFrame,
        success_first: set,
    ) -> List[tuple]:
        """找出"严格首板但 max_gap 内未爆出 valid second"的首板."""
        out: List[tuple] = []
        df_lu = (df["change_pct"] >= self.threshold)
        for code, g in df.groupby("code", sort=False):
            g = g.reset_index(drop=True)
            lu_arr = df_lu.loc[g.index].to_numpy() if False else (g["change_pct"] >= self.threshold).to_numpy()
            n = len(g)
            for i in range(n):
                if not lu_arr[i]:
                    continue
                # 前 cooldown_days 内无涨停 (严格首板)
                lo = max(0, i - self.cooldown_days)
                if lu_arr[lo:i].any():
                    continue
                if (str(code), pd.Timestamp(g["date"].iloc[i])) in success_first:
                    continue  # 已成功的不算 failed
                out.append((str(code), pd.Timestamp(g["date"].iloc[i]), float(g["open"].iloc[i])))
        return out
 (df["change_pct"] >= self.threshold)
        for code, g in df.groupby("code", sort=False):
            g = g.reset_index(drop=True)
            lu_arr = (g["change_pct"] >= self.threshold).to_numpy()
            n = len(g)
            for i in range(n):
                if not lu_arr[i]:
                    continue
                lo = max(0, i - self.cooldown_days)
                if lu_arr[lo:i].any():
                    continue
                if (str(code), pd.Timestamp(g["date"].iloc[i])) in success_first:
                    continue
                out.append((str(code), pd.Timestamp(g["date"].iloc[i]), float(g["open"].iloc[i])))
        return out
