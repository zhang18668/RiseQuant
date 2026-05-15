"""Sample builder for the wash_ambush strategy.

This strategy trains on earlier entries: positive samples are days that are
still a few sessions before the second limit-up, so the model learns pullback
ambush points instead of last-minute chase points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.event.wash_second_detector import WashSecondDetector
from src.event.wash_sample_builder import SAMPLE_COLS


@dataclass
class WashAmbushSampleBuilder:
    positive_min_lead: int = 2
    positive_max_lead: int = 6
    min_gap_for_neg: int = 3
    max_gap_for_neg: int = 30
    threshold: float = 9.9
    cooldown_days: int = 3

    def build(
        self,
        daily_data: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        include_negatives: bool = True,
    ) -> pd.DataFrame:
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
        code_to_dates = {c: g["date"].tolist() for c, g in df.groupby("code")}

        rows: List[dict] = []
        positive_keys = set()
        events_by_code = {c: g for c, g in events.groupby("code")} if events is not None and not events.empty else {}

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
                for t_idx in range(first_idx + self.min_gap_for_neg, second_idx):
                    t = dates[t_idx]
                    days_to_second = second_idx - t_idx
                    label = 1 if self.positive_min_lead <= days_to_second <= self.positive_max_lead else 0
                    weight = 1.0 / max(days_to_second, 1) if label == 1 else 0.5
                    rows.append({
                        "sample_id": f"{code}_{first_d.strftime('%Y%m%d')}_{t.strftime('%Y%m%d')}",
                        "code": code,
                        "first_date": first_d,
                        "potential_date": t,
                        "second_date": second_d,
                        "gap_so_far": t_idx - first_idx,
                        "days_to_second": days_to_second,
                        "label_pre": label,
                        "sample_weight": weight,
                        "stop_loss_price": float(ev["stop_loss_price"]),
                        "first_open": float(ev["first_open"]),
                        "first_close": float(ev["first_close"]),
                    })
                    positive_keys.add((code, first_d, t))

        if include_negatives:
            success_first = set()
            if events is not None and not events.empty:
                for _, ev in events.iterrows():
                    success_first.add((ev["code"], pd.Timestamp(ev["first_date"])))

            for code, first_d, first_open in self._find_failed_firsts(df, success_first):
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
                        "sample_id": f"{code}_{first_d.strftime('%Y%m%d')}_{t.strftime('%Y%m%d')}_neg",
                        "code": code,
                        "first_date": first_d,
                        "potential_date": t,
                        "second_date": pd.NaT,
                        "gap_so_far": t_idx - first_idx,
                        "days_to_second": -1,
                        "label_pre": 0,
                        "sample_weight": 1.0,
                        "stop_loss_price": first_open,
                        "first_open": first_open,
                        "first_close": float("nan"),
                    })

        if not rows:
            return pd.DataFrame(columns=SAMPLE_COLS)
        return pd.DataFrame(rows)[SAMPLE_COLS].sort_values(
            ["code", "first_date", "potential_date"]
        ).reset_index(drop=True)

    def _find_failed_firsts(self, df: pd.DataFrame, success_first: set) -> List[tuple]:
        out: List[tuple] = []
        for code, g in df.groupby("code", sort=False):
            g = g.reset_index(drop=True)
            lu_arr = (g["change_pct"] >= self.threshold).to_numpy()
            for i, is_lu in enumerate(lu_arr):
                if not is_lu:
                    continue
                lo = max(0, i - self.cooldown_days)
                if lu_arr[lo:i].any():
                    continue
                first_d = pd.Timestamp(g["date"].iloc[i])
                if (str(code), first_d) in success_first:
                    continue
                out.append((str(code), first_d, float(g["open"].iloc[i])))
        return out
