"""DataCleaner: standard cleaning on raw daily bars.

Pipeline (each step optional):
  1) drop rows with null date/code/close
  2) ensure dtypes
  3) recompute change_pct if missing
  4) drop invalid OHLCV (close<=0, NaN ohlc)
  5) drop ST rows (optional)
  6) drop |change_pct|>30 (preserve NaN — those are first-day rows)
  7) dedup (code, date)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLS = ["date", "code", "open", "high", "low", "close", "volume"]


@dataclass
class CleanReport:
    n_input: int = 0
    n_output: int = 0
    n_dropped_null: int = 0
    n_dropped_st: int = 0
    n_dropped_invalid_ohlc: int = 0
    n_dropped_abnormal_change: int = 0
    n_dropped_dup: int = 0
    n_recomputed_change_pct: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_input": int(self.n_input),
            "n_output": int(self.n_output),
            "n_dropped_null": int(self.n_dropped_null),
            "n_dropped_st": int(self.n_dropped_st),
            "n_dropped_invalid_ohlc": int(self.n_dropped_invalid_ohlc),
            "n_dropped_abnormal_change": int(self.n_dropped_abnormal_change),
            "n_dropped_dup": int(self.n_dropped_dup),
            "n_recomputed_change_pct": int(self.n_recomputed_change_pct),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        return (
            f"CleanReport(in={self.n_input}, out={self.n_output}, "
            f"null={self.n_dropped_null}, st={self.n_dropped_st}, "
            f"ohlc={self.n_dropped_invalid_ohlc}, "
            f"abnchg={self.n_dropped_abnormal_change}, "
            f"dup={self.n_dropped_dup}, recomp={self.n_recomputed_change_pct})"
        )


@dataclass
class DataCleaner:
    exclude_st: bool = True
    max_abs_change_pct: float = 30.0
    recompute_change_pct_if_missing: bool = True

    def clean(self, df):
        rep = CleanReport()
        if df is None or df.empty:
            return pd.DataFrame(), rep
        rep.n_input = len(df)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"missing required cols: {missing}")

        out = df.copy()
        before = len(out)
        out = out.dropna(subset=["date", "code", "close"])
        rep.n_dropped_null = before - len(out)

        out["date"] = pd.to_datetime(out["date"])
        for c in ["open", "high", "low", "close", "volume"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        if "change_pct" not in out.columns or (
            self.recompute_change_pct_if_missing
            and out["change_pct"].isna().mean() > 0.5
        ):
            out = out.sort_values(["code", "date"])
            out["change_pct"] = out.groupby("code")["close"].pct_change() * 100.0
            rep.n_recomputed_change_pct = int(out["change_pct"].notna().sum())

        before = len(out)
        ok = (
            (out["close"] > 0)
            & (out["volume"] >= 0)
            & out[["open", "high", "low", "close"]].notna().all(axis=1)
        )
        out = out[ok]
        rep.n_dropped_invalid_ohlc = before - len(out)

        if self.exclude_st and "is_st" in out.columns:
            before = len(out)
            out = out[~out["is_st"].astype(bool).fillna(False)]
            rep.n_dropped_st = before - len(out)

        if "change_pct" in out.columns:
            before = len(out)
            mask = out["change_pct"].isna() | (
                out["change_pct"].abs() <= self.max_abs_change_pct
            )
            out = out[mask]
            rep.n_dropped_abnormal_change = before - len(out)

        before = len(out)
        out = out.sort_values(["code", "date"]).drop_duplicates(
            ["code", "date"], keep="last"
        )
        rep.n_dropped_dup = before - len(out)

        out = out.reset_index(drop=True)
        rep.n_output = len(out)
        logger.info(str(rep))
        return out, rep
