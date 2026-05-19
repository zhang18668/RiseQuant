"""DataValidatorPro — sanity checks on cleaned daily bars.

Returns a dict report. Failed = anything that should block downstream training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class DataValidatorPro:
    min_codes: int = 50
    min_total_rows: int = 10_000
    max_missing_dates_ratio: float = 0.1

    def validate(self, df: pd.DataFrame) -> Dict:
        problems: List[str] = []
        if df is None or df.empty:
            return {"ok": False, "problems": ["empty dataframe"]}

        n_codes = int(df["code"].nunique())
        n_rows = len(df)
        date_min, date_max = df["date"].min(), df["date"].max()
        expected_days = pd.bdate_range(date_min, date_max)
        per_code_rows = df.groupby("code").size()
        # 检查每个 code 的覆盖度
        covered_ratio = (per_code_rows / len(expected_days)).clip(upper=1.0)
        n_low_coverage = int((covered_ratio < (1 - self.max_missing_dates_ratio)).sum())

        if n_codes < self.min_codes:
            problems.append(f"codes={n_codes} < {self.min_codes}")
        if n_rows < self.min_total_rows:
            problems.append(f"rows={n_rows} < {self.min_total_rows}")
        if n_low_coverage > n_codes * 0.5:
            problems.append(
                f"more than 50% of codes have < {(1 - self.max_missing_dates_ratio):.0%} date coverage"
            )

        return {
            "ok": len(problems) == 0,
            "problems": problems,
            "n_codes": n_codes,
            "n_rows": n_rows,
            "date_min": str(date_min.date()) if not pd.isna(date_min) else None,
            "date_max": str(date_max.date()) if not pd.isna(date_max) else None,
            "expected_business_days": len(expected_days),
            "n_codes_low_coverage": n_low_coverage,
        }
