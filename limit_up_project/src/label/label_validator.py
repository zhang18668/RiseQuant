"""L-002 标签校验器

输入 :class:`LabelBuilder.build` 的输出，做基本合法性检查：
- ``label_short`` ∈ {0, 1}；
- ``label_combined`` ∈ {0, 1, 2}；
- ``label_short == 0`` 时 ``label_combined`` 必为 0；
- ``sample_id`` 全表唯一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class LabelValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)


class LabelValidator:
    """标签合法性校验。"""

    SHORT_VALUES = {0, 1}
    COMBINED_VALUES = {0, 1, 2}

    def validate(self, labels: pd.DataFrame) -> LabelValidationResult:
        result = LabelValidationResult()
        for col in ("sample_id", "label_short", "label_combined"):
            if col not in labels.columns:
                result.is_valid = False
                result.errors.append(f"missing column: {col}")
        if not result.is_valid:
            return result

        if labels["sample_id"].duplicated().any():
            result.is_valid = False
            result.errors.append("duplicate sample_id")

        bad_short = ~labels["label_short"].isin(self.SHORT_VALUES)
        if bad_short.any():
            result.is_valid = False
            result.errors.append(f"invalid label_short rows: {int(bad_short.sum())}")

        bad_combined = ~labels["label_combined"].isin(self.COMBINED_VALUES)
        if bad_combined.any():
            result.is_valid = False
            result.errors.append(f"invalid label_combined rows: {int(bad_combined.sum())}")

        inconsistent = (labels["label_short"] == 0) & (labels["label_combined"] != 0)
        if inconsistent.any():
            result.is_valid = False
            result.errors.append(
                f"label_short=0 but label_combined!=0: {int(inconsistent.sum())} rows"
            )
        return result
