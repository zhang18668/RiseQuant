"""L-001 标签构建器

支持两种标签：
- ``label_short``: 0/1, 表示首板后能否成功"打到二板"。
- ``label_combined``: 0/1/2,
    - 0: 二板失败；
    - 1: 二板成功但主升浪 (二板后 ``main_wave_days`` 内涨幅) 不足；
    - 2: 二板成功且主升浪 (涨幅 >= ``main_wave_return``)。

输入是 :class:`EventSequenceBuilder` 产出的样本表 + 日线数据。也可以接受
更简单的 ``{has_second, period_return}`` 格式 dict 来计算单条标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class LabelBuilder:
    """复合标签构建器。"""

    LABEL_FAIL: int = 0
    LABEL_SECOND: int = 1
    LABEL_PERFECT: int = 2

    second_board_days: int = 5
    main_wave_days: int = 10
    main_wave_return: float = 0.15

    # ------------------------------------------------------------------
    def calc_label_short(self, row: dict) -> int:
        """二板是否成功 (布尔标签)。"""
        return 1 if bool(row.get("has_second", False)) else 0

    def calc_label_combined(self, row: dict) -> int:
        """复合标签。"""
        if not row.get("has_second", False):
            return self.LABEL_FAIL
        period_return = row.get("period_return")
        if period_return is None or pd.isna(period_return):
            return self.LABEL_SECOND
        if float(period_return) >= self.main_wave_return:
            return self.LABEL_PERFECT
        return self.LABEL_SECOND

    # ------------------------------------------------------------------
    def build(
        self,
        event_data: pd.DataFrame,
        daily_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """从样本事件表构建标签。

        ``event_data`` 推荐使用 :meth:`EventSequenceBuilder.build` 的输出，
        但任何包含 ``sample_id, code, first_date, second_date, period_return``
        的 DataFrame 都可以。

        Returns
        -------
        DataFrame
            列 ``sample_id, code, first_date, label_short, label_combined``。
        """
        if event_data is None or event_data.empty:
            return pd.DataFrame(columns=[
                "sample_id", "code", "first_date", "label_short", "label_combined",
            ])

        df = event_data.copy()
        if "has_second" not in df.columns:
            # second_date 非空即视为二板成功
            df["has_second"] = df.get("second_date").notna() if "second_date" in df.columns else False
        if "period_return" not in df.columns:
            df["period_return"] = float("nan")
        if "sample_id" not in df.columns:
            df["sample_id"] = df["code"].astype(str) + "_" + pd.to_datetime(df["first_date"]).dt.strftime("%Y%m%d")

        df["label_short"] = df["has_second"].astype(bool).astype(int)
        df["label_combined"] = df.apply(
            lambda r: self.calc_label_combined({
                "has_second": bool(r["has_second"]),
                "period_return": r["period_return"],
            }),
            axis=1,
        )
        return df[["sample_id", "code", "first_date", "label_short", "label_combined"]].reset_index(drop=True)
