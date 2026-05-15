"""U-006 WalkForwardScheduler — Pattern-Cluster v2

按月滚动生成训练窗口:
- 每个窗口的 train 区间从 ``start`` 开始, 不断扩展 (expanding); test 区间是接下来的 ``step_months``;
- 首窗口的 train 至少有 ``min_train_months`` 个月.

输出
----
``List[WalkForwardWindow]``, 每个含 ``train_start, train_end, test_start, test_end,
window_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class WalkForwardWindow:
    window_id: str            # e.g. "2024H1"
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
        }


@dataclass
class WalkForwardScheduler:
    """生成 walk-forward 窗口."""

    start: pd.Timestamp
    end: pd.Timestamp
    step_months: int = 6
    min_train_months: int = 12

    def __post_init__(self) -> None:
        self.start = pd.Timestamp(self.start)
        self.end = pd.Timestamp(self.end)
        if self.step_months < 1:
            raise ValueError("step_months >= 1")
        if self.min_train_months < 1:
            raise ValueError("min_train_months >= 1")

    # ------------------------------------------------------------------
    def generate_windows(self) -> List[WalkForwardWindow]:
        windows: List[WalkForwardWindow] = []
        # 首窗口: train_end = start + min_train_months
        train_end = self.start + pd.DateOffset(months=self.min_train_months)
        while True:
            test_start = train_end + pd.Timedelta(days=1)
            test_end = train_end + pd.DateOffset(months=self.step_months)
            if test_start > self.end:
                break
            if test_end > self.end:
                test_end = self.end
            wid = self._window_id(test_start, test_end)
            windows.append(WalkForwardWindow(
                window_id=wid,
                train_start=self.start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            # 滚动一个 step
            train_end = train_end + pd.DateOffset(months=self.step_months)
            if train_end >= self.end:
                break
        return windows

    @staticmethod
    def _window_id(start: pd.Timestamp, end: pd.Timestamp) -> str:
        """根据测试区间生成可读 id, e.g. 2024H1 / 2024Q3 / 2024M07."""
        # 半年制
        if start.month in (1, 7) and (end.month - start.month) in (5, 11, -7):
            half = "H1" if start.month == 1 else "H2"
            return f"{start.year}{half}"
        # 季度
        if (end.month - start.month) in (2, -10):
            q = (start.month - 1) // 3 + 1
            return f"{start.year}Q{q}"
        # fallback: YYYYMM
        return f"{start.year}{start.month:02d}"
