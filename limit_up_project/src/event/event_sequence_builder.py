"""E-004 事件序列构建

将三个事件检测器串联，从日线数据出发产生最终的样本表：

``sample_id | code | first_date | second_date | main_wave_date | period_return | label_short | label_combined``

其中 ``label_short`` / ``label_combined`` 的定义见
:class:`src.label.label_builder.LabelBuilder`。本类只负责事件层的串联，
标签的最终构建在标签层完成 -- 但同时这里也提供 :meth:`build`
返回包含核心结果的样本表，方便回测/特征模块直接使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.event.limit_up_detector import LimitUpEventDetector
from src.event.main_wave_detector import MainWaveDetector
from src.event.second_board_detector import SecondBoardDetector


@dataclass
class EventSequenceBuilder:
    """事件序列构建器。"""

    limit_up: LimitUpEventDetector
    second_board: SecondBoardDetector
    main_wave: MainWaveDetector

    @classmethod
    def from_config(cls, config: dict) -> "EventSequenceBuilder":
        ev = config.get("event", config)
        return cls(
            limit_up=LimitUpEventDetector(
                threshold=ev.get("limit_up_threshold", 9.9),
                exclude_st=ev.get("exclude_st", True),
            ),
            second_board=SecondBoardDetector(n_days=ev.get("second_board_days", 5)),
            main_wave=MainWaveDetector(
                n_days=ev.get("main_wave_days", 10),
                return_threshold=ev.get("main_wave_return", 0.15),
            ),
        )

    # ------------------------------------------------------------------
    def build(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """从日线数据出发，输出样本级事件序列表。"""
        limit_up_events = self.limit_up.detect(daily_data)
        second_events = self.second_board.detect(limit_up_events, daily_data)
        main_events = self.main_wave.detect(second_events, daily_data)

        if second_events.empty:
            return self._empty()

        df = second_events.merge(
            main_events,
            on=["code", "second_date"],
            how="left",
        )
        df["is_main_wave"] = df["is_main_wave"].fillna(False).astype(bool)
        df["sample_id"] = (
            df["code"].astype(str) + "_" + df["first_date"].dt.strftime("%Y%m%d")
        )
        cols = [
            "sample_id", "code", "first_date", "second_date",
            "gap_days", "main_wave_date", "period_return", "is_main_wave",
        ]
        return df[cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "sample_id", "code", "first_date", "second_date",
                "gap_days", "main_wave_date", "period_return", "is_main_wave",
            ]
        )
