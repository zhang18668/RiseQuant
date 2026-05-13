"""E-004 事件序列构建

把三个事件检测器串成最终的样本表:

``sample_id | code | first_date | second_date | gap_days | main_wave_date | period_return | is_main_wave | label_short | label_combined``

两种使用模式
--------------
1) 一步到位 -- 从日线数据走完整链路::

       builder = EventSequenceBuilder.from_config(cfg)   # 或 EventSequenceBuilder()
       samples = builder.build(daily_data)

2) 已有预先检测好的事件表, 只做合并 (pipeline 用法)::

       builder = EventSequenceBuilder()
       samples = builder.build(first_board_events, second_board_events, main_wave_events)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.event.limit_up_detector import LimitUpEventDetector
from src.event.main_wave_detector import MainWaveDetector
from src.event.second_board_detector import SecondBoardDetector


SAMPLE_COLUMNS = [
    "sample_id", "code", "first_date", "second_date",
    "gap_days", "main_wave_date", "period_return", "is_main_wave",
    "label_short", "label_combined",
]


class EventSequenceBuilder:
    """事件序列构建器."""

    # 复合标签常量 (与 src.label.label_builder.LabelBuilder 保持一致)
    LABEL_FAIL = 0
    LABEL_SECOND = 1
    LABEL_PERFECT = 2

    def __init__(
        self,
        limit_up: Optional[LimitUpEventDetector] = None,
        second_board: Optional[SecondBoardDetector] = None,
        main_wave: Optional[MainWaveDetector] = None,
        main_wave_return: float = 0.15,
    ) -> None:
        self.limit_up = limit_up or LimitUpEventDetector()
        self.second_board = second_board or SecondBoardDetector()
        self.main_wave = main_wave or MainWaveDetector()
        self.main_wave_return = float(main_wave_return)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "EventSequenceBuilder":
        ev = config.get("event", config) if isinstance(config, dict) else {}
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
            main_wave_return=ev.get("main_wave_return", 0.15),
        )

    # ------------------------------------------------------------------
    def build(
        self,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """重载: 接受 ``daily_data`` 或 ``(first_board, second_board, main_wave)``."""
        # 命名参数优先
        first = kwargs.get("first_board_events")
        second = kwargs.get("second_board_events")
        main = kwargs.get("main_wave_events")
        daily = kwargs.get("daily_data")

        if daily is not None:
            return self._build_from_daily(daily)
        if first is not None and second is not None:
            return self._build_from_events(first, second, main)

        # 位置参数
        if len(args) == 1:
            return self._build_from_daily(args[0])
        if len(args) >= 2:
            first = args[0]
            second = args[1]
            main = args[2] if len(args) >= 3 else None
            return self._build_from_events(first, second, main)

        raise TypeError(
            "build() expects daily_data, or (first_board_events, second_board_events[, main_wave_events])"
        )

    # ------------------------------------------------------------------
    def _build_from_daily(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        limit_up_events = self.limit_up.detect(daily_data)
        # 若有 limit_up_type 列, 仅取首板作为 "起点"
        if "limit_up_type" in limit_up_events.columns:
            first_board = limit_up_events[
                limit_up_events["limit_up_type"] == "first_board"
            ].copy()
        else:
            first_board = limit_up_events.copy()
        second_events = self.second_board.detect(limit_up_events, daily_data)
        main_events = self.main_wave.detect(second_events, daily_data)
        return self._build_from_events(first_board, second_events, main_events)

    # ------------------------------------------------------------------
    def _build_from_events(
        self,
        first_board_events: pd.DataFrame,
        second_board_events: pd.DataFrame,
        main_wave_events: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """以 (首板, 二板, 主升浪) 三张事件表为输入合并样本.

        以 ``first_board_events`` (左表) 为起点 left-join 二板, 二板 left-join 主升浪.
        - 未出现二板的首板 -> 二板失败 (label_combined=0);
        - 出现二板但未达主升浪 -> label_combined=1;
        - 二板且主升浪 -> label_combined=2.
        """
        if first_board_events is None or first_board_events.empty:
            return self._empty()

        fb = first_board_events.copy()
        # 兼容: pipeline 可能传入未重命名的 ``date`` 列
        if "first_date" not in fb.columns and "date" in fb.columns:
            fb = fb.rename(columns={"date": "first_date"})
        if "first_date" not in fb.columns:
            return self._empty()

        sb = (
            second_board_events.copy()
            if second_board_events is not None
            else pd.DataFrame(columns=["code", "first_date", "second_date", "gap_days"])
        )
        mw = (
            main_wave_events.copy()
            if main_wave_events is not None
            else pd.DataFrame(columns=["code", "second_date", "main_wave_date",
                                       "period_return", "is_main_wave"])
        )

        fb["first_date"] = pd.to_datetime(fb["first_date"])
        if not sb.empty:
            sb["first_date"] = pd.to_datetime(sb["first_date"])
            sb["second_date"] = pd.to_datetime(sb["second_date"])
        if not mw.empty:
            mw["second_date"] = pd.to_datetime(mw["second_date"])

        df = fb[["code", "first_date"]].drop_duplicates().merge(
            sb[["code", "first_date", "second_date", "gap_days"]] if not sb.empty
            else pd.DataFrame(columns=["code", "first_date", "second_date", "gap_days"]),
            on=["code", "first_date"],
            how="left",
        )
        df = df.merge(
            mw[["code", "second_date", "main_wave_date", "period_return", "is_main_wave"]]
            if not mw.empty
            else pd.DataFrame(columns=["code", "second_date", "main_wave_date",
                                       "period_return", "is_main_wave"]),
            on=["code", "second_date"],
            how="left",
        )

        df["is_main_wave"] = df["is_main_wave"].astype("boolean").fillna(False).astype(bool)
        df["has_second"] = df["second_date"].notna()
        df["sample_id"] = (
            df["code"].astype(str) + "_" + df["first_date"].dt.strftime("%Y%m%d")
        )

        # 标签 ------------------------------------------------------------
        def _lbl_combined(r):
            if not r["has_second"]:
                return self.LABEL_FAIL
            pr = r.get("period_return")
            if pr is None or pd.isna(pr):
                return self.LABEL_SECOND
            return self.LABEL_PERFECT if float(pr) >= self.main_wave_return else self.LABEL_SECOND

        df["label_short"] = df["has_second"].astype(int)
        df["label_combined"] = df.apply(_lbl_combined, axis=1)

        df = df[SAMPLE_COLUMNS].sort_values(
            ["code", "first_date"]
        ).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
