"""E-004 EventSequenceBuilder 测试。"""
from __future__ import annotations

import pandas as pd

from src.event.event_sequence_builder import EventSequenceBuilder
from src.event.limit_up_detector import LimitUpEventDetector
from src.event.main_wave_detector import MainWaveDetector
from src.event.second_board_detector import SecondBoardDetector


class TestEventSequenceBuilder:
    def _builder(self):
        return EventSequenceBuilder(
            limit_up=LimitUpEventDetector(),
            second_board=SecondBoardDetector(n_days=5),
            main_wave=MainWaveDetector(n_days=10, return_threshold=0.15),
        )

    def test_build_outputs_required_columns(self, sample_daily_data):
        out = self._builder().build(sample_daily_data)
        required = {
            "sample_id", "code", "first_date", "second_date",
            "gap_days", "main_wave_date", "period_return", "is_main_wave",
        }
        assert required.issubset(set(out.columns))

    def test_from_config_works(self):
        b = EventSequenceBuilder.from_config({
            "event": {
                "limit_up_threshold": 9.5,
                "exclude_st": True,
                "second_board_days": 3,
                "main_wave_days": 7,
                "main_wave_return": 0.2,
            }
        })
        assert b.limit_up.threshold == 9.5
        assert b.second_board.n_days == 3
        assert b.main_wave.return_threshold == 0.2

    def test_sample_id_unique(self, sample_daily_data):
        out = self._builder().build(sample_daily_data)
        assert out["sample_id"].is_unique
