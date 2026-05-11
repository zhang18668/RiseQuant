"""L-001 LabelBuilder 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.label.label_builder import LabelBuilder


class TestLabelBuilder:
    def test_label_perfect(self):
        b = LabelBuilder()
        assert b.calc_label_combined({"has_second": True, "period_return": 0.2}) == b.LABEL_PERFECT

    def test_label_second_only(self):
        b = LabelBuilder()
        assert b.calc_label_combined({"has_second": True, "period_return": 0.1}) == b.LABEL_SECOND

    def test_label_fail(self):
        b = LabelBuilder()
        assert b.calc_label_combined({"has_second": False}) == b.LABEL_FAIL

    def test_label_short(self):
        b = LabelBuilder()
        assert b.calc_label_short({"has_second": True}) == 1
        assert b.calc_label_short({"has_second": False}) == 0

    def test_build_from_event_df(self):
        b = LabelBuilder(main_wave_return=0.15)
        df = pd.DataFrame({
            "sample_id": ["a", "b", "c"],
            "code": ["000001", "000002", "000003"],
            "first_date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "second_date": pd.to_datetime(["2023-01-04", "2023-01-05", pd.NaT]),
            "period_return": [0.2, 0.1, float("nan")],
            "is_main_wave": [True, False, False],
        })
        out = b.build(df)
        assert out["label_combined"].tolist() == [b.LABEL_PERFECT, b.LABEL_SECOND, b.LABEL_FAIL]
        assert out["label_short"].tolist() == [1, 1, 0]

    def test_build_empty(self):
        out = LabelBuilder().build(pd.DataFrame())
        assert out.empty
