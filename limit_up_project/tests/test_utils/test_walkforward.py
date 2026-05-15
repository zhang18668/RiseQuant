"""单测: WalkForwardScheduler."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.walkforward import WalkForwardScheduler, WalkForwardWindow


def test_basic_6month_windows():
    sch = WalkForwardScheduler(
        start="2022-01-01", end="2024-12-31",
        step_months=6, min_train_months=12,
    )
    wins = sch.generate_windows()
    # 首窗口 test_start ≈ 2023-01-02
    assert len(wins) >= 3
    assert wins[0].train_start == pd.Timestamp("2022-01-01")
    assert wins[0].test_start >= pd.Timestamp("2023-01-01")
    # train 是 expanding
    for i in range(1, len(wins)):
        assert wins[i].train_end > wins[i - 1].train_end
        assert wins[i].train_start == wins[0].train_start


def test_window_id_half():
    sch = WalkForwardScheduler(
        start="2022-01-01", end="2025-12-31",
        step_months=6, min_train_months=12,
    )
    wins = sch.generate_windows()
    ids = [w.window_id for w in wins]
    # 至少有 H1 / H2 之类的 id
    assert any("H1" in i or "H2" in i for i in ids), ids


def test_test_end_clipped_to_end():
    sch = WalkForwardScheduler(
        start="2022-01-01", end="2023-09-30",
        step_months=6, min_train_months=12,
    )
    wins = sch.generate_windows()
    for w in wins:
        assert w.test_end <= pd.Timestamp("2023-09-30")


def test_no_window_if_data_too_short():
    sch = WalkForwardScheduler(
        start="2022-01-01", end="2022-06-30",
        step_months=6, min_train_months=12,
    )
    wins = sch.generate_windows()
    assert wins == []


def test_to_dict_serializable():
    sch = WalkForwardScheduler(
        start="2022-01-01", end="2024-06-30",
        step_months=6, min_train_months=12,
    )
    wins = sch.generate_windows()
    import json
    for w in wins:
        d = w.to_dict()
        json.dumps(d)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        WalkForwardScheduler(start="2022-01-01", end="2024-12-31", step_months=0)
    with pytest.raises(ValueError):
        WalkForwardScheduler(start="2022-01-01", end="2024-12-31", min_train_months=0)
