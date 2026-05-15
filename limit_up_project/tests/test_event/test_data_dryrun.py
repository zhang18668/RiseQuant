"""单测: DataDryRun."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.event.data_dryrun import DataDryRun, DryRunReport


def _synth_daily_with_event(
    code: str, dates, first_idx: int, second_idx: int,
    future_pct: float, base: float = 10.0,
):
    """生成一条 (first, second) 事件 + 后续涨幅 future_pct 的合成日线."""
    n = len(dates)
    closes = [base] * n
    cp = [0.0] * n
    # 普通波动
    closes[0] = base
    for i in range(1, n):
        if i == first_idx or i == second_idx:
            cp[i] = 9.95
            closes[i] = closes[i - 1] * 1.0995
        else:
            cp[i] = 0.1
            closes[i] = closes[i - 1] * 1.001
    # second 之后达到 future_pct
    second_close = closes[second_idx]
    target = second_close * (1.0 + future_pct)
    # 把 second 之后第 5 天的 close 设为 target (保证 future_max 命中)
    if second_idx + 5 < n:
        closes[second_idx + 5] = target
    df = pd.DataFrame({
        "date": dates,
        "code": code,
        "close": closes,
        "open": [c * 0.99 for c in closes],
        "high": [c * 1.01 for c in closes],
        "low":  [c * 0.98 for c in closes],
        "volume": [1_000_000.0] * n,
        "change_pct": cp,
        "is_st": [False] * n,
    })
    return df


def test_empty_daily_returns_not_ok():
    rep = DataDryRun().summarize(pd.DataFrame())
    assert rep.ok is False
    assert "daily_data 为空" in rep.reasons[0]


def test_no_paired_events_returns_not_ok():
    # 全部小幅波动, 没有涨停 -> 没有 first/second 对
    dates = pd.bdate_range("2022-01-03", periods=100)
    n = len(dates)
    df = pd.DataFrame({
        "date": dates,
        "code": "A",
        "close": np.linspace(10, 11, n),
        "open":  np.linspace(10, 11, n),
        "high":  np.linspace(10.1, 11.1, n),
        "low":   np.linspace(9.9, 10.9, n),
        "volume": np.ones(n) * 1e6,
        "change_pct": np.zeros(n),
        "is_st": [False] * n,
    })
    rep = DataDryRun(min_events_required=1).summarize(df)
    assert rep.ok is False
    assert any("无配对事件" in r for r in rep.reasons)


def test_golden_count_within_target_bin():
    """合成一只股票多次出 (first, second) + 后续涨 25%, 全部落在 [0.15, 0.35] 区间."""
    dates = pd.bdate_range("2022-01-03", periods=300)
    frames = []
    # 制造 5 个独立事件 (间隔 50 天足够远, 不会互相影响 cooldown)
    for i, base in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
        slot = i * 50
        # first at slot+10, second at slot+15 (gap=5)
        frames.append(_synth_daily_with_event(
            f"A{i:03d}", dates, first_idx=slot + 10, second_idx=slot + 15,
            future_pct=0.25, base=base,
        ))
    df = pd.concat(frames, ignore_index=True)

    dry = DataDryRun(
        lo_hi_bins=[(0.15, 0.35), (0.20, 0.30)],
        target_bin=(0.15, 0.35),
        min_events_required=3,
    )
    rep = dry.summarize(df)
    # 各 code 一个事件 -> 5 个金标准
    assert rep.n_paired_events >= 5
    assert rep.bin_counts["[0.15,0.35]"] >= 5
    # 0.20-0.30 也覆盖 0.25
    assert rep.bin_counts["[0.20,0.30]"] >= 5
    assert rep.ok is True


def test_bin_year_distribution():
    """生成跨年事件, 检查 bin_year_dist."""
    dates = pd.bdate_range("2022-01-03", periods=600)   # 跨 2 年多
    frames = []
    # 5 个事件分布在不同年份
    for i, base in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
        slot = i * 110   # 跨年分布
        frames.append(_synth_daily_with_event(
            f"A{i:03d}", dates, first_idx=slot + 10, second_idx=slot + 15,
            future_pct=0.25, base=base,
        ))
    df = pd.concat(frames, ignore_index=True)
    dry = DataDryRun(
        lo_hi_bins=[(0.15, 0.35)],
        target_bin=(0.15, 0.35),
        min_events_required=1,
    )
    rep = dry.summarize(df)
    yd = rep.bin_year_dist["[0.15,0.35]"]
    assert sum(yd.values()) == rep.bin_counts["[0.15,0.35]"]
    # 至少两个年份
    assert len(yd) >= 2


def test_estimated_min_cluster_size_decreases_with_k():
    dates = pd.bdate_range("2022-01-03", periods=300)
    frames = []
    for i, base in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
        slot = i * 50
        frames.append(_synth_daily_with_event(
            f"A{i:03d}", dates, first_idx=slot + 10, second_idx=slot + 15,
            future_pct=0.25, base=base,
        ))
    df = pd.concat(frames, ignore_index=True)
    dry = DataDryRun(
        lo_hi_bins=[(0.15, 0.35)],
        target_bin=(0.15, 0.35),
        min_events_required=1,
        cluster_k_candidates=[3, 5, 8],
    )
    rep = dry.summarize(df)
    sizes = rep.estimated_min_cluster_size
    assert sizes["3"] >= sizes["5"] >= sizes["8"]


def test_report_dict_serializable():
    rep = DryRunReport(
        n_codes=10,
        date_range=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        n_first_boards=100,
        n_paired_events=20,
        bin_counts={"[0.15,0.35]": 5},
        bin_year_dist={"[0.15,0.35]": {2024: 5}},
    )
    d = rep.to_dict()
    import json
    s = json.dumps(d)
    # roundtrip OK
    d2 = json.loads(s)
    assert d2["n_codes"] == 10


def test_pretty_print_runs():
    rep = DryRunReport(
        n_codes=10,
        date_range=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        n_first_boards=100,
        n_paired_events=20,
        bin_counts={"[0.15,0.35]": 500, "[0.20,0.30]": 100},
        bin_year_dist={"[0.15,0.35]": {2024: 500}},
        target_bin_key="[0.15,0.35]",
        estimated_min_cluster_size={"3": 100, "5": 80},
        ok=True,
    )
    s = rep.pretty_print()
    assert "数据 Dry-run 报告" in s
    assert "[0.15,0.35]" in s
