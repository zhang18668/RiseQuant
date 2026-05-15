"""E-007 数据 Dry-run (Pattern-Cluster v2)

训练之前必跑的快速诊断:
  1) 跑 WashSecondDetector 看有多少 (first, second) 对
  2) 跑 GoldenLabelFilter 看不同 [lo, hi] 区间下的金标准事件数
  3) 按年统计样本分布
  4) 估算每个 cluster 平均能分到多少样本

输出一个 ``DryRunReport`` 对象 (含 dict 形式) + 把报告打印到 logger.
如果金标准事件数 < ``min_events_required`` (默认 300), ``DryRunReport.ok=False``.

调用入口
--------
::

    rep = DataDryRun(lo_hi_bins=[(0.10,0.40),(0.15,0.35),(0.20,0.30)]).summarize(daily)
    print(rep)
    if not rep.ok:
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.event.golden_label_filter import GoldenLabelFilter
from src.event.wash_second_detector import WashSecondDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DryRunReport:
    """Dry-run 输出报告."""

    n_codes: int = 0
    date_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None)
    n_first_boards: int = 0
    n_paired_events: int = 0          # WashSecondDetector 输出条数
    bin_counts: Dict[str, int] = field(default_factory=dict)   # "[0.15,0.35]" -> int
    bin_year_dist: Dict[str, Dict[int, int]] = field(default_factory=dict)
    estimated_min_cluster_size: Dict[str, int] = field(default_factory=dict)  # 按金标准/k*0.8 估
    min_events_required: int = 300
    target_bin_key: str = ""
    ok: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_codes": self.n_codes,
            "date_range": [
                str(self.date_range[0]) if self.date_range[0] is not None else None,
                str(self.date_range[1]) if self.date_range[1] is not None else None,
            ],
            "n_first_boards": self.n_first_boards,
            "n_paired_events": self.n_paired_events,
            "bin_counts": self.bin_counts,
            "bin_year_dist": {k: dict(v) for k, v in self.bin_year_dist.items()},
            "estimated_min_cluster_size": self.estimated_min_cluster_size,
            "min_events_required": self.min_events_required,
            "target_bin_key": self.target_bin_key,
            "ok": self.ok,
            "reasons": list(self.reasons),
        }

    def pretty_print(self) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("数据 Dry-run 报告")
        lines.append("=" * 60)
        if self.date_range[0] is not None:
            lines.append(
                f"日线区间: {self.date_range[0].date()} ~ {self.date_range[1].date()}, "
                f"股票池 {self.n_codes} 只"
            )
        lines.append(f"首板事件总数: {self.n_first_boards}")
        lines.append(f"有 second 配对: {self.n_paired_events}")
        lines.append("金标准事件 (不同 [lo, hi] 区间):")
        for key, cnt in self.bin_counts.items():
            tag = "  ✅ 充足" if cnt >= self.min_events_required else "  ⚠️ 不足, 建议放宽"
            lines.append(f"  {key:<14s} → {cnt:5d} 个  {tag}")
        if self.target_bin_key and self.target_bin_key in self.bin_year_dist:
            yd = self.bin_year_dist[self.target_bin_key]
            lines.append(f"按年分布 ({self.target_bin_key}):")
            lines.append("  " + " / ".join(f"{y}: {c}" for y, c in sorted(yd.items())))
        if self.estimated_min_cluster_size:
            lines.append("假设聚 k 类, 估计最少 cluster 样本数:")
            for k_str, n in self.estimated_min_cluster_size.items():
                lines.append(f"  k={k_str:<3s} → {n}")
        lines.append("-" * 60)
        lines.append(f"OK: {self.ok}")
        if not self.ok:
            for r in self.reasons:
                lines.append(f"  ! {r}")
        lines.append("=" * 60)
        return "\n".join(lines)

    __str__ = pretty_print


@dataclass
class DataDryRun:
    """快速诊断: 给一段日线能产出多少金标准事件."""

    lo_hi_bins: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.10, 0.40), (0.15, 0.35), (0.20, 0.30)]
    )
    horizon: int = 22
    min_events_required: int = 300
    cluster_k_candidates: List[int] = field(default_factory=lambda: [3, 5, 8])
    target_bin: Tuple[float, float] = (0.15, 0.35)
    # detector 参数
    detector_threshold: float = 9.9
    detector_cooldown_days: int = 3
    detector_min_gap: int = 3
    detector_max_gap: int = 30
    detector_exclude_st: bool = True

    # ------------------------------------------------------------------
    def summarize(self, daily_data: pd.DataFrame) -> DryRunReport:
        rep = DryRunReport(min_events_required=self.min_events_required)
        rep.target_bin_key = self._bin_key(*self.target_bin)

        if daily_data is None or daily_data.empty:
            rep.ok = False
            rep.reasons.append("daily_data 为空")
            return rep

        df = daily_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        rep.n_codes = int(df["code"].nunique())
        rep.date_range = (df["date"].min(), df["date"].max())

        # 首板数 (粗算, 不去重)
        lu_mask = df["change_pct"] >= self.detector_threshold
        rep.n_first_boards = int(lu_mask.sum())

        # WashSecondDetector
        det = WashSecondDetector(
            threshold=self.detector_threshold,
            cooldown_days=self.detector_cooldown_days,
            min_gap=self.detector_min_gap,
            max_gap=self.detector_max_gap,
            exclude_st=self.detector_exclude_st,
        )
        events = det.detect(df)
        rep.n_paired_events = int(len(events))

        if events.empty:
            rep.ok = False
            rep.reasons.append("WashSecondDetector 无配对事件 (first→second)")
            return rep

        # 多区间金标准
        for lo, hi in self.lo_hi_bins:
            flt = GoldenLabelFilter(lo=lo, hi=hi, horizon=self.horizon)
            ev = flt.filter(events, df)
            key = self._bin_key(lo, hi)
            n_gold = int(ev["is_golden"].fillna(False).astype(bool).sum())
            rep.bin_counts[key] = n_gold

            # 按年分布
            if n_gold > 0:
                ev_g = ev[ev["is_golden"].fillna(False).astype(bool)].copy()
                ev_g["yr"] = pd.to_datetime(ev_g["second_date"]).dt.year
                rep.bin_year_dist[key] = (
                    ev_g.groupby("yr").size().astype(int).to_dict()
                )

        # 估算每 cluster 最小样本数 (按 target_bin 算)
        n_target = rep.bin_counts.get(rep.target_bin_key, 0)
        for k in self.cluster_k_candidates:
            # 0.8 倍系数: 假设最不平衡的 cluster 占 1/(k * 1.25)
            min_size = int(n_target / max(k, 1) * 0.8)
            rep.estimated_min_cluster_size[str(k)] = min_size

        # ok 判定
        if n_target < self.min_events_required:
            rep.ok = False
            rep.reasons.append(
                f"目标区间 {rep.target_bin_key} 金标准事件数 {n_target} "
                f"< 阈值 {self.min_events_required}"
            )

        return rep

    @staticmethod
    def _bin_key(lo: float, hi: float) -> str:
        return f"[{lo:.2f},{hi:.2f}]"
