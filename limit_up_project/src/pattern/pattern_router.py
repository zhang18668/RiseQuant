"""P-003 形态路由器 (Pattern-Cluster v2)

接受一条 20 日回看窗口序列, 返回 (cluster_id, dtw_distance).
- 算到每个 medoid 的 DTW 距离, 取最近的;
- 若最近距离 > 该 cluster 的阈值 -> 视为"陌生形态", cluster_id = -1.

用法
----
::

    router = PatternRouter(prototypes={0: seq0, 1: seq1, ...},
                            dist_thresholds={0: 18.3, 1: 22.1, ...})
    cid, dist = router.route(my_seq)
    # cid == -1 表示陌生
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pattern.dtw import dtw_distance
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatternRouter:
    """硬路由 — 最近 medoid + 距离阈值过滤."""

    prototypes: Dict[int, np.ndarray] = field(default_factory=dict)   # cluster_id -> medoid (T, C)
    dist_thresholds: Dict[int, float] = field(default_factory=dict)    # cluster_id -> 95 分位
    dtw_band: int = 5

    # ------------------------------------------------------------------
    def route(self, sequence: np.ndarray) -> Tuple[int, float]:
        """对单条序列返回 (cluster_id, dtw_distance).

        - cluster_id = -1 表示距离超过任何 cluster 的阈值 (陌生形态)
        - 若 sequence 为 None / 空 -> (-1, inf)
        """
        if sequence is None or len(sequence) == 0:
            return -1, float("inf")
        if not self.prototypes:
            return -1, float("inf")

        best_cid = -1
        best_dist = float("inf")
        for cid, proto in self.prototypes.items():
            d = dtw_distance(sequence, proto, band=self.dtw_band, normalize=True)
            if d < best_dist:
                best_dist = d
                best_cid = int(cid)

        # 与阈值比较
        threshold = float(self.dist_thresholds.get(best_cid, float("inf")))
        if best_dist > threshold:
            return -1, float(best_dist)
        return int(best_cid), float(best_dist)

    # ------------------------------------------------------------------
    def route_batch(
        self,
        sequences: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """批量路由. 返回 DataFrame [sample_key, cluster_id, dtw_dist].

        sequences : dict {sample_key -> ndarray (T, C)}.
        """
        rows: List[dict] = []
        for k, s in sequences.items():
            cid, d = self.route(s)
            rows.append({"sample_key": k, "cluster_id": cid, "dtw_dist": d})
        return pd.DataFrame(rows, columns=["sample_key", "cluster_id", "dtw_dist"])

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "prototypes": {int(k): np.asarray(v) for k, v in self.prototypes.items()},
                "dist_thresholds": {int(k): float(v) for k, v in self.dist_thresholds.items()},
                "dtw_band": int(self.dtw_band),
            }, f)

    @classmethod
    def load(cls, path) -> "PatternRouter":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            prototypes=data["prototypes"],
            dist_thresholds=data["dist_thresholds"],
            dtw_band=int(data.get("dtw_band", 5)),
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_fit_result(cls, fit_result, dtw_band: int = 5) -> "PatternRouter":
        """从 ``ClusterFitResult`` 构造."""
        protos = {int(cid): seq for cid, seq in enumerate(fit_result.medoid_sequences)}
        return cls(
            prototypes=protos,
            dist_thresholds={int(k): float(v) for k, v in fit_result.dist_thresholds.items()},
            dtw_band=dtw_band,
        )
