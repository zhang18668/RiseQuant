"""P-002 形态聚类器 (DTW + KMedoids) — Pattern-Cluster v2

- 输入: 一批 20 日回看窗口序列 (List[ndarray (20, 10)]) — **必须全部来自 train 切片**.
- 内部:
    1) 计算两两 DTW 距离矩阵 (Sakoe-Chiba band=5)
    2) silhouette 自动选 k (3..8), 也支持手动固定
    3) PAM KMedoids, 多次随机初始化取 inertia 最小
    4) 算每个 cluster 的 95 分位 DTW 距离作为路由阈值
- 输出: ``ClusterFitResult`` (labels, medoids 序列, 距离阈值, best_k, silhouette)

数据泄露防线
-----------
``fit()`` 接 ``train_only=True`` (默认), 调用时强制校验输入对应日期 ≤ train_end.
具体的日期校验在上层 pipeline 完成 (这里只接序列), 这里只确保 API 上有显式参数.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pattern.dtw import pairwise_dtw
from src.pattern.kmedoids import fit_kmedoids, silhouette_score_precomputed
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterFitResult:
    """``PatternClusterer.fit()`` 的返回值."""

    labels: np.ndarray                  # (n,) cluster_id ∈ [0, k-1]
    medoid_indices: np.ndarray          # (k,) 训练序列内的下标
    medoid_sequences: List[np.ndarray]  # (k,) 每个 cluster 的原型 ndarray (T, C)
    dist_thresholds: Dict[int, float]   # cluster_id -> 95 分位 DTW 距离 (路由用)
    best_k: int
    silhouette: float
    cluster_sizes: Dict[int, int]
    inertia: float
    dist_matrix: Optional[np.ndarray] = None    # (n, n), 调试用, 可选
    sample_keys: Optional[List[str]] = None     # 与 labels 对齐的样本 id, 调试用

    def to_summary(self) -> dict:
        return {
            "best_k": int(self.best_k),
            "silhouette": float(self.silhouette),
            "cluster_sizes": {int(k): int(v) for k, v in self.cluster_sizes.items()},
            "dist_thresholds": {int(k): float(v) for k, v in self.dist_thresholds.items()},
            "inertia": float(self.inertia),
        }


@dataclass
class PatternClusterer:
    """DTW + KMedoids 形态聚类."""

    k_range: Tuple[int, int] = (3, 8)
    k_fixed: Optional[int] = None
    dtw_band: int = 5
    kmedoids_init_n: int = 10
    route_dist_quantile: float = 0.95
    random_state: int = 42
    sample_cap: int = 3000     # 序列超过此数 -> 随机降采样后聚类

    # ------------------------------------------------------------------
    def fit(
        self,
        sequences: List[np.ndarray],
        sample_keys: Optional[List[str]] = None,
        train_only: bool = True,
    ) -> ClusterFitResult:
        """对一批序列做 DTW + KMedoids 聚类.

        Parameters
        ----------
        sequences : 长度 n 的 list, 每个元素 ndarray (T, C).
        sample_keys : 与 sequences 一一对应的 id (e.g. sample_id), 仅用于追溯.
        train_only : 接口标识 — 仅声明 "调用者保证 sequences 全来自 train 切片",
                     这里不能检查 (没有日期信息), 但留作显式契约.
        """
        assert train_only, "PatternClusterer.fit 必须以 train_only=True 调用 (数据泄露防线)"
        if not sequences:
            raise ValueError("sequences 为空")
        n = len(sequences)
        if sample_keys is not None and len(sample_keys) != n:
            raise ValueError("sample_keys 长度与 sequences 不一致")

        # 降采样
        rng = np.random.default_rng(self.random_state)
        if n > self.sample_cap:
            idx = rng.choice(n, size=self.sample_cap, replace=False)
            idx.sort()
            sub_sequences = [sequences[i] for i in idx]
            sub_keys = [sample_keys[i] for i in idx] if sample_keys else None
            logger.info(f"PatternClusterer: 序列 {n} > 上限 {self.sample_cap}, 降采样到 {len(sub_sequences)}")
        else:
            sub_sequences = sequences
            sub_keys = sample_keys

        logger.info(f"计算 DTW 距离矩阵 ({len(sub_sequences)} 序列, band={self.dtw_band})…")
        D = pairwise_dtw(sub_sequences, band=self.dtw_band, normalize=True)

        # 选 k
        if self.k_fixed is not None:
            best_k = int(self.k_fixed)
            best_labels, best_medoids, best_inertia = self._kmedoids_multi(D, best_k)
            best_sil = silhouette_score_precomputed(D, best_labels)
        else:
            lo, hi = self.k_range
            best_sil = -np.inf
            best_k = lo
            best_labels = None
            best_medoids = None
            best_inertia = float("inf")
            for k in range(lo, hi + 1):
                if k >= len(sub_sequences):
                    continue
                labels, medoids, inertia = self._kmedoids_multi(D, k)
                sil = silhouette_score_precomputed(D, labels)
                logger.info(f"  k={k}: silhouette={sil:.4f}, inertia={inertia:.4f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
                    best_medoids = medoids
                    best_inertia = inertia

        if best_labels is None:
            raise RuntimeError("聚类失败 (k_range 与序列数不匹配?)")

        # cluster 统计
        cluster_sizes: Dict[int, int] = {
            int(c): int(np.sum(best_labels == c)) for c in range(best_k)
        }

        # 取每个 cluster 的 medoid 序列
        medoid_sequences = [sub_sequences[i] for i in best_medoids]

        # 路由阈值: 每个 cluster, 取簇内成员到 medoid 的距离 95 分位
        dist_thresholds: Dict[int, float] = {}
        for c in range(best_k):
            members = np.where(best_labels == c)[0]
            if len(members) == 0:
                dist_thresholds[c] = float("inf")
                continue
            dists = D[members, best_medoids[c]]
            dist_thresholds[c] = float(np.quantile(dists, self.route_dist_quantile))

        return ClusterFitResult(
            labels=best_labels.astype(np.int64),
            medoid_indices=best_medoids.astype(np.int64),
            medoid_sequences=medoid_sequences,
            dist_thresholds=dist_thresholds,
            best_k=int(best_k),
            silhouette=float(best_sil),
            cluster_sizes=cluster_sizes,
            inertia=float(best_inertia),
            dist_matrix=D,
            sample_keys=sub_keys,
        )

    # ------------------------------------------------------------------
    def _kmedoids_multi(self, D: np.ndarray, k: int):
        """跑 ``kmedoids_init_n`` 次, 取 inertia 最小."""
        best = None
        for i in range(self.kmedoids_init_n):
            labels, medoids, inertia = fit_kmedoids(
                D, k=k, random_state=self.random_state + i,
            )
            if best is None or inertia < best[2]:
                best = (labels, medoids, inertia)
        return best
