"""KMedoids — 基于预计算距离矩阵的 PAM (Partitioning Around Medoids).

API
---
- ``fit(dist_matrix, k, max_iter, random_state)`` -> ``(labels, medoid_indices, inertia)``

inertia = sum over all samples of dist(x, medoid_of_x).

收敛性: 经典 PAM 收敛, 但容易陷入局部最优 — 上层通过多次 random_state
取 inertia 最小的那次.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_kmedoids(
    dist_matrix: np.ndarray,
    k: int,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """PAM-style KMedoids.

    Parameters
    ----------
    dist_matrix : (n, n) 对称距离矩阵, 主对角为 0
    k : 聚类数 (>= 2)
    max_iter : 最大迭代次数
    random_state : 随机种子

    Returns
    -------
    labels : (n,) int — 每个样本的 cluster_id (0..k-1)
    medoid_indices : (k,) int — 各 cluster 的 medoid 在 dist_matrix 中的行号
    inertia : float — sum dist(x, medoid_of_x)
    """
    D = np.asarray(dist_matrix, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"dist_matrix must be square, got {D.shape}")
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError(f"k={k} > n={n}")

    rng = np.random.default_rng(random_state)
    # 初始: 随机选 k 个 medoid
    medoids = rng.choice(n, size=k, replace=False)

    def assign(meds):
        # labels[i] = argmin_k D[i, meds[k]]
        sub = D[:, meds]
        return np.argmin(sub, axis=1)

    labels = assign(medoids)
    # 当前 inertia
    cur_inertia = float(D[np.arange(n), medoids[labels]].sum())

    for _ in range(max_iter):
        # 对每个 cluster 找该簇内 medoid 最优 (使簇内距离和最小的点)
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                # 空簇 — 把最远那个样本拉过来
                far_idx = int(np.argmax(D[np.arange(n), medoids[labels]]))
                new_medoids[c] = far_idx
                continue
            sub = D[np.ix_(members, members)]
            sums = sub.sum(axis=1)
            best_local = int(np.argmin(sums))
            new_medoids[c] = members[best_local]

        new_labels = assign(new_medoids)
        new_inertia = float(D[np.arange(n), new_medoids[new_labels]].sum())
        if new_inertia + 1e-12 >= cur_inertia and np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            break
        medoids = new_medoids
        labels = new_labels
        cur_inertia = new_inertia

    return labels.astype(np.int64), medoids.astype(np.int64), cur_inertia


def silhouette_score_precomputed(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    """轮廓系数 (precomputed 距离). 返回平均轮廓系数, [-1, 1].

    a(i) = 簇内平均距离, b(i) = 距最近其他簇的平均距离;
    s(i) = (b - a) / max(a, b); 总分 = mean s(i).
    """
    D = np.asarray(dist_matrix, dtype=np.float64)
    n = D.shape[0]
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    if len(uniq) < 2:
        return float("nan")
    s = np.zeros(n)
    for i in range(n):
        same = (labs == labs[i])
        same[i] = False
        if not same.any():
            s[i] = 0.0
            continue
        a = D[i, same].mean()
        b_min = float("inf")
        for c in uniq:
            if c == labs[i]:
                continue
            mask = labs == c
            if not mask.any():
                continue
            b_c = D[i, mask].mean()
            if b_c < b_min:
                b_min = b_c
        denom = max(a, b_min)
        s[i] = (b_min - a) / denom if denom > 0 else 0.0
    return float(s.mean())
