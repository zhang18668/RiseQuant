"""DTW (Dynamic Time Warping) 距离 — 纯 numpy 实现.

针对本项目的回看窗口序列 (固定长度 20, 多通道), 用 Sakoe-Chiba 带宽限制路径,
返回标准 DTW 距离 (各通道欧氏距离的累积).

设计选择
--------
- **多通道**: 距离 = 各时间步两两点之间的多维欧氏距离;
- **band**: Sakoe-Chiba 带宽限制 (默认 5), 避免极端 warp + 加速;
- **normalize**: 返回 sqrt(累计距离 / 路径长度), 让不同长度可比 (本项目长度都=20, 不必须).

精度 vs. 速度
-------------
20 帧 × 10 通道两两 DTW: 大约 0.1 ms / pair. 1000 个样本 → 500k pair → 50s,
可接受. 若 > 3000 样本, 上层会先做随机降采样到 3000.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


INF = float("inf")


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    band: Optional[int] = 5,
    normalize: bool = True,
) -> float:
    """计算两条多通道序列的 DTW 距离.

    Parameters
    ----------
    a, b : ndarray (T, C) — 等长或不等长均可.
    band : int or None — Sakoe-Chiba 带宽, None 表示不限制.
    normalize : bool — True 时返回 sqrt(D / path_len).

    Returns
    -------
    float — DTW 距离 (越小越相似).
    """
    if a is None or b is None:
        return INF
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"channel mismatch: {a.shape[1]} vs {b.shape[1]}")

    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return INF

    # 两两距离平方 (向量化)
    # |a[i] - b[j]|^2 = sum_c (a[i,c]-b[j,c])^2
    # 用广播
    diff = a[:, None, :] - b[None, :, :]      # (n, m, C)
    sqd = np.sum(diff * diff, axis=2)         # (n, m)

    # DP 表
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    if band is None:
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = sqd[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    else:
        b_int = int(band)
        for i in range(1, n + 1):
            # 让 j 落在 [i - band, i + band] 内 (与序列长度对齐)
            j_lo = max(1, i - b_int)
            j_hi = min(m, i + b_int)
            for j in range(j_lo, j_hi + 1):
                D[i, j] = sqd[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    dist = D[n, m]
    if not np.isfinite(dist):
        return INF
    if normalize:
        return float(np.sqrt(dist / (n + m)))
    return float(np.sqrt(dist))


def pairwise_dtw(
    sequences: list,
    band: Optional[int] = 5,
    normalize: bool = True,
) -> np.ndarray:
    """对一个序列列表计算两两 DTW 距离矩阵 (对称)."""
    n = len(sequences)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(sequences[i], sequences[j], band=band, normalize=normalize)
            M[i, j] = d
            M[j, i] = d
    return M
