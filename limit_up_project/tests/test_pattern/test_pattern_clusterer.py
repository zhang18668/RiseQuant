"""单测: DTW + KMedoids + PatternClusterer."""
from __future__ import annotations

import numpy as np
import pytest

from src.pattern.dtw import dtw_distance, pairwise_dtw
from src.pattern.kmedoids import fit_kmedoids, silhouette_score_precomputed
from src.pattern.pattern_clusterer import PatternClusterer


# ============================== DTW =====================================

def test_dtw_identical_zero():
    a = np.ones((20, 3), dtype=np.float32)
    assert dtw_distance(a, a, band=5) == pytest.approx(0.0, abs=1e-6)


def test_dtw_symmetric():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 4)).astype(np.float32)
    b = rng.normal(size=(20, 4)).astype(np.float32)
    d1 = dtw_distance(a, b, band=5)
    d2 = dtw_distance(b, a, band=5)
    assert d1 == pytest.approx(d2, abs=1e-6)


def test_dtw_shifted_sequence_smaller_than_random():
    # 一条序列 + 它的时间平移版本 — DTW 距离应该比纯随机的小
    rng = np.random.default_rng(1)
    base = rng.normal(size=(20, 1)).astype(np.float32)
    # 平移 2 步 (shift right by 2)
    shifted = np.concatenate([base[2:], base[:2]], axis=0)
    rand = rng.normal(size=(20, 1)).astype(np.float32)
    d_shift = dtw_distance(base, shifted, band=5)
    d_rand = dtw_distance(base, rand, band=5)
    assert d_shift < d_rand


def test_dtw_band_smaller_or_equal_to_full():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(15, 2)).astype(np.float32)
    b = rng.normal(size=(15, 2)).astype(np.float32)
    d_full = dtw_distance(a, b, band=None)
    d_band = dtw_distance(a, b, band=3)
    # 带宽限制只会让距离 >= 全 DTW (因为禁止某些路径)
    assert d_band >= d_full - 1e-6


def test_dtw_channel_mismatch_raises():
    a = np.zeros((5, 3))
    b = np.zeros((5, 2))
    with pytest.raises(ValueError):
        dtw_distance(a, b)


def test_pairwise_dtw_shape_and_diag():
    rng = np.random.default_rng(3)
    seqs = [rng.normal(size=(20, 2)).astype(np.float32) for _ in range(5)]
    M = pairwise_dtw(seqs, band=5)
    assert M.shape == (5, 5)
    # 对角线 = 0, 对称
    for i in range(5):
        assert M[i, i] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(M, M.T)


# ============================== KMedoids =====================================

def test_kmedoids_recovers_clusters():
    # 3 个明显分离的簇
    rng = np.random.default_rng(4)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    points = np.vstack([
        centers[c] + rng.normal(0, 0.5, size=(20, 2))
        for c in range(3)
    ])
    # 用欧氏距离做距离矩阵
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(points[i] - points[j])

    labels, medoids, inertia = fit_kmedoids(D, k=3, random_state=42)
    # 同一真实簇的点应该同 label
    for true_c in range(3):
        true_mask = slice(true_c * 20, (true_c + 1) * 20)
        labs = labels[true_mask]
        # 簇内 label 多数应当一致 (允许少数离群)
        most_common = np.bincount(labs).max()
        assert most_common >= 18, f"cluster {true_c} 多数被一致归类失败"


def test_kmedoids_k_too_large_raises():
    D = np.zeros((4, 4))
    with pytest.raises(ValueError):
        fit_kmedoids(D, k=5)


def test_silhouette_high_for_separated():
    rng = np.random.default_rng(5)
    centers = np.array([[0, 0], [10, 0]], dtype=float)
    points = np.vstack([
        centers[c] + rng.normal(0, 0.3, size=(15, 2)) for c in range(2)
    ])
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(points[i] - points[j])
    labels = np.array([0] * 15 + [1] * 15)
    sil = silhouette_score_precomputed(D, labels)
    assert sil > 0.8


def test_silhouette_single_cluster_returns_nan():
    D = np.zeros((4, 4))
    labels = np.array([0, 0, 0, 0])
    sil = silhouette_score_precomputed(D, labels)
    assert np.isnan(sil)


# ============================== PatternClusterer =====================================

def _make_pattern_sequences(rng, n_per_cluster: int = 10):
    """造 3 种明显不同的形态: 上行 / 横盘 / 下行."""
    seqs = []
    keys = []
    # 上行
    for i in range(n_per_cluster):
        t = np.linspace(0, 1, 20)
        sig = t[:, None] + 0.05 * rng.normal(size=(20, 1))
        sig = np.concatenate([sig, np.zeros((20, 1))], axis=1)
        seqs.append(sig.astype(np.float32))
        keys.append(f"up_{i}")
    # 横盘
    for i in range(n_per_cluster):
        sig = 0.05 * rng.normal(size=(20, 1)) + 0.5
        sig = np.concatenate([sig, np.zeros((20, 1))], axis=1)
        seqs.append(sig.astype(np.float32))
        keys.append(f"flat_{i}")
    # 下行
    for i in range(n_per_cluster):
        t = np.linspace(1, 0, 20)
        sig = t[:, None] + 0.05 * rng.normal(size=(20, 1))
        sig = np.concatenate([sig, np.zeros((20, 1))], axis=1)
        seqs.append(sig.astype(np.float32))
        keys.append(f"down_{i}")
    return seqs, keys


def test_clusterer_finds_three_clusters():
    rng = np.random.default_rng(10)
    seqs, keys = _make_pattern_sequences(rng, n_per_cluster=10)
    pc = PatternClusterer(k_range=(2, 5), kmedoids_init_n=5)
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    # 应该选到 3 类
    assert res.best_k == 3
    # 每类约 10 个
    for c in range(3):
        assert 5 <= res.cluster_sizes[c] <= 15


def test_clusterer_fixed_k():
    rng = np.random.default_rng(11)
    seqs, keys = _make_pattern_sequences(rng, n_per_cluster=8)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=3)
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    assert res.best_k == 2
    assert len(res.medoid_sequences) == 2
    assert set(res.dist_thresholds.keys()) == {0, 1}


def test_clusterer_train_only_must_be_true():
    rng = np.random.default_rng(12)
    seqs, _ = _make_pattern_sequences(rng, 5)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=2)
    with pytest.raises(AssertionError):
        pc.fit(seqs, train_only=False)


def test_clusterer_empty_raises():
    pc = PatternClusterer(k_fixed=2)
    with pytest.raises(ValueError):
        pc.fit([], train_only=True)


def test_clusterer_dist_thresholds_finite():
    rng = np.random.default_rng(13)
    seqs, _ = _make_pattern_sequences(rng, 8)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=3)
    res = pc.fit(seqs, train_only=True)
    for c, t in res.dist_thresholds.items():
        assert np.isfinite(t) or t == float("inf")


def test_to_summary_serializable():
    rng = np.random.default_rng(14)
    seqs, _ = _make_pattern_sequences(rng, 6)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=2)
    res = pc.fit(seqs, train_only=True)
    s = res.to_summary()
    import json
    json.dumps(s)   # 不抛错
