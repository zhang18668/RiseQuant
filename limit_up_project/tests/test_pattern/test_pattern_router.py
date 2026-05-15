"""单测: PatternRouter."""
from __future__ import annotations

import numpy as np
import pytest

from src.pattern.pattern_clusterer import PatternClusterer
from src.pattern.pattern_router import PatternRouter


def _mk_seqs(rng, n_per: int = 8):
    seqs, keys = [], []
    # 上行
    for i in range(n_per):
        s = np.linspace(0, 1, 20)[:, None] + 0.03 * rng.normal(size=(20, 1))
        seqs.append(np.concatenate([s, np.zeros((20, 1))], axis=1).astype(np.float32))
        keys.append(f"up_{i}")
    # 下行
    for i in range(n_per):
        s = np.linspace(1, 0, 20)[:, None] + 0.03 * rng.normal(size=(20, 1))
        seqs.append(np.concatenate([s, np.zeros((20, 1))], axis=1).astype(np.float32))
        keys.append(f"dn_{i}")
    return seqs, keys


def test_route_picks_nearest_cluster():
    rng = np.random.default_rng(20)
    seqs, keys = _mk_seqs(rng, 8)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=3)
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    router = PatternRouter.from_fit_result(res)

    # 用一条明显的上行序列, 应该归到与 up 同类
    up_test = np.concatenate([
        np.linspace(0, 1, 20)[:, None],
        np.zeros((20, 1)),
    ], axis=1).astype(np.float32)
    cid_up, d_up = router.route(up_test)
    # 用下行
    dn_test = np.concatenate([
        np.linspace(1, 0, 20)[:, None],
        np.zeros((20, 1)),
    ], axis=1).astype(np.float32)
    cid_dn, d_dn = router.route(dn_test)
    assert cid_up != -1
    assert cid_dn != -1
    assert cid_up != cid_dn   # 两条对立形态应该分到不同 cluster


def test_route_unknown_returns_minus_one():
    rng = np.random.default_rng(21)
    seqs, keys = _mk_seqs(rng, 6)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=2)
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    router = PatternRouter.from_fit_result(res)
    # 故意把所有阈值压到极小 -> 任何序列都超出
    router.dist_thresholds = {k: 0.0001 for k in router.prototypes.keys()}
    rand = np.random.default_rng(99).normal(size=(20, 2)).astype(np.float32)
    cid, d = router.route(rand)
    assert cid == -1


def test_empty_input_returns_minus_one():
    router = PatternRouter(prototypes={0: np.zeros((20, 2), dtype=np.float32)},
                            dist_thresholds={0: 1e9})
    assert router.route(None) == (-1, float("inf"))
    assert router.route(np.zeros((0, 2))) == (-1, float("inf"))


def test_no_prototypes_returns_minus_one():
    router = PatternRouter()
    assert router.route(np.zeros((20, 2), dtype=np.float32)) == (-1, float("inf"))


def test_route_batch_shape():
    rng = np.random.default_rng(22)
    seqs, keys = _mk_seqs(rng, 6)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=2)
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    router = PatternRouter.from_fit_result(res)

    inputs = {f"new_{i}": seqs[i] for i in range(5)}
    df = router.route_batch(inputs)
    assert list(df.columns) == ["sample_key", "cluster_id", "dtw_dist"]
    assert len(df) == 5
    assert df["cluster_id"].isin([-1, 0, 1]).all()


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(23)
    seqs, keys = _mk_seqs(rng, 6)
    pc = PatternClusterer(k_fixed=2, kmedoids_init_n=2)
    res = pc.fit(seqs, sample_keys=keys, train_only=True)
    router = PatternRouter.from_fit_result(res)
    p = tmp_path / "router.pkl"
    router.save(p)
    r2 = PatternRouter.load(p)
    assert set(r2.prototypes.keys()) == set(router.prototypes.keys())
    # 同一条序列两个 router 给出一致结果
    test_seq = seqs[0]
    assert router.route(test_seq) == r2.route(test_seq)
