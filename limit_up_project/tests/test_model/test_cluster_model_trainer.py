"""单测: ClusterModelTrainer (方案 A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from src.model.cluster_model_trainer import ClusterModelTrainer


def _make_synthetic(n_clusters=2, n_per_cluster=200, seed=0):
    """造合成数据: cluster_id 决定标签生成机制, 训练后每个 cluster AUC 应较高."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        for i in range(n_per_cluster):
            f0 = rng.normal()
            f1 = rng.normal()
            # cluster 0: label ~ f0>0; cluster 1: label ~ f1>0
            if c == 0:
                p = 1.0 / (1.0 + np.exp(-2.0 * f0))
            else:
                p = 1.0 / (1.0 + np.exp(-2.0 * f1))
            y = int(rng.random() < p)
            rows.append({
                "sample_id": f"c{c}_s{i}",
                "cluster_id": c,
                "label_pre": y,
                "potential_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                "sample_weight": 1.0,
                "f_w_f0": f0,
                "f_w_f1": f1,
                "f_w_f2": rng.normal(),
            })
    df = pd.DataFrame(rows)
    samples = df[["sample_id", "cluster_id", "label_pre", "potential_date", "sample_weight"]]
    features = df[["sample_id", "f_w_f0", "f_w_f1", "f_w_f2"]]
    return samples, features


def test_train_per_cluster_returns_artifacts():
    samples, features = _make_synthetic(n_clusters=2, n_per_cluster=200, seed=1)
    train_end = pd.Timestamp("2023-04-30")
    valid_end = pd.Timestamp("2023-06-15")
    test_end = pd.Timestamp("2023-12-31")
    tr = ClusterModelTrainer(
        min_train_samples=30, min_pos_ratio=0.05, max_pos_ratio=0.95,
        min_auc_test=0.50, model_params={"n_estimators": 30, "verbose": -1},
    )
    arts = tr.train(samples, features, train_end=train_end, valid_end=valid_end, test_end=test_end)
    assert set(arts.keys()) == {0, 1}
    for cid, art in arts.items():
        assert art.n_train > 30
        assert art.trainer is not None


def test_unusable_when_too_few_samples():
    samples, features = _make_synthetic(n_clusters=1, n_per_cluster=30, seed=2)
    tr = ClusterModelTrainer(
        min_train_samples=100, model_params={"n_estimators": 10, "verbose": -1},
    )
    arts = tr.train(samples, features,
                    train_end=pd.Timestamp("2023-01-15"),
                    valid_end=pd.Timestamp("2023-01-22"),
                    test_end=pd.Timestamp("2023-12-31"))
    assert 0 in arts
    assert arts[0].usable is False
    assert "训练样本数" in arts[0].reason_if_not_usable


def test_skip_negative_cluster_id():
    samples, features = _make_synthetic(n_clusters=2, n_per_cluster=80, seed=3)
    samples = pd.concat([
        samples,
        samples.head(20).assign(cluster_id=-1, sample_id=samples.head(20)["sample_id"] + "_neg"),
    ], ignore_index=True)
    features = pd.concat([
        features,
        features.head(20).assign(sample_id=features.head(20)["sample_id"] + "_neg"),
    ], ignore_index=True)
    tr = ClusterModelTrainer(
        min_train_samples=10, model_params={"n_estimators": 10, "verbose": -1},
    )
    arts = tr.train(samples, features,
                    train_end=pd.Timestamp("2023-02-15"),
                    valid_end=pd.Timestamp("2023-02-22"),
                    test_end=pd.Timestamp("2023-12-31"))
    # cluster_id = -1 不应该出现
    assert -1 not in arts


def test_to_summary_row():
    samples, features = _make_synthetic(n_clusters=1, n_per_cluster=100, seed=4)
    tr = ClusterModelTrainer(
        min_train_samples=20, min_auc_test=0.0,
        model_params={"n_estimators": 10, "verbose": -1},
    )
    arts = tr.train(samples, features,
                    train_end=pd.Timestamp("2023-02-15"),
                    valid_end=pd.Timestamp("2023-02-22"),
                    test_end=pd.Timestamp("2023-12-31"))
    art = arts[0]
    row = art.to_summary_row()
    assert "cluster_id" in row
    assert "n_train" in row
    assert "usable" in row
