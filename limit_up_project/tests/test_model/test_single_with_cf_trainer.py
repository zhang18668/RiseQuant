"""单测: SingleWithCFTrainer (方案 B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from src.model.single_with_cf_trainer import SingleWithCFTrainer


def _make_synthetic(n_clusters=2, n_per_cluster=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        for i in range(n_per_cluster):
            f0 = rng.normal()
            f1 = rng.normal()
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


def test_train_single_with_cf_returns_artifact():
    samples, features = _make_synthetic(seed=1)
    tr = SingleWithCFTrainer(model_params={"n_estimators": 30, "verbose": -1})
    art = tr.train(samples, features,
                   train_end=pd.Timestamp("2023-04-30"),
                   valid_end=pd.Timestamp("2023-06-15"),
                   test_end=pd.Timestamp("2023-12-31"))
    assert art.trainer is not None
    assert art.n_train > 0
    assert "cluster_id" in art.trainer.feature_names_
    # per-cluster metrics 应有 0 和 1
    assert set(art.metrics_per_cluster.keys()).issubset({0, 1})


def test_drop_unknown_default_true():
    samples, features = _make_synthetic(seed=2)
    # 加一批 cluster_id=-1 的样本
    n = 50
    extra_samples = pd.DataFrame({
        "sample_id": [f"x{i}" for i in range(n)],
        "cluster_id": [-1] * n,
        "label_pre": [0] * n,
        "potential_date": [pd.Timestamp("2023-01-01") + pd.Timedelta(days=i) for i in range(n)],
        "sample_weight": [1.0] * n,
    })
    extra_features = pd.DataFrame({
        "sample_id": [f"x{i}" for i in range(n)],
        "f_w_f0": [0.0] * n, "f_w_f1": [0.0] * n, "f_w_f2": [0.0] * n,
    })
    samples_all = pd.concat([samples, extra_samples], ignore_index=True)
    features_all = pd.concat([features, extra_features], ignore_index=True)
    tr = SingleWithCFTrainer(model_params={"n_estimators": 10, "verbose": -1})
    art = tr.train(samples_all, features_all,
                   train_end=pd.Timestamp("2023-04-30"),
                   valid_end=pd.Timestamp("2023-06-15"),
                   test_end=pd.Timestamp("2023-12-31"))
    assert art.n_train + art.n_valid + art.n_test < len(samples_all)


def test_summary_serializable():
    samples, features = _make_synthetic(seed=3)
    tr = SingleWithCFTrainer(model_params={"n_estimators": 10, "verbose": -1})
    art = tr.train(samples, features,
                   train_end=pd.Timestamp("2023-04-30"),
                   valid_end=pd.Timestamp("2023-06-15"),
                   test_end=pd.Timestamp("2023-12-31"))
    import json
    s = art.to_summary()
    json.dumps(s)
