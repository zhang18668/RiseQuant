"""单测: ModelBundle + BundleArchive."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lightgbm = pytest.importorskip("lightgbm")

from src.model.model_bundle import BundleArchive, ModelBundle
from src.model.model_trainer import LimitUpModelTrainer
from src.pattern.pattern_router import PatternRouter


def _make_trainer(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, size=120))
    t = LimitUpModelTrainer({"params": {"n_estimators": 10, "verbose": -1}})
    t.train(X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:], feature_names=list(X.columns))
    return t, X, y


def _make_router():
    seq0 = np.zeros((20, 10), dtype=np.float32)
    seq1 = np.ones((20, 10), dtype=np.float32)
    return PatternRouter(
        prototypes={0: seq0, 1: seq1},
        dist_thresholds={0: 100.0, 1: 100.0},
    )


def test_save_and_load_per_cluster(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="x", run_id="t1")
    router = _make_router()
    t0, _, _ = _make_trainer(0)
    t1, _, _ = _make_trainer(1)
    arch.save_shared(
        feature_cols=list(t0.feature_names_),
        router=router,
        cluster_sizes={0: 50, 1: 60},
        training_data_range=["2022-01-01", "2024-12-31"],
        training_config={"min_gap": 3},
    )
    arch.save_per_cluster(
        trainers={0: t0, 1: t1},
        metrics_per_cluster={0: {"auc": 0.7}, 1: {"auc": 0.65}},
        usability_flags={0: True, 1: True},
        default_min_score_per_cluster={0: 0.6, 1: 0.65},
    )

    # 加载
    bundle = ModelBundle.load(arch.run_dir, bundle_type="per_cluster")
    assert bundle.bundle_type == "per_cluster"
    assert set(bundle.trainers_per_cluster.keys()) == {0, 1}
    assert bundle.usability_flags == {0: True, 1: True}
    assert bundle.default_min_score_per_cluster == {0: 0.6, 1: 0.65}
    assert bundle.feature_cols == list(t0.feature_names_)
    assert set(bundle.router.prototypes.keys()) == {0, 1}


def test_save_and_load_single_with_cf(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="y", run_id="t2")
    router = _make_router()
    t, X, y = _make_trainer(0)
    arch.save_shared(
        feature_cols=list(t.feature_names_),
        router=router,
        cluster_sizes={0: 50, 1: 60},
    )
    arch.save_single_with_cf(
        trainer=t,
        metrics={"auc": 0.71},
        categorical_feature=["cluster_id"],
    )
    bundle = ModelBundle.load(arch.run_dir, bundle_type="single_with_cf")
    assert bundle.bundle_type == "single_with_cf"
    assert bundle.trainer_single is not None
    assert bundle.categorical_feature == ["cluster_id"]


def test_predict_per_cluster(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="z", run_id="t3")
    router = _make_router()
    t0, _, _ = _make_trainer(0)
    t1, _, _ = _make_trainer(1)
    arch.save_shared(feature_cols=list(t0.feature_names_), router=router)
    arch.save_per_cluster(
        trainers={0: t0, 1: t1},
        metrics_per_cluster={0: {}, 1: {}},
        usability_flags={0: True, 1: True},
    )
    bundle = ModelBundle.load(arch.run_dir, bundle_type="per_cluster")

    rng = np.random.default_rng(99)
    X = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f_{i}" for i in range(5)])
    cids = pd.Series([0] * 10 + [1] * 10)
    scores = bundle.predict(X, cluster_ids=cids)
    assert scores.shape == (20,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_predict_per_cluster_skips_unusable(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="u", run_id="t4")
    router = _make_router()
    t0, _, _ = _make_trainer(0)
    t1, _, _ = _make_trainer(1)
    arch.save_shared(feature_cols=list(t0.feature_names_), router=router)
    arch.save_per_cluster(
        trainers={0: t0, 1: t1},
        metrics_per_cluster={0: {}, 1: {}},
        usability_flags={0: True, 1: False},   # cluster_1 不可用
    )
    bundle = ModelBundle.load(arch.run_dir, bundle_type="per_cluster")
    X = pd.DataFrame(np.zeros((10, 5)), columns=[f"f_{i}" for i in range(5)])
    cids = pd.Series([0] * 5 + [1] * 5)
    scores = bundle.predict(X, cluster_ids=cids)
    assert (scores[5:] == 0.0).all()   # cluster_1 不可用, 分数全 0


def test_predict_per_cluster_minus_one_returns_zero(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="m1", run_id="t5")
    router = _make_router()
    t0, _, _ = _make_trainer(0)
    arch.save_shared(feature_cols=list(t0.feature_names_), router=router)
    arch.save_per_cluster(
        trainers={0: t0},
        metrics_per_cluster={0: {}},
        usability_flags={0: True},
    )
    bundle = ModelBundle.load(arch.run_dir, bundle_type="per_cluster")
    X = pd.DataFrame(np.zeros((5, 5)), columns=[f"f_{i}" for i in range(5)])
    cids = pd.Series([-1, -1, -1, -1, -1])   # 全部陌生形态
    scores = bundle.predict(X, cluster_ids=cids)
    assert (scores == 0.0).all()


def test_save_training_data_parquet_or_csv(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="d", run_id="t6")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    arch.save_training_data(samples_with_cluster=df, features=df)
    shared = arch.run_dir / "shared"
    has_a = (shared / "samples_with_cluster.parquet").exists() or \
            (shared / "samples_with_cluster.csv").exists()
    assert has_a


def test_save_training_summary(tmp_path):
    arch = BundleArchive(base_dir=str(tmp_path), bundle_name="s", run_id="t7")
    arch.save_training_summary({"hello": "world", "n": 5})
    p = arch.run_dir / "training_summary.json"
    assert p.exists()
    manifest = arch.run_dir / "manifest.json"
    assert manifest.exists()
    import json
    with open(p) as f:
        d = json.load(f)
    assert d["hello"] == "world"
    with open(manifest) as f:
        m = json.load(f)
    assert m["schema_version"] == "artifact-manifest/v1"
    assert m["artifact_type"] == "training_bundle"
    assert m["strategy"] == "s"
    assert m["summary"]["hello"] == "world"
