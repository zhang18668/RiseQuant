"""DS-002 CustomDataset 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.dataset.custom_dataset import CustomDataset


def _make_dataset():
    features = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(10)],
        "code": ["000001"] * 10,
        "event_date": pd.bdate_range("2023-01-02", periods=10),
        "f_x1": list(range(10)),
        "f_x2": [i * 0.5 for i in range(10)],
    })
    labels = pd.DataFrame({
        "sample_id": features["sample_id"],
        "label_short": [0, 1] * 5,
        "label_combined": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    })
    return CustomDataset(features, labels)


def test_xy_extraction():
    ds = _make_dataset()
    assert set(ds.feature_cols) == {"f_x1", "f_x2"}
    assert ds.label_col == "label_combined"
    assert len(ds.X) == 10
    assert len(ds.y) == 10


def test_split_by_index_no_overlap():
    ds = _make_dataset()
    segs = ds.split_by_index(test_ratio=0.3, valid_ratio=0.2, random_seed=0)
    n = len(ds.df)
    assert len(segs.test) == int(n * 0.3)
    assert len(segs.valid) == int(n * 0.2)
    # train + valid + test = n
    assert len(segs.train) + len(segs.valid) + len(segs.test) == n


def test_split_by_time():
    ds = _make_dataset()
    train_end = ds.df["event_date"].iloc[4]
    segs = ds.split_by_time("event_date", train_end=train_end)
    assert (segs.train["event_date"] <= train_end).all()
    assert segs.test is not None
    assert (segs.test["event_date"] > train_end).all()


def test_split_by_time_with_valid():
    ds = _make_dataset()
    train_end = ds.df["event_date"].iloc[3]
    valid_end = ds.df["event_date"].iloc[6]
    segs = ds.split_by_time("event_date", train_end=train_end, valid_end=valid_end)
    assert segs.valid is not None
    assert (segs.valid["event_date"] > train_end).all()
    assert (segs.valid["event_date"] <= valid_end).all()


def test_get_xy():
    ds = _make_dataset()
    X, y = ds.get_xy(ds.df)
    assert list(X.columns) == ds.feature_cols
    assert (y == ds.df[ds.label_col]).all()


def test_missing_label_raises():
    features = pd.DataFrame({"sample_id": ["a"], "f_x": [1.0]})
    labels = pd.DataFrame({"sample_id": ["a"], "label_short": [0]})
    with pytest.raises(ValueError):
        CustomDataset(features, labels, label_col="missing")
