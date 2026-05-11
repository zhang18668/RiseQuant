"""L-002 LabelValidator 测试。"""
from __future__ import annotations

import pandas as pd

from src.label.label_validator import LabelValidator


def test_validator_happy():
    df = pd.DataFrame({
        "sample_id": ["a", "b"],
        "label_short": [0, 1],
        "label_combined": [0, 2],
    })
    res = LabelValidator().validate(df)
    assert res.is_valid and res.errors == []


def test_validator_inconsistent():
    df = pd.DataFrame({
        "sample_id": ["a"],
        "label_short": [0],
        "label_combined": [2],  # 不一致
    })
    res = LabelValidator().validate(df)
    assert not res.is_valid


def test_validator_invalid_values():
    df = pd.DataFrame({
        "sample_id": ["a"],
        "label_short": [5],
        "label_combined": [3],
    })
    res = LabelValidator().validate(df)
    assert not res.is_valid


def test_validator_duplicate_id():
    df = pd.DataFrame({
        "sample_id": ["a", "a"],
        "label_short": [0, 0],
        "label_combined": [0, 0],
    })
    res = LabelValidator().validate(df)
    assert not res.is_valid


def test_validator_missing_column():
    df = pd.DataFrame({"sample_id": ["a"]})
    res = LabelValidator().validate(df)
    assert not res.is_valid
