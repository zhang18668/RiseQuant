"""Smoke test: 验证 anti-look-ahead 修复无回归.

直接 python 执行 (绕开 pytest collection):
    PYTHONPATH=. python tests/_smoke_no_lookahead.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings

import numpy as np
import pandas as pd

from src.backtest.backtester import Backtester
from src.dataset.custom_dataset import CustomDataset
from src.dataset.sector_split import SectorStockSplitter
from src.utils.validator import LookAheadValidator, ValidationError


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


# ----------------------------------------------------------------------
# 1. Backtester: 信号日 T 的撮合必须发生在 T+entry_delay
# ----------------------------------------------------------------------
def test_backtester_entry_delay_1():
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    signals = pd.DataFrame({
        "date": [dates[0]],
        "code": ["000001"],
        "score": [1.0],
    })
    rows = []
    for i, d in enumerate(dates):
        if i == 0:
            o, c = 10.0, 10.0
        elif i == 1:
            o, c = 20.0, 20.0
        else:
            o, c = 20.0 + i * 0.1, 20.0 + i * 0.1
        rows.append({"date": d, "code": "000001", "open": o, "close": c})
    daily = pd.DataFrame(rows)

    bt = Backtester(initial_cash=100000, topk=1, sell_n=3, slippage=0.0, entry_delay=1)
    bt.run(signals, daily)
    trades = bt.get_trades()
    buys = trades[trades["action"] == "BUY"]
    assert len(buys) == 1, f"expected 1 buy, got {len(buys)}"
    assert buys.iloc[0]["date"] == "2023-01-03", f"buy must happen on T+1, got {buys.iloc[0]['date']}"
    assert abs(buys.iloc[0]["price"] - 20.0) < 1e-6, f"buy price must use T+1 open=20, got {buys.iloc[0]['price']}"
    _ok("entry_delay=1: 信号在 T+1 开盘成交 (T+1 open=20.0)")


def test_backtester_entry_delay_0_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Backtester(entry_delay=0)
        assert any("entry_delay=0" in str(wi.message) for wi in w), \
            "entry_delay=0 should emit a future-leak warning"
    _ok("entry_delay=0 触发未来函数警告")


# ----------------------------------------------------------------------
# 2. SectorStockSplitter: 时间断言
# ----------------------------------------------------------------------
def test_random_split_with_date_col_raises():
    df = pd.DataFrame({
        "code": ["A", "B", "C", "D"] * 5,
        "first_date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "x": np.arange(20),
    })
    splitter = SectorStockSplitter(test_ratio=0.25, valid_ratio=0.25, random_seed=0)
    try:
        splitter.split_by_stock(df, stock_col="code", date_col="first_date")
    except ValueError as e:
        assert "look-ahead" in str(e), str(e)
        _ok("按股票随机切分时, 启用 date_col 会 raise look-ahead leakage")
        return
    raise AssertionError("expected ValueError due to time leakage")


def test_split_by_time_safe():
    df = pd.DataFrame({
        "code": ["A", "B"] * 50,
        "first_date": pd.date_range("2023-01-01", periods=100, freq="D"),
    })
    splitter = SectorStockSplitter(test_ratio=0.2, valid_ratio=0.2, random_seed=0)
    splits = splitter.split_by_time(df, date_col="first_date")
    assert splits["train"]["first_date"].max() < splits["valid"]["first_date"].min()
    assert splits["valid"]["first_date"].max() < splits["test"]["first_date"].min()
    _ok("split_by_time: train.max < valid.min < test.min")


def test_split_by_time_boundary_aligned():
    """同一天多样本时, 切点必须对齐到日期边界, 不允许 train.max == valid.min."""
    dates = []
    codes = []
    for d in pd.date_range("2023-01-01", periods=50, freq="D"):
        for c in range(5):
            dates.append(d)
            codes.append(f"S{c:03d}")
    df = pd.DataFrame({"code": codes, "first_date": dates})
    splitter = SectorStockSplitter(test_ratio=0.2, valid_ratio=0.2, random_seed=0)
    splits = splitter.split_by_time(df, date_col="first_date")
    train_max = splits["train"]["first_date"].max()
    valid_min = splits["valid"]["first_date"].min()
    valid_max = splits["valid"]["first_date"].max()
    test_min = splits["test"]["first_date"].min()
    assert train_max < valid_min, f"train.max ({train_max}) must be < valid.min ({valid_min})"
    assert valid_max < test_min, f"valid.max ({valid_max}) must be < test.min ({test_min})"
    LookAheadValidator.assert_no_time_leakage(splits, date_col="first_date")
    _ok("split_by_time: 同一天多样本时切点对齐到日期边界")


def test_purged_kfold():
    df = pd.DataFrame({
        "first_date": pd.date_range("2023-01-01", periods=200, freq="D"),
        "x": np.arange(200),
    })
    splitter = SectorStockSplitter(random_seed=0)
    folds = list(splitter.purged_kfold_by_time(
        df, date_col="first_date", n_splits=5, embargo_days=7
    ))
    assert len(folds) == 5
    for f in folds:
        if f["train"].empty or f["test"].empty:
            continue
        train_dates = pd.to_datetime(f["train"]["first_date"])
        test_dates = pd.to_datetime(f["test"]["first_date"])
        for td in train_dates:
            gap_days = (test_dates - td).dt.days.abs().min()
            assert gap_days > 7, f"embargo violated: fold={f['fold']} gap={gap_days}"
    _ok("purged_kfold_by_time: 任一训练样本距 test 边界 > embargo_days")


# ----------------------------------------------------------------------
# 3. CustomDataset.split_by_index
# ----------------------------------------------------------------------
def test_custom_dataset_split_by_index_warns():
    features = pd.DataFrame({
        "sample_id": list(range(20)),
        "first_date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "f1": np.arange(20, dtype=float),
        "f2": np.arange(20, dtype=float) * 0.5,
    })
    labels = pd.DataFrame({
        "sample_id": list(range(20)),
        "label_combined": [0, 1, 2] * 6 + [0, 1],
    })
    ds = CustomDataset(features, labels, label_col="label_combined")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.split_by_index(test_ratio=0.2, valid_ratio=0.2, shuffle=True)
        assert any("未来函数" in str(wi.message) for wi in w)
    _ok("split_by_index(shuffle=True) 触发未来函数警告")

    segs = ds.split_by_index(test_ratio=0.2, valid_ratio=0.2, shuffle=False,
                              date_col="first_date")
    assert segs.train["first_date"].max() < segs.valid["first_date"].min()
    assert segs.valid["first_date"].max() < segs.test["first_date"].min()
    _ok("split_by_index(shuffle=False) 严格时间序")


# ----------------------------------------------------------------------
# 4. LookAheadValidator: 直接 API
# ----------------------------------------------------------------------
def test_look_ahead_validator_apis():
    # 4a) 时间泄漏检测
    splits = {
        "train": pd.DataFrame({"first_date": pd.to_datetime(["2023-06-01"])}),
        "test":  pd.DataFrame({"first_date": pd.to_datetime(["2023-01-01"])}),
    }
    try:
        LookAheadValidator.assert_no_time_leakage(splits, "first_date")
    except Exception as e:
        assert "look-ahead leakage" in str(e)
        _ok("assert_no_time_leakage 检测出 train.max > test.min")

    # 4b) 特征命名扫描
    try:
        LookAheadValidator.scan_feature_names(
            ["f_ret_5", "future_return", "f_ma_20"], raise_error=True
        )
    except Exception as e:
        assert "future_return" in str(e)
        _ok("scan_feature_names 命中 'future_return'")

    # 4c) features 时间 > signal 时间 → raise
    feats = pd.DataFrame({
        "code": ["A", "A"],
        "event_date": pd.to_datetime(["2023-06-10", "2023-07-10"]),
    })
    sig_dates = pd.to_datetime(["2023-06-05", "2023-07-15"])
    try:
        LookAheadValidator.assert_features_before_signal(
            feats, "event_date", sig_dates
        )
    except Exception as e:
        assert "look-ahead in features" in str(e)
        _ok("assert_features_before_signal 检测出 feature_date > signal_date")


# ----------------------------------------------------------------------
def main():
    print("running anti-look-ahead smoke tests ...")
    test_backtester_entry_delay_1()
    test_backtester_entry_delay_0_warns()
    test_random_split_with_date_col_raises()
    test_split_by_time_safe()
    test_split_by_time_boundary_aligned()
    test_purged_kfold()
    test_custom_dataset_split_by_index_warns()
    test_look_ahead_validator_apis()
    print("\nALL PASS - 训练管线已杜绝未来函数")


if __name__ == "__main__":
    main()
okAheadValidator.assert_features_before_signal(
            feats, "event_date", sig_dates
        )
    except Exception as e:
        assert "look-ahead in features" in str(e)
        _ok("assert_features_before_signal 检测出 feature_date > signal_date")


# ----------------------------------------------------------------------
def main():
    print("running anti-look-ahead smoke tests ...")
    test_backtester_entry_delay_1()
    test_backtester_entry_delay_0_warns()
    test_random_split_with_date_col_raises()
    test_split_by_time_safe()
    test_split_by_time_boundary_aligned()
    test_purged_kfold()
    test_custom_dataset_split_by_index_warns()
    test_look_ahead_validator_apis()
    print("\nALL PASS - look-ahead bias eliminated")


if __name__ == "__main__":
    main()
main()
