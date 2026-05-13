"""Smoke test: anti-look-ahead + 风控验证"""
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
from src.utils.validator import LookAheadValidator


def _ok(msg): print(f"  [OK] {msg}")


def test_backtester_entry_delay_1():
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    signals = pd.DataFrame({"date": [dates[0]], "code": ["000001"], "score": [1.0]})
    rows = []
    for i, d in enumerate(dates):
        if i == 0:   o, c, h, l = 10.0, 10.0, 10.1, 9.9
        elif i == 1: o, c, h, l = 20.0, 20.0, 20.1, 19.9
        else:        o, c, h, l = 20.0+i*0.1, 20.0+i*0.1, 20.2+i*0.1, 19.9+i*0.1
        rows.append({"date": d, "code": "000001", "open": o, "high": h, "low": l, "close": c})
    daily = pd.DataFrame(rows)
    bt = Backtester(initial_cash=100000, topk=1, sell_n=3, slippage=0.0,
                    entry_delay=1, skip_zhangting_open=False)
    bt.run(signals, daily)
    buys = bt.get_trades()
    buys = buys[buys["action"] == "BUY"]
    assert len(buys) == 1
    assert buys.iloc[0]["date"] == "2023-01-03"
    assert abs(buys.iloc[0]["price"] - 20.0) < 1e-6
    _ok("entry_delay=1: T+1 open 撮合")


def test_backtester_zhangting_skip():
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    signals = pd.DataFrame({"date": [dates[0]], "code": ["000001"], "score": [1.0]})
    rows = []
    for i, d in enumerate(dates):
        if i == 0:   o, c, h, l = 10.0, 10.0, 10.1, 9.9
        elif i == 1: o, c, h, l = 11.0, 11.5, 11.5, 11.0
        else:        o, c, h, l = 11.5, 11.5, 11.6, 11.4
        rows.append({"date": d, "code": "000001", "open": o, "high": h, "low": l, "close": c})
    daily = pd.DataFrame(rows)
    bt = Backtester(initial_cash=100000, topk=1, sell_n=3, slippage=0.0,
                    entry_delay=1, skip_zhangting_open=True)
    bt.run(signals, daily)
    buys = bt.get_trades()
    buys = buys[buys["action"] == "BUY"] if len(buys) else buys
    assert len(buys) == 0
    _ok("skip_zhangting_open: 一字高开被跳过")


def test_backtester_stop_loss():
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    signals = pd.DataFrame({"date": [dates[0]], "code": ["000001"], "score": [1.0]})
    rows = []
    for i, d in enumerate(dates):
        if i == 0:   o, c, h, l = 10.0, 10.0, 10.1, 9.9
        elif i == 1: o, c, h, l = 10.0, 10.0, 10.0, 10.0
        elif i == 2: o, c, h, l = 9.9, 9.7, 9.95, 9.5
        else:        o, c, h, l = 9.5, 9.5, 9.5, 9.5
        rows.append({"date": d, "code": "000001", "open": o, "high": h, "low": l, "close": c})
    daily = pd.DataFrame(rows)
    bt = Backtester(initial_cash=100000, topk=1, sell_n=10, slippage=0.0,
                    entry_delay=1, stop_loss=0.03, skip_zhangting_open=False)
    bt.run(signals, daily)
    sells = bt.get_trades()
    sells = sells[sells["action"].str.startswith("SELL")]
    assert len(sells) == 1
    assert sells.iloc[0]["action"] == "SELL_STOP_LOSS"
    assert abs(sells.iloc[0]["price"] - 9.7) < 1e-6
    _ok("stop_loss: -3% 触发硬止损")


def test_backtester_min_score():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    signals = pd.DataFrame({"date": [dates[0]], "code": ["000001"], "score": [0.5]})
    daily = pd.DataFrame([
        {"date": d, "code": "000001", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0}
        for d in dates
    ])
    bt = Backtester(initial_cash=100000, topk=1, entry_delay=1,
                    min_score=0.7, skip_zhangting_open=False)
    bt.run(signals, daily)
    assert len(bt.get_trades()) == 0
    _ok("min_score: 低分信号被过滤")


def test_backtester_entry_delay_0_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Backtester(entry_delay=0)
        assert any("entry_delay=0" in str(wi.message) for wi in w)
    _ok("entry_delay=0 触发未来函数警告")


def test_random_split_with_date_col_raises():
    df = pd.DataFrame({
        "code": ["A", "B", "C", "D"] * 5,
        "first_date": pd.date_range("2023-01-01", periods=20, freq="D"),
    })
    sp = SectorStockSplitter(test_ratio=0.25, valid_ratio=0.25, random_seed=0)
    try:
        sp.split_by_stock(df, stock_col="code", date_col="first_date")
    except ValueError as e:
        assert "look-ahead" in str(e)
        _ok("按股票随机切分 + date_col -> raise look-ahead")
        return
    raise AssertionError("expected ValueError")


def test_split_by_time_safe():
    df = pd.DataFrame({
        "code": ["A", "B"] * 50,
        "first_date": pd.date_range("2023-01-01", periods=100, freq="D"),
    })
    sp = SectorStockSplitter(test_ratio=0.2, valid_ratio=0.2)
    s = sp.split_by_time(df, date_col="first_date")
    assert s["train"]["first_date"].max() < s["valid"]["first_date"].min()
    assert s["valid"]["first_date"].max() < s["test"]["first_date"].min()
    _ok("split_by_time 严格时间序")


def test_split_by_time_boundary_aligned():
    rows = []
    for d in pd.date_range("2023-01-01", periods=50, freq="D"):
        for c in range(5):
            rows.append({"code": f"S{c:03d}", "first_date": d})
    df = pd.DataFrame(rows)
    sp = SectorStockSplitter(test_ratio=0.2, valid_ratio=0.2)
    s = sp.split_by_time(df, date_col="first_date")
    assert s["train"]["first_date"].max() < s["valid"]["first_date"].min()
    assert s["valid"]["first_date"].max() < s["test"]["first_date"].min()
    LookAheadValidator.assert_no_time_leakage(s, date_col="first_date")
    _ok("split_by_time 同日样本日期对齐")


def test_purged_kfold():
    df = pd.DataFrame({
        "first_date": pd.date_range("2023-01-01", periods=200, freq="D"),
    })
    sp = SectorStockSplitter()
    folds = list(sp.purged_kfold_by_time(df, "first_date", n_splits=5, embargo_days=7))
    assert len(folds) == 5
    for f in folds:
        if f["train"].empty or f["test"].empty: continue
        td = pd.to_datetime(f["train"]["first_date"])
        sd = pd.to_datetime(f["test"]["first_date"])
        for t in td:
            assert (sd - t).dt.days.abs().min() > 7
    _ok("purged_kfold embargo 生效")


def test_custom_dataset_split_by_index_warns():
    feats = pd.DataFrame({
        "sample_id": list(range(20)),
        "first_date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "f1": np.arange(20, dtype=float),
    })
    labels = pd.DataFrame({"sample_id": list(range(20)),
                            "label_combined": [0,1,2]*6+[0,1]})
    ds = CustomDataset(feats, labels, label_col="label_combined")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.split_by_index(test_ratio=0.2, valid_ratio=0.2, shuffle=True)
        assert any("未来函数" in str(x.message) for x in w)
    _ok("split_by_index(shuffle=True) 警告")


def test_look_ahead_validator_apis():
    splits = {
        "train": pd.DataFrame({"first_date": pd.to_datetime(["2023-06-01"])}),
        "test":  pd.DataFrame({"first_date": pd.to_datetime(["2023-01-01"])}),
    }
    try:
        LookAheadValidator.assert_no_time_leakage(splits, "first_date")
        raise AssertionError("should have raised")
    except Exception as e:
        assert "look-ahead leakage" in str(e)
    _ok("assert_no_time_leakage 检测出泄漏")

    try:
        LookAheadValidator.scan_feature_names(["f_ret_5", "future_return"], raise_error=True)
        raise AssertionError("should have raised")
    except Exception as e:
        assert "future_return" in str(e)
    _ok("scan_feature_names 命中 future_return")


def main():
    print("running anti-look-ahead smoke tests ...")
    test_backtester_entry_delay_1()
    test_backtester_zhangting_skip()
    test_backtester_stop_loss()
    test_backtester_min_score()
    test_backtester_entry_delay_0_warns()
    test_random_split_with_date_col_raises()
    test_split_by_time_safe()
    test_split_by_time_boundary_aligned()
    test_purged_kfold()
    test_custom_dataset_split_by_index_warns()
    test_look_ahead_validator_apis()
    print("\nALL PASS - 训练管线已杜绝未来函数 + 风控生效")


if __name__ == "__main__":
    main()
