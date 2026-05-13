"""DS-002 自定义数据集

将特征宽表 + 标签表合并，并把列拆成 ``feature_cols`` 与 ``label_col``，
方便直接喂给 sklearn / lightgbm。

支持两种切分方式：
- 按时间切分: ``split_by_time``;
- 按样本切分: ``split_by_index`` (默认随机)。

后续的"按板块"/"按股票"切分由 :class:`SectorStockSplitter` 提供。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetSegments:
    """简单的数据切分容器。"""

    train: pd.DataFrame
    valid: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None

    def __getitem__(self, key: str) -> pd.DataFrame:
        seg = getattr(self, key, None)
        if seg is None:
            raise KeyError(key)
        return seg


class CustomDataset:
    """特征+标签数据集。"""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        feature_cols: Optional[Iterable[str]] = None,
        label_col: str = "label_combined",
        key_cols: Iterable[str] = ("sample_id",),
    ) -> None:
        self.key_cols: List[str] = list(key_cols)
        self.label_col = label_col

        df = features.merge(labels, on=list(key_cols), how="inner", suffixes=("", "_lbl"))
        if label_col not in df.columns:
            raise ValueError(f"missing label column after merge: {label_col}")

        if feature_cols is None:
            exclude = set(self.key_cols) | {label_col, "label_short", "code", "event_date",
                                              "first_date", "second_date", "main_wave_date",
                                              "is_main_wave", "gap_days", "period_return"}
            feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]
        self.feature_cols: List[str] = list(feature_cols)
        if not self.feature_cols:
            raise ValueError("no feature columns selected")

        self.df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.feature_cols]

    @property
    def y(self) -> pd.Series:
        return self.df[self.label_col]

    # ------------------------------------------------------------------
    def split_by_time(
        self,
        date_col: str,
        train_end,
        valid_end=None,
    ) -> DatasetSegments:
        """按 ``date_col`` 切分: (-inf, train_end], (train_end, valid_end], (valid_end, +inf)。"""
        if date_col not in self.df.columns:
            raise ValueError(f"date column not found: {date_col}")
        dates = pd.to_datetime(self.df[date_col])
        train_end = pd.Timestamp(train_end)
        valid_end_ts = pd.Timestamp(valid_end) if valid_end else None

        train_mask = dates <= train_end
        if valid_end_ts is not None:
            valid_mask = (dates > train_end) & (dates <= valid_end_ts)
            test_mask = dates > valid_end_ts
        else:
            valid_mask = pd.Series(False, index=self.df.index)
            test_mask = dates > train_end

        return DatasetSegments(
            train=self.df.loc[train_mask].reset_index(drop=True),
            valid=self.df.loc[valid_mask].reset_index(drop=True) if valid_mask.any() else None,
            test=self.df.loc[test_mask].reset_index(drop=True) if test_mask.any() else None,
        )

    def split_by_index(
        self,
        test_ratio: float = 0.2,
        valid_ratio: float = 0.0,
        random_seed: int = 42,
        shuffle: bool = True,
        date_col: Optional[str] = None,
    ) -> DatasetSegments:
        """按样本下标切分.

        ``shuffle=True`` (默认) 会随机洗牌, 在时间序列任务中会引入未来函数,
        仅在 ``date_col is None`` 且确认 IID 时可用. 建议:
        - 时间序列任务: ``shuffle=False``, 或直接使用 :meth:`split_by_time`;
        - 必须随机时: 传入 ``date_col`` 以触发训练/测试时间断言.
        """
        n = len(self.df)
        if n == 0:
            return DatasetSegments(train=self.df, valid=None, test=None)

        if shuffle:
            warnings.warn(
                "split_by_index(shuffle=True) 会打乱时间序, 在时间序列/量化任务"
                "中会引入未来函数. 请使用 split_by_time 或传入 shuffle=False.",
                stacklevel=2,
            )
            rng = np.random.default_rng(random_seed)
            idx = rng.permutation(n)
        else:
            idx = np.arange(n)
        n_test = int(n * test_ratio)
        n_valid = int(n * valid_ratio)
        # 按时间序: 训练集在前, 验证集中, 测试集在后 (shuffle=False 时)
        if shuffle:
            test_idx = idx[:n_test]
            valid_idx = idx[n_test : n_test + n_valid]
            train_idx = idx[n_test + n_valid :]
        else:
            n_train = n - n_test - n_valid
            train_idx = idx[:n_train]
            valid_idx = idx[n_train : n_train + n_valid]
            test_idx = idx[n_train + n_valid :]

        segments = DatasetSegments(
            train=self.df.iloc[train_idx].reset_index(drop=True),
            valid=self.df.iloc[valid_idx].reset_index(drop=True) if n_valid else None,
            test=self.df.iloc[test_idx].reset_index(drop=True) if n_test else None,
        )

        # 时间断言: 任一段非空时, train.max < valid.min < test.min
        if date_col is not None and date_col in self.df.columns:
            train_max = pd.to_datetime(segments.train[date_col]).max() if not segments.train.empty else None
            for name, seg in (("valid", segments.valid), ("test", segments.test)):
                if seg is None or seg.empty:
                    continue
                seg_min = pd.to_datetime(seg[date_col]).min()
                if train_max is not None and pd.notna(seg_min) and pd.notna(train_max) and seg_min <= train_max:
                    raise ValueError(
                        f"detected look-ahead leakage: train.{date_col}.max()={train_max}"
                        f" >= {name}.{date_col}.min()={seg_min}. 请使用 split_by_time."
                    )
        return segments

    # ------------------------------------------------------------------
    def get_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return df[self.feature_cols], df[self.label_col]
