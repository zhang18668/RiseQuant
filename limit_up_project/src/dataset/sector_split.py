"""DS-003 板块/股票切分器

提供四种切分策略：
- :meth:`split_by_time`
- :meth:`split_by_sector`
- :meth:`split_by_stock`
- :meth:`split_by_sector_and_stock`

以及两种交叉验证：
- :meth:`cross_validate_by_sector`
- :meth:`cross_validate_by_stock`
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd


class SectorStockSplitter:
    """以股票/板块为粒度划分训练/验证/测试集。"""

    def __init__(
        self,
        test_ratio: float = 0.2,
        valid_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        if test_ratio < 0 or valid_ratio < 0 or test_ratio + valid_ratio >= 1:
            raise ValueError("invalid split ratios")
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    def _split_by_keys(
        self,
        df: pd.DataFrame,
        keys: pd.Series,
    ) -> dict:
        unique = keys.dropna().unique().tolist()
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(unique)
        n = len(unique)
        n_test = int(n * self.test_ratio)
        n_valid = int(n * self.valid_ratio)
        test_keys = set(unique[:n_test])
        valid_keys = set(unique[n_test : n_test + n_valid])
        train_keys = set(unique[n_test + n_valid :])
        return {
            "train": df[keys.isin(train_keys)].reset_index(drop=True),
            "valid": df[keys.isin(valid_keys)].reset_index(drop=True),
            "test": df[keys.isin(test_keys)].reset_index(drop=True),
        }

    # ------------------------------------------------------------------
    def split_by_time(
        self,
        df: pd.DataFrame,
        date_col: str = "event_date",
    ) -> dict:
        """按时间顺序切分。"""
        if date_col not in df.columns:
            raise ValueError(f"date column not found: {date_col}")
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_sorted)
        n_test = int(n * self.test_ratio)
        n_valid = int(n * self.valid_ratio)
        n_train = n - n_test - n_valid
        return {
            "train": df_sorted.iloc[:n_train].reset_index(drop=True),
            "valid": df_sorted.iloc[n_train : n_train + n_valid].reset_index(drop=True),
            "test": df_sorted.iloc[n_train + n_valid :].reset_index(drop=True),
        }

    def split_by_sector(self, df: pd.DataFrame, sector_col: str = "sector") -> dict:
        if sector_col not in df.columns:
            raise ValueError(f"sector column not found: {sector_col}")
        return self._split_by_keys(df, df[sector_col])

    def split_by_stock(self, df: pd.DataFrame, stock_col: str = "code") -> dict:
        if stock_col not in df.columns:
            raise ValueError(f"stock column not found: {stock_col}")
        return self._split_by_keys(df, df[stock_col])

    def split_by_sector_and_stock(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        stock_col: str = "code",
    ) -> dict:
        if sector_col not in df.columns or stock_col not in df.columns:
            raise ValueError(f"missing column: {sector_col}/{stock_col}")
        composite = df[sector_col].astype(str) + "::" + df[stock_col].astype(str)
        return self._split_by_keys(df, composite)

    # ------------------------------------------------------------------
    def _cross_validate(self, df: pd.DataFrame, keys: pd.Series, n_splits: int) -> Iterator[dict]:
        unique = sorted(keys.dropna().unique().tolist())
        rng = np.random.default_rng(self.random_seed)
        unique = list(unique)
        rng.shuffle(unique)
        folds = np.array_split(unique, n_splits)
        for i in range(n_splits):
            test_keys = set(folds[i].tolist()) if len(folds[i]) else set()
            test_df = df[keys.isin(test_keys)].reset_index(drop=True)
            train_df = df[~keys.isin(test_keys)].reset_index(drop=True)
            yield {"fold": i, "train": train_df, "test": test_df}

    def cross_validate_by_sector(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        n_splits: int = 5,
    ) -> Iterator[dict]:
        if sector_col not in df.columns:
            raise ValueError(f"sector column not found: {sector_col}")
        return self._cross_validate(df, df[sector_col], n_splits)

    def cross_validate_by_stock(
        self,
        df: pd.DataFrame,
        stock_col: str = "code",
        n_splits: int = 5,
    ) -> Iterator[dict]:
        if stock_col not in df.columns:
            raise ValueError(f"stock column not found: {stock_col}")
        return self._cross_validate(df, df[stock_col], n_splits)
