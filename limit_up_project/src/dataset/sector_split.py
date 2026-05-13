"""DS-003 板块/股票切分器

提供四种切分策略：
- :meth:`split_by_time`                — *时间序*, 推荐, 杜绝未来函数;
- :meth:`split_by_sector`              — 按板块随机, 必须配合 ``date_col`` 做时间断言;
- :meth:`split_by_stock`               — 按股票随机, 必须配合 ``date_col`` 做时间断言;
- :meth:`split_by_sector_and_stock`    — 同上;
- :meth:`split_by_stock_then_time`     — *推荐*, 先按时间切, 再在每段内分股票.

以及两种交叉验证：
- :meth:`cross_validate_by_sector`
- :meth:`cross_validate_by_stock`
- :meth:`purged_kfold_by_time`         — *推荐*, 基于时间的 Purged K-Fold,
  在 train/test 之间留 ``embargo_days`` 天断带, 防止 label-window 泄漏.

杜绝未来函数说明
------------------
按 stock/sector 的随机切分本身 *不一定* 直接泄漏未来, 但若不附加时间约束,
训练集会出现 2024-Q4 的样本, 测试集出现 2024-Q1 的样本 — 模型相当于
"用未来训练、回到过去预测". 本模块为所有按 key 的切分方法加入 ``date_col``
可选参数, 一旦传入即对 train.max(date) < test.min(date) 做严格断言;
未传入时仅发出 ``warnings.warn`` 提醒.
"""

from __future__ import annotations

import warnings
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
    def _assert_no_time_leakage(
        self,
        splits: dict,
        date_col: Optional[str],
    ) -> None:
        """断言切分后训练集时间不晚于测试集 (杜绝未来函数)."""
        if date_col is None:
            warnings.warn(
                "未来函数风险: 按 key 随机切分时未提供 date_col, 训练/测试集可能"
                "存在时间交叉. 建议传入 date_col 启用时间断言, 或改用 "
                "split_by_time / split_by_stock_then_time.",
                stacklevel=3,
            )
            return
        train = splits.get("train")
        valid = splits.get("valid")
        test = splits.get("test")
        if train is None or train.empty or date_col not in train.columns:
            return
        train_max = pd.to_datetime(train[date_col]).max()
        for name, seg in (("valid", valid), ("test", test)):
            if seg is None or seg.empty or date_col not in seg.columns:
                continue
            seg_min = pd.to_datetime(seg[date_col]).min()
            if pd.notna(seg_min) and pd.notna(train_max) and seg_min <= train_max:
                raise ValueError(
                    f"detected look-ahead leakage: train.{date_col}.max()="
                    f"{train_max} >= {name}.{date_col}.min()={seg_min}. "
                    "请改用按时间切分, 或在每个时间段内再按 key 切分."
                )

    def _split_by_keys(
        self,
        df: pd.DataFrame,
        keys: pd.Series,
        date_col: Optional[str] = None,
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
        splits = {
            "train": df[keys.isin(train_keys)].reset_index(drop=True),
            "valid": df[keys.isin(valid_keys)].reset_index(drop=True),
            "test": df[keys.isin(test_keys)].reset_index(drop=True),
        }
        self._assert_no_time_leakage(splits, date_col)
        return splits

    # ------------------------------------------------------------------
    def split_by_time(
        self,
        df: pd.DataFrame,
        date_col: str = "event_date",
    ) -> dict:
        """按时间顺序切分 — 切点对齐到 *日期边界* , 严格保证 train.max < valid.min < test.min.

        同一天的样本会被整体放到同一段, 避免:
          - 训练集与验证集在同一天并存 (panel data leakage)
          - 由于横切边界导致的 train.max == valid.min, 触发未来函数断言
        """
        if date_col not in df.columns:
            raise ValueError(f"date column not found: {date_col}")
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return {
                "train": df_sorted, "valid": df_sorted.iloc[0:0], "test": df_sorted.iloc[0:0]
            }
        n_test = int(n * self.test_ratio)
        n_valid = int(n * self.valid_ratio)
        n_train = max(n - n_test - n_valid, 0)

        dates_sorted = pd.to_datetime(df_sorted[date_col]).values

        # 把切点对齐到日期边界:
        # train_end_idx 之前的所有样本 (含同一天的全部) 归入 train
        def _align_right(cut: int) -> int:
            """把 cut 向右扩展, 包含与 dates_sorted[cut-1] 相同的所有同期样本."""
            if cut <= 0 or cut >= n:
                return cut
            boundary_date = dates_sorted[cut - 1]
            # searchsorted side='right' 返回 boundary_date 之后第一个不等的位置
            return int(np.searchsorted(dates_sorted, boundary_date, side="right"))

        train_end = _align_right(n_train)
        valid_end = _align_right(train_end + n_valid) if n_valid else train_end

        return {
            "train": df_sorted.iloc[:train_end].reset_index(drop=True),
            "valid": df_sorted.iloc[train_end:valid_end].reset_index(drop=True),
            "test": df_sorted.iloc[valid_end:].reset_index(drop=True),
        }

    # ------------------------------------------------------------------
    def split_by_date_ranges(
        self,
        df: pd.DataFrame,
        date_col: str = "first_date",
        train_end: Optional[object] = None,
        valid_end: Optional[object] = None,
        test_end: Optional[object] = None,
    ) -> dict:
        """按显式日期边界切分 (杜绝未来函数).

        切分规则 (闭区间, 含等号):
            train:  date <= train_end
            valid:  train_end <  date <= valid_end
            test :  valid_end <  date <= test_end  (test_end=None 表示不限上界)

        Parameters
        ----------
        df : DataFrame
        date_col : str
            日期列名, 例如 "first_date" / "event_date".
        train_end, valid_end, test_end : str / pd.Timestamp / None
            分段右边界. ``valid_end=None`` 时不构造 valid (train 直接邻接 test).
        """
        if date_col not in df.columns:
            raise ValueError(f"date column not found: {date_col}")
        if train_end is None:
            raise ValueError("train_end is required for split_by_date_ranges")

        d = pd.to_datetime(df[date_col])
        train_end_ts = pd.Timestamp(train_end)
        valid_end_ts = pd.Timestamp(valid_end) if valid_end is not None else None
        test_end_ts  = pd.Timestamp(test_end)  if test_end  is not None else None

        train_mask = d <= train_end_ts
        if valid_end_ts is not None:
            valid_mask = (d > train_end_ts) & (d <= valid_end_ts)
            base_mask  = (d > valid_end_ts)
        else:
            valid_mask = pd.Series(False, index=df.index)
            base_mask  = (d > train_end_ts)
        test_mask = base_mask & (d <= test_end_ts) if test_end_ts is not None else base_mask

        splits = {
            "train": df.loc[train_mask].reset_index(drop=True),
            "valid": df.loc[valid_mask].reset_index(drop=True),
            "test":  df.loc[test_mask].reset_index(drop=True),
        }
        self._assert_no_time_leakage(splits, date_col)
        return splits

    def split_by_sector(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        date_col: Optional[str] = None,
    ) -> dict:
        if sector_col not in df.columns:
            raise ValueError(f"sector column not found: {sector_col}")
        return self._split_by_keys(df, df[sector_col], date_col=date_col)

    def split_by_stock(
        self,
        df: pd.DataFrame,
        stock_col: str = "code",
        date_col: Optional[str] = None,
    ) -> dict:
        if stock_col not in df.columns:
            raise ValueError(f"stock column not found: {stock_col}")
        return self._split_by_keys(df, df[stock_col], date_col=date_col)

    def split_by_sector_and_stock(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        stock_col: str = "code",
        date_col: Optional[str] = None,
    ) -> dict:
        if sector_col not in df.columns or stock_col not in df.columns:
            raise ValueError(f"missing column: {sector_col}/{stock_col}")
        composite = df[sector_col].astype(str) + "::" + df[stock_col].astype(str)
        return self._split_by_keys(df, composite, date_col=date_col)

    # ------------------------------------------------------------------
    def split_by_stock_then_time(
        self,
        df: pd.DataFrame,
        date_col: str = "first_date",
        stock_col: str = "code",
    ) -> dict:
        """先按时间切, 再在每段内做 stock 切分, 杜绝未来函数.

        步骤:
        1. 按 ``date_col`` 升序排序后, 按 ``test_ratio``/``valid_ratio`` 切成
           train / valid / test 三段;
        2. 不做随机洗牌, 直接保留时间段内的全部样本.

        相对于 :meth:`split_by_time` 的差异: 在每段内可进一步按股票做去重
        或抽样, 但本实现保持简单, 与 ``split_by_time`` 等价 — 给出更明确
        的命名以提示调用者 "时间安全 + 股票可扩展".
        """
        if date_col not in df.columns:
            raise ValueError(f"date column not found: {date_col}")
        if stock_col not in df.columns:
            raise ValueError(f"stock column not found: {stock_col}")
        # 直接复用 split_by_time 的日期边界对齐逻辑, 杜绝同期跨段
        splits = self.split_by_time(df, date_col=date_col)
        self._assert_no_time_leakage(splits, date_col)
        return splits

    # ------------------------------------------------------------------
    def _cross_validate(
        self,
        df: pd.DataFrame,
        keys: pd.Series,
        n_splits: int,
        date_col: Optional[str] = None,
    ) -> Iterator[dict]:
        unique = sorted(keys.dropna().unique().tolist())
        rng = np.random.default_rng(self.random_seed)
        unique = list(unique)
        rng.shuffle(unique)
        folds = np.array_split(unique, n_splits)
        for i in range(n_splits):
            test_keys = set(folds[i].tolist()) if len(folds[i]) else set()
            test_df = df[keys.isin(test_keys)].reset_index(drop=True)
            train_df = df[~keys.isin(test_keys)].reset_index(drop=True)
            self._assert_no_time_leakage(
                {"train": train_df, "test": test_df}, date_col=date_col
            )
            yield {"fold": i, "train": train_df, "test": test_df}

    def cross_validate_by_sector(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        n_splits: int = 5,
        date_col: Optional[str] = None,
    ) -> Iterator[dict]:
        if sector_col not in df.columns:
            raise ValueError(f"sector column not found: {sector_col}")
        return self._cross_validate(df, df[sector_col], n_splits, date_col=date_col)

    def cross_validate_by_stock(
        self,
        df: pd.DataFrame,
        stock_col: str = "code",
        n_splits: int = 5,
        date_col: Optional[str] = None,
    ) -> Iterator[dict]:
        if stock_col not in df.columns:
            raise ValueError(f"stock column not found: {stock_col}")
        return self._cross_validate(df, df[stock_col], n_splits, date_col=date_col)

    # ------------------------------------------------------------------
    def purged_kfold_by_time(
        self,
        df: pd.DataFrame,
        date_col: str = "first_date",
        n_splits: int = 5,
        embargo_days: int = 10,
    ) -> Iterator[dict]:
        """Time-aware Purged K-Fold (no look-ahead)."""
        if date_col not in df.columns:
            raise ValueError(f"date column not found: {date_col}")
        if int(n_splits) < 2:
            raise ValueError("n_splits must be >= 2")
        if int(embargo_days) < 0:
            raise ValueError("embargo_days must be >= 0")

        df_sorted = df.assign(_dt=pd.to_datetime(df[date_col])).sort_values("_dt")
        df_sorted = df_sorted.reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return
        fold_sizes = np.array([n // n_splits] * n_splits)
        fold_sizes[: n % n_splits] += 1
        starts = np.concatenate(([0], np.cumsum(fold_sizes)))
        embargo = pd.Timedelta(days=int(embargo_days))

        for i in range(n_splits):
            test_start, test_end = starts[i], starts[i + 1]
            test_df = df_sorted.iloc[test_start:test_end].copy()
            if test_df.empty:
                continue
            t_min = test_df["_dt"].min()
            t_max = test_df["_dt"].max()
            mask = (df_sorted["_dt"] < t_min - embargo) | (df_sorted["_dt"] > t_max + embargo)
            train_df = df_sorted.loc[mask].copy()
            train_df = train_df.drop(columns=["_dt"]).reset_index(drop=True)
            test_df = test_df.drop(columns=["_dt"]).reset_index(drop=True)
            yield {"fold": i, "train": train_df, "test": test_df}
