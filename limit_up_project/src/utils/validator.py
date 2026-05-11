"""U-004 数据质量校验工具

提供一组 *无副作用* 的静态方法，可以单独使用，也可以通过
:meth:`DataValidator.validate_daily_data` 进行成批校验。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


class ValidationError(Exception):
    """数据校验错误。"""


class DataValidator:
    """A 股行情数据校验器。"""

    # ------------------------------------------------------------------
    # 基础检查
    # ------------------------------------------------------------------
    @staticmethod
    def check_required_columns(
        df: pd.DataFrame,
        required: Iterable[str],
        raise_error: bool = False,
    ) -> Tuple[bool, List[str]]:
        missing = [c for c in required if c not in df.columns]
        if missing and raise_error:
            raise ValidationError(f"missing required columns: {missing}")
        return len(missing) == 0, missing

    @staticmethod
    def check_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        raise_error: bool = False,
    ) -> Tuple[bool, int]:
        dup_count = int(df.duplicated(subset=subset).sum())
        if dup_count > 0 and raise_error:
            raise ValidationError(f"found {dup_count} duplicate rows")
        return dup_count > 0, dup_count

    @staticmethod
    def check_missing_ratio(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {c: 0.0 for c in df.columns}
        return {c: float(df[c].isna().mean()) for c in df.columns}

    @staticmethod
    def check_numeric_range(
        df: pd.DataFrame,
        col: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Tuple[bool, int]:
        if col not in df.columns:
            raise ValidationError(f"column not found: {col}")
        s = df[col].dropna()
        n_low = int((s < min_value).sum()) if min_value is not None else 0
        n_high = int((s > max_value).sum()) if max_value is not None else 0
        n_bad = n_low + n_high
        return n_bad == 0, n_bad

    @staticmethod
    def check_stock_codes(
        df: pd.DataFrame,
        code_col: str = "code",
        expected_length: int = 6,
    ) -> Tuple[bool, List[str]]:
        if code_col not in df.columns:
            raise ValidationError(f"column not found: {code_col}")
        codes = df[code_col].astype(str)
        invalid = sorted(codes[codes.str.len() != expected_length].unique().tolist())
        return len(invalid) == 0, invalid

    # ------------------------------------------------------------------
    # 组合校验：日线数据
    # ------------------------------------------------------------------
    REQUIRED_DAILY_COLS = (
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    @classmethod
    def validate_daily_data(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """校验日线数据。

        返回 ``{is_valid, errors, warnings, missing_ratio}``。
        """
        result: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "missing_ratio": {},
        }

        ok, missing = cls.check_required_columns(df, cls.REQUIRED_DAILY_COLS)
        if not ok:
            result["is_valid"] = False
            result["errors"].append(f"missing columns: {missing}")
            return result

        # 价格逻辑
        if (df["high"] < df["low"]).any():
            result["is_valid"] = False
            result["errors"].append("high < low in some rows")
        if (df["close"] > df["high"]).any():
            result["is_valid"] = False
            result["errors"].append("close > high in some rows")
        if (df["close"] < df["low"]).any():
            result["is_valid"] = False
            result["errors"].append("close < low in some rows")

        # 缺失率
        result["missing_ratio"] = cls.check_missing_ratio(df)
        high_missing = {c: r for c, r in result["missing_ratio"].items() if r > 0.1}
        if high_missing:
            result["warnings"].append(f"high missing ratio: {high_missing}")

        # 重复
        has_dup, n_dup = cls.check_duplicates(df, subset=["date", "code"])
        if has_dup:
            result["is_valid"] = False
            result["errors"].append(f"{n_dup} duplicate (date, code) rows")

        return result
