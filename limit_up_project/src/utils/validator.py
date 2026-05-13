"""U-004 数据质量校验工具

提供一组 *无副作用* 的静态方法，可以单独使用，也可以通过
:meth:`DataValidator.validate_daily_data` 进行成批校验。

新增 :class:`LookAheadValidator`: 集中检测训练管道中的未来函数 (look-ahead
bias), 包括: 时间切分泄漏、特征-标签时间错位、特征列名嫌疑词扫描.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


# ----------------------------------------------------------------------
# Anti look-ahead bias validator
# ----------------------------------------------------------------------
class LookAheadValidator:
    """杜绝未来函数 (look-ahead bias) 的统一校验工具.

    用法
    ----
    在 pipeline 训练前/回测前调用, 任一项失败则 raise ``ValidationError``::

        LookAheadValidator.assert_no_time_leakage(splits, date_col="first_date")
        LookAheadValidator.assert_features_before_signal(features, "event_date",
            signal_dates, code_col="code")
        LookAheadValidator.scan_feature_names(features.columns)
    """

    #: 特征命名中疑似含有"未来"语义的关键词, 命中时 raise.
    FORBIDDEN_NAME_TOKENS: Tuple[str, ...] = (
        "future", "next_", "lead_", "fwd_", "ahead", "tomorrow", "_t+", "post_",
    )

    # ------------------------------------------------------------------
    @staticmethod
    def assert_no_time_leakage(
        splits: Mapping[str, pd.DataFrame],
        date_col: str,
    ) -> None:
        """断言 train 的最大日期 < valid/test 的最小日期."""
        train = splits.get("train")
        if train is None or train.empty or date_col not in train.columns:
            return
        train_max = pd.to_datetime(train[date_col]).max()
        for name in ("valid", "test"):
            seg = splits.get(name)
            if seg is None or len(seg) == 0 or date_col not in seg.columns:
                continue
            seg_min = pd.to_datetime(seg[date_col]).min()
            if pd.notna(seg_min) and pd.notna(train_max) and seg_min <= train_max:
                raise ValidationError(
                    f"look-ahead leakage: train.{date_col}.max()={train_max} >= "
                    f"{name}.{date_col}.min()={seg_min}"
                )

    # ------------------------------------------------------------------
    @staticmethod
    def assert_features_before_signal(
        features: pd.DataFrame,
        feature_date_col: str,
        signal_dates: pd.Series,
        code_col: str = "code",
    ) -> None:
        """断言每个样本的特征日期 <= 对应的信号日 (即不含未来数据).

        ``signal_dates`` 与 ``features`` 行对齐 (同一 index 或同一 code+date).
        """
        if feature_date_col not in features.columns:
            raise ValidationError(f"feature_date_col not in features: {feature_date_col}")
        fdates = pd.to_datetime(features[feature_date_col]).reset_index(drop=True)
        sdates = pd.to_datetime(pd.Series(signal_dates).reset_index(drop=True))
        if len(fdates) != len(sdates):
            raise ValidationError(
                f"length mismatch: features={len(fdates)}, signal_dates={len(sdates)}"
            )
        violation = fdates > sdates
        if violation.any():
            n = int(violation.sum())
            first_bad = features.loc[violation, [code_col, feature_date_col]].head(3)
            raise ValidationError(
                f"look-ahead in features: {n} rows have {feature_date_col} > signal_date. "
                f"first violations:\n{first_bad}"
            )

    # ------------------------------------------------------------------
    @classmethod
    def scan_feature_names(
        cls,
        names: Iterable[str],
        forbidden: Optional[Sequence[str]] = None,
        raise_error: bool = True,
    ) -> List[str]:
        """扫描特征列名, 找出疑似 *未来语义* 的关键词."""
        tokens = tuple(forbidden) if forbidden is not None else cls.FORBIDDEN_NAME_TOKENS
        suspects = sorted({
            str(n) for n in names
            if any(tok in str(n).lower() for tok in tokens)
        })
        if suspects:
            msg = f"suspicious feature names (look-ahead?): {suspects}"
            if raise_error:
                raise ValidationError(msg)
            warnings.warn(msg, stacklevel=2)
        return suspects

    # ------------------------------------------------------------------
    @staticmethod
    def assert_label_after_feature(
        feature_dates: pd.Series,
        label_dates: pd.Series,
        min_gap_days: int = 0,
    ) -> None:
        """断言标签发生时间 > 特征日期 + min_gap_days.

        在涨停二板项目里, label 基于 (second_date 之后 N 日的最高价), 必然
        在 first_date (特征日) 之后, 此校验主要防呆: 防止有人不小心把
        label 提前到 first_date 之前.
        """
        fd = pd.to_datetime(pd.Series(feature_dates).reset_index(drop=True))
        ld = pd.to_datetime(pd.Series(label_dates).reset_index(drop=True))
        if len(fd) != len(ld):
            raise ValidationError("feature_dates and label_dates length mismatch")
        gap = (ld - fd).dt.days
        bad = gap < int(min_gap_days)
        if bad.any():
            raise ValidationError(
                f"label timing leak: {int(bad.sum())} rows have label_date - "
                f"feature_date < {min_gap_days} days"
            )
