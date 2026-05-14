"""M-001 模型训练器 (LightGBM 多分类)

封装一个最小可用的训练器:
- ``train(X_train, y_train, X_valid=None, y_valid=None, feature_names=None)``
- ``predict(X)``  -> 类别 (np.ndarray[int])
- ``predict_proba(X)`` -> 概率矩阵 (np.ndarray[float, shape=(n, n_classes)])
- ``save(path)`` / ``load(path)``
- ``feature_importance()`` / ``get_feature_importance()`` -> pd.DataFrame
- ``self.config`` -> 训练参数字典 (含 ``objective`` 等键)

参数从 ``config["model"]["params"]`` 取 (允许只传 ``params`` 字典),
缺省提供保守的默认值.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - 导入异常分支
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # noqa: BLE001
    lgb = None  # type: ignore
    _HAS_LGB = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LimitUpModelTrainer:
    """LightGBM 多分类训练器."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "multiclass",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42,
        # 类不平衡: 二分类时让 LGBM 自动按 (n_neg/n_pos) 加权
        # 多分类下 LGBM 会忽略这个参数, 安全
        "is_unbalance": True,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        params = dict(self.DEFAULT_PARAMS)
        if isinstance(cfg, dict):
            # 同时兼容 ``{"params": {...}}`` 与 ``{"model": {"params": {...}}}``
            if "params" in cfg:
                params.update(cfg.get("params", {}))
            params.update(cfg.get("model", {}).get("params", {}))
        # 提供 ``self.config`` 别名, 方便外部读取
        self.params: Dict[str, Any] = params
        self.config: Dict[str, Any] = params
        self.model = None  # type: ignore
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[pd.Series] = None,
        eval_sample_weight: Optional[pd.Series] = None,
    ):
        if not _HAS_LGB:
            raise ImportError("lightgbm is required for LimitUpModelTrainer.train()")

        if len(X_train) == 0:
            raise ValueError("empty training data")

        # 防御: 自动剔除非数值列 (如 datetime / object)
        X_train = self._coerce_numeric(X_train, drop_warning=True)
        if X_valid is not None:
            X_valid = self._coerce_numeric(X_valid, drop_warning=False)

        y_series = pd.Series(y_train)
        n_classes = int(y_series.nunique())

        model_params = {
            k: v for k, v in self.params.items() if k not in ("objective", "num_class")
        }
        if n_classes <= 2:
            model_params["objective"] = "binary"
            # 二分类时, 让 is_unbalance / scale_pos_weight 生效
            if "is_unbalance" not in model_params and "scale_pos_weight" not in model_params:
                model_params["is_unbalance"] = True
        else:
            model_params["objective"] = "multiclass"
            model_params["num_class"] = n_classes
            # 多分类下 LightGBM 不识别 is_unbalance, 移除避免警告
            model_params.pop("is_unbalance", None)
            model_params.pop("scale_pos_weight", None)

        self.model = lgb.LGBMClassifier(**model_params)
        eval_set = (
            [(X_valid, y_valid)]
            if X_valid is not None and y_valid is not None
            else None
        )
        # B3: sample_weight 支持
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None and eval_sample_weight is not None:
            fit_kwargs["eval_sample_weight"] = [eval_sample_weight]
        self.model.fit(X_train, y_train, eval_set=eval_set, **fit_kwargs)
        self.feature_names_ = (
            list(feature_names) if feature_names is not None else list(X_train.columns)
        )
        self.classes_ = self.model.classes_
        logger.info(
            f"trained LGBM with {len(X_train)} samples, "
            f"{len(self.feature_names_)} features"
        )
        return self.model

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_numeric(X: pd.DataFrame, drop_warning: bool = False) -> pd.DataFrame:
        """剔除非数值列 (datetime / object / category), 仅保留 LightGBM 能直接吃的列."""
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns
        dropped = [c for c in X.columns if c not in numeric_cols]
        if dropped and drop_warning:
            logger.warning(f"dropping non-numeric feature columns: {dropped}")
        return X[numeric_cols].copy()

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = self._coerce_numeric(X)
        if self.feature_names_:
            X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = self._coerce_numeric(X)
        if self.feature_names_:
            X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names_": self.feature_names_,
                "classes_": self.classes_,
                "params": self.params,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LimitUpModelTrainer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        trainer = cls(config={"model": {"params": data.get("params", {})}})
        trainer.model = data["model"]
        trainer.feature_names_ = data.get("feature_names_")
        trainer.classes_ = data.get("classes_")
        return trainer

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        """返回 (按重要性降序的) ``pd.Series``: index=feature_name."""
        self._check_fitted()
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
            name="importance",
        )
        return importances.sort_values(ascending=False)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """返回宽表 ``feature, importance`` (按重要性降序)."""
        s = self.feature_importance()
        df = s.reset_index()
        df.columns = ["feature", "importance"]
        if top_n is not None:
            df = df.head(int(top_n))
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("model has not been trained yet")
