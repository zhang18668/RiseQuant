"""M-001 模型训练器 (LightGBM 多分类)

封装一个最小可用的训练器：
- ``train(X_train, y_train, X_valid=None, y_valid=None)``
- ``predict(X)``  -> 类别 (np.ndarray[int])
- ``predict_proba(X)`` -> 概率矩阵 (np.ndarray[float, shape=(n, n_classes)])
- ``save(path)`` / ``load(path)``

参数从 ``config["model"]["params"]`` 取，缺省提供保守的默认值。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

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
    """LightGBM 多分类训练器。"""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "multiclass",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        if not _HAS_LGB:
            raise ImportError("lightgbm is required for LimitUpModelTrainer")
        cfg = config or {}
        params = dict(self.DEFAULT_PARAMS)
        if isinstance(cfg, dict):
            params.update(cfg.get("model", {}).get("params", {}))
        self.params = params
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: Optional[list] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> "LimitUpModelTrainer":
        if len(X_train) == 0:
            raise ValueError("empty training data")
        n_classes = int(pd.Series(y_train).nunique())
        params = dict(self.params)
        params["num_class"] = max(n_classes, 2) if params.get("objective") == "multiclass" else None
        params = {k: v for k, v in params.items() if v is not None}

        # 使用 sklearn API 更简洁
        model_params = {k: v for k, v in params.items() if k not in ("objective", "num_class")}
        if n_classes <= 2:
            model_params["objective"] = "binary"
        else:
            model_params["objective"] = "multiclass"
            model_params["num_class"] = n_classes

        self.model = lgb.LGBMClassifier(**model_params)
        eval_set = [(X_valid, y_valid)] if X_valid is not None and y_valid is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
        )
        self.feature_names_ = list(X_train.columns)
        self.classes_ = self.model.classes_
        logger.info(f"trained LGBM with {len(X_train)} samples, {len(self.feature_names_)} features")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
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
        self._check_fitted()
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
            name="importance",
        )
        return importances.sort_values(ascending=False)

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("model has not been trained yet")
