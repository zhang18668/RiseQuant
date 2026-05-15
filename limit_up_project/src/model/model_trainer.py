"""M-001 LightGBM trainer (binary/multiclass).

API:
- train(X, y, X_val, y_val, feature_names, sample_weight,
        eval_sample_weight, categorical_feature)
- predict / predict_proba / save / load
- feature_importance / get_feature_importance

Pattern-Cluster v2: added categorical_feature for plan B (cluster_id).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    lgb = None
    _HAS_LGB = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LimitUpModelTrainer:
    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "multiclass",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42,
        "is_unbalance": True,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        params = dict(self.DEFAULT_PARAMS)
        if isinstance(cfg, dict):
            if "params" in cfg:
                params.update(cfg.get("params", {}))
            params.update(cfg.get("model", {}).get("params", {}))
        self.params: Dict[str, Any] = params
        self.config: Dict[str, Any] = params
        self.model = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self._categorical_feature: Optional[List[str]] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[pd.Series] = None,
        eval_sample_weight: Optional[pd.Series] = None,
        categorical_feature: Optional[List[str]] = None,
    ):
        if not _HAS_LGB:
            raise ImportError("lightgbm required")
        if len(X_train) == 0:
            raise ValueError("empty training data")

        X_train = self._coerce_numeric(X_train, drop_warning=True)
        if X_valid is not None:
            X_valid = self._coerce_numeric(X_valid, drop_warning=False)

        n_classes = int(pd.Series(y_train).nunique())
        mp = {k: v for k, v in self.params.items() if k not in ("objective", "num_class")}
        if n_classes <= 2:
            mp["objective"] = "binary"
            if "is_unbalance" not in mp and "scale_pos_weight" not in mp:
                mp["is_unbalance"] = True
        else:
            mp["objective"] = "multiclass"
            mp["num_class"] = n_classes
            mp.pop("is_unbalance", None)
            mp.pop("scale_pos_weight", None)

        self.model = lgb.LGBMClassifier(**mp)
        eval_set = ([(X_valid, y_valid)]
                    if X_valid is not None and y_valid is not None else None)
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None and eval_sample_weight is not None:
            fit_kwargs["eval_sample_weight"] = [eval_sample_weight]
        if categorical_feature:
            cats = [c for c in categorical_feature if c in X_train.columns]
            if cats:
                fit_kwargs["categorical_feature"] = cats
                for c in cats:
                    X_train[c] = X_train[c].astype("int64")
                    if X_valid is not None and c in X_valid.columns:
                        X_valid[c] = X_valid[c].astype("int64")
                self._categorical_feature = list(cats)
        self.model.fit(X_train, y_train, eval_set=eval_set, **fit_kwargs)
        self.feature_names_ = (list(feature_names) if feature_names is not None
                                else list(X_train.columns))
        self.classes_ = self.model.classes_
        logger.info(f"trained LGBM: {len(X_train)} samples, {len(self.feature_names_)} feats")
        return self.model

    @staticmethod
    def _coerce_numeric(X: pd.DataFrame, drop_warning: bool = False) -> pd.DataFrame:
        keep = ["number", "bool", "category"]
        try:
            cols = X.select_dtypes(include=keep).columns
        except Exception:
            cols = X.select_dtypes(include=["number", "bool"]).columns
        dropped = [c for c in X.columns if c not in cols]
        if dropped and drop_warning:
            logger.warning(f"dropping non-numeric cols: {dropped}")
        return X[cols].copy()

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

    def save(self, path: str) -> None:
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names_": self.feature_names_,
                "classes_": self.classes_,
                "params": self.params,
                "categorical_feature": self._categorical_feature,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LimitUpModelTrainer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        t = cls(config={"model": {"params": data.get("params", {})}})
        t.model = data["model"]
        t.feature_names_ = data.get("feature_names_")
        t.classes_ = data.get("classes_")
        t._categorical_feature = data.get("categorical_feature")
        return t

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        s = pd.Series(self.model.feature_importances_,
                      index=self.feature_names_, name="importance")
        return s.sort_values(ascending=False)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        s = self.feature_importance()
        df = s.reset_index()
        df.columns = ["feature", "importance"]
        if top_n is not None:
            df = df.head(int(top_n))
        return df.reset_index(drop=True)

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("model not trained yet")
