"""M-006 SingleWithCFTrainer (方案 B: 单模型 + cluster_id 类别特征)

把 cluster_id 作为 LightGBM 的 categorical_feature 喂给单一模型,
让 LightGBM 内部按 cluster_id 做条件分裂.

与 ClusterModelTrainer 的区别
---------------------------
- 方案 A 训 N 个模型, 样本被分桶, 互不交叉
- 方案 B 训 1 个模型, 全部样本喂进去, 多了一列 ``cluster_id`` 作为类别特征

cluster_id == -1 的样本可选: 默认丢弃 (drop_unknown=True).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.dataset.sector_split import SectorStockSplitter
from src.model.model_evaluator import ModelEvaluator
from src.model.model_trainer import LimitUpModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SingleWithCFArtifact:
    """方案 B 训练产物."""

    trainer: Optional[LimitUpModelTrainer]
    metrics: Dict[str, Any]
    feature_importance: Optional[pd.DataFrame]
    metrics_per_cluster: Dict[int, Dict[str, Any]]   # 在 test 集上按 cluster 拆分
    n_train: int
    n_valid: int
    n_test: int

    def to_summary(self) -> dict:
        out = {
            "n_train": int(self.n_train),
            "n_valid": int(self.n_valid),
            "n_test": int(self.n_test),
            "metrics_overall": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                                 for k, v in (self.metrics or {}).items()},
            "metrics_per_cluster": {
                int(k): {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                          for kk, vv in (m or {}).items()}
                for k, m in (self.metrics_per_cluster or {}).items()
            },
        }
        return out


@dataclass
class SingleWithCFTrainer:
    """单模型 + cluster_id 类别特征."""

    model_params: Dict[str, Any] = field(default_factory=dict)
    test_ratio: float = 0.2
    valid_ratio: float = 0.1
    feature_cols: Optional[List[str]] = None
    date_col: str = "potential_date"
    cluster_col: str = "cluster_id"
    label_col: str = "label_pre"
    weight_col: Optional[str] = "sample_weight"
    drop_unknown: bool = True    # 丢掉 cluster_id == -1 的样本

    # ------------------------------------------------------------------
    def train(
        self,
        samples_with_cluster: pd.DataFrame,
        features_df: pd.DataFrame,
        train_end: Optional[Any] = None,
        valid_end: Optional[Any] = None,
        test_end: Optional[Any] = None,
        split_method: str = "date",
    ) -> SingleWithCFArtifact:
        if self.feature_cols is None:
            self.feature_cols = [c for c in features_df.columns
                                 if c.startswith("f_w_") or c.startswith("f_mkt_")]
        # 加上 cluster_id 作为特征
        cf_col = self.cluster_col
        feature_cols_full = list(self.feature_cols) + [cf_col]

        sample_id_col = "sample_id"
        if sample_id_col not in features_df.columns:
            raise ValueError("features_df 缺少 sample_id 列")
        feat = features_df[[sample_id_col] + list(self.feature_cols)].drop_duplicates(sample_id_col)
        need = [sample_id_col, cf_col, self.label_col, self.date_col]
        if self.weight_col and self.weight_col in samples_with_cluster.columns:
            need = need + [self.weight_col]
        df = samples_with_cluster[need].merge(feat, on=sample_id_col, how="inner")

        if self.drop_unknown:
            n_before = len(df)
            df = df[df[cf_col] >= 0].copy()
            logger.info(f"丢弃 cluster_id=-1 样本: {n_before - len(df)} 条")

        if len(df) == 0:
            return SingleWithCFArtifact(
                trainer=None, metrics={}, feature_importance=None,
                metrics_per_cluster={}, n_train=0, n_valid=0, n_test=0,
            )

        splitter = SectorStockSplitter(test_ratio=self.test_ratio, valid_ratio=self.valid_ratio)
        if split_method == "date" and (train_end is not None or valid_end is not None):
            splits = splitter.split_by_date_ranges(
                df, date_col=self.date_col,
                train_end=train_end, valid_end=valid_end, test_end=test_end,
            )
        else:
            splits = splitter.split_by_time(df, date_col=self.date_col)

        # X 含 cluster_id 列, 后面 fit 时通过 categorical_feature=[cluster_id] 标记
        X_train = splits["train"][feature_cols_full].fillna(0)
        y_train = splits["train"][self.label_col]
        X_valid = splits["valid"][feature_cols_full].fillna(0)
        y_valid = splits["valid"][self.label_col]
        X_test  = splits["test"][feature_cols_full].fillna(0)
        y_test  = splits["test"][self.label_col]

        w_train = splits["train"][self.weight_col] if (self.weight_col and self.weight_col in splits["train"].columns) else None
        w_valid = splits["valid"][self.weight_col] if (self.weight_col and self.weight_col in splits["valid"].columns) else None

        trainer = LimitUpModelTrainer({"params": self.model_params})
        trainer.train(
            X_train, y_train, X_valid, y_valid,
            feature_names=feature_cols_full,
            sample_weight=w_train, eval_sample_weight=w_valid,
            categorical_feature=[cf_col],
        )

        # 总体 metrics
        metrics: Dict[str, Any] = {}
        metrics_per_cluster: Dict[int, Dict[str, Any]] = {}
        if len(X_test) > 0 and y_test.nunique() >= 2:
            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test)
            ev = ModelEvaluator()
            metrics = dict(ev.evaluate(y_test.values, y_pred, y_proba))
            p_pos = y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba[:, 0]
            order = np.argsort(-p_pos)
            top_k = max(1, len(order) // 10)
            metrics["precision_top10pct"] = float(y_test.values[order[:top_k]].mean())
            metrics["brier_score"] = float(np.mean((p_pos - y_test.values) ** 2))
            metrics["signal_ratio_at_06"] = float(np.mean(p_pos >= 0.6))

            # per-cluster 拆分
            test_with_cid = splits["test"].copy()
            test_with_cid["_y_true"] = y_test.values
            test_with_cid["_y_proba"] = p_pos
            test_with_cid["_y_pred"] = y_pred
            for cid, g in test_with_cid.groupby(cf_col):
                if len(g) < 5 or g["_y_true"].nunique() < 2:
                    metrics_per_cluster[int(cid)] = {"n": int(len(g)), "skipped": True}
                    continue
                m = ev.evaluate(g["_y_true"].values, g["_y_pred"].values,
                                np.column_stack([1 - g["_y_proba"].values, g["_y_proba"].values]))
                m["n"] = int(len(g))
                metrics_per_cluster[int(cid)] = dict(m)

        imp = trainer.get_feature_importance()
        return SingleWithCFArtifact(
            trainer=trainer, metrics=metrics, feature_importance=imp,
            metrics_per_cluster=metrics_per_cluster,
            n_train=len(X_train), n_valid=len(X_valid), n_test=len(X_test),
        )
