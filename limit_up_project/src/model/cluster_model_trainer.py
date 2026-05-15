"""M-005 ClusterModelTrainer (方案 A: per_cluster) — Pattern-Cluster v2

对每个 cluster 单独训练一个 LightGBM 二分类:
- 输入: 已打了 ``cluster_id`` 的 samples + 全量特征 DataFrame + 时序切分配置
- 输出: List[per-cluster artifact] (含 trainer、metrics、importance、usable 标签)

usable 判定规则 (与文档 §8 对齐):
- ``n_samples_train >= min_train_samples`` (默认 80)
- ``pos_ratio_train >= 0.05 and <= 0.30``
- ``auc_test >= 0.55`` (放宽阈值, 文档 0.65 是期望; 0.55 是必须)
- 其余指标 (calibration / signal_ratio) 由上层在记录时附加, 不强制
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataset.sector_split import SectorStockSplitter
from src.model.model_evaluator import ModelEvaluator
from src.model.model_trainer import LimitUpModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterArtifact:
    """单个 cluster 训练后的产物."""

    cluster_id: int
    trainer: Optional[LimitUpModelTrainer]
    metrics: Dict[str, Any]
    feature_importance: Optional[pd.DataFrame]
    n_train: int
    n_valid: int
    n_test: int
    pos_ratio_train: float
    usable: bool
    reason_if_not_usable: str = ""

    def to_summary_row(self) -> dict:
        row = {
            "cluster_id": int(self.cluster_id),
            "n_train": int(self.n_train),
            "n_valid": int(self.n_valid),
            "n_test": int(self.n_test),
            "pos_ratio_train": float(self.pos_ratio_train),
            "usable": bool(self.usable),
            "reason": self.reason_if_not_usable,
        }
        for k, v in (self.metrics or {}).items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[f"m_{k}"] = float(v)
        return row


@dataclass
class ClusterModelTrainer:
    """对一组带 cluster_id 的样本, 按 cluster 分桶训练多个 LightGBM."""

    min_train_samples: int = 80
    min_pos_ratio: float = 0.03
    max_pos_ratio: float = 0.50
    min_auc_test: float = 0.55
    model_params: Dict[str, Any] = field(default_factory=dict)
    test_ratio: float = 0.2
    valid_ratio: float = 0.1
    feature_cols: Optional[List[str]] = None
    date_col: str = "potential_date"
    cluster_col: str = "cluster_id"
    label_col: str = "label_pre"
    weight_col: Optional[str] = "sample_weight"

    # ------------------------------------------------------------------
    def train(
        self,
        samples_with_cluster: pd.DataFrame,
        features_df: pd.DataFrame,
        train_end: Optional[Any] = None,
        valid_end: Optional[Any] = None,
        test_end: Optional[Any] = None,
        split_method: str = "date",
    ) -> Dict[int, ClusterArtifact]:
        """按 cluster_id 分桶训练.

        ``samples_with_cluster`` 必须含: sample_id, cluster_id, label_pre, potential_date,
        (可选) sample_weight.
        ``features_df`` 含 sample_id + 全部 feature_cols.

        Returns
        -------
        dict {cluster_id: ClusterArtifact}.
        """
        if self.feature_cols is None:
            self.feature_cols = [c for c in features_df.columns
                                 if c.startswith("f_w_") or c.startswith("f_mkt_")]

        # join 一次
        sample_id_col = "sample_id"
        if sample_id_col not in features_df.columns:
            raise ValueError("features_df 缺少 sample_id 列")
        keep_cols = [sample_id_col] + list(self.feature_cols)
        feat = features_df[keep_cols].drop_duplicates(sample_id_col)

        need_cols = [sample_id_col, self.cluster_col, self.label_col, self.date_col]
        for c in need_cols:
            if c not in samples_with_cluster.columns:
                raise ValueError(f"samples_with_cluster 缺少 {c} 列")
        if self.weight_col and self.weight_col in samples_with_cluster.columns:
            need_cols = need_cols + [self.weight_col]

        df = samples_with_cluster[need_cols].merge(feat, on=sample_id_col, how="inner")

        out: Dict[int, ClusterArtifact] = {}
        cluster_ids = sorted([int(c) for c in df[self.cluster_col].unique() if int(c) >= 0])
        logger.info(f"ClusterModelTrainer: {len(cluster_ids)} 个 cluster 待训练")

        for cid in cluster_ids:
            sub = df[df[self.cluster_col] == cid].copy()
            logger.info(f"--- cluster {cid}: {len(sub)} 样本 ---")
            artifact = self._train_one(sub, cid, train_end, valid_end, test_end, split_method)
            out[cid] = artifact

        return out

    # ------------------------------------------------------------------
    def _train_one(
        self,
        sub: pd.DataFrame,
        cluster_id: int,
        train_end,
        valid_end,
        test_end,
        split_method: str,
    ) -> ClusterArtifact:
        if len(sub) < 10:
            return ClusterArtifact(
                cluster_id=cluster_id, trainer=None, metrics={},
                feature_importance=None, n_train=0, n_valid=0, n_test=0,
                pos_ratio_train=0.0, usable=False,
                reason_if_not_usable=f"样本数 {len(sub)} 过少 (<10)",
            )

        splitter = SectorStockSplitter(test_ratio=self.test_ratio, valid_ratio=self.valid_ratio)
        if split_method == "date" and (train_end is not None or valid_end is not None):
            splits = splitter.split_by_date_ranges(
                sub, date_col=self.date_col,
                train_end=train_end, valid_end=valid_end, test_end=test_end,
            )
        else:
            splits = splitter.split_by_time(sub, date_col=self.date_col)

        X_train = splits["train"][self.feature_cols].fillna(0)
        y_train = splits["train"][self.label_col]
        X_valid = splits["valid"][self.feature_cols].fillna(0)
        y_valid = splits["valid"][self.label_col]
        X_test  = splits["test"][self.feature_cols].fillna(0)
        y_test  = splits["test"][self.label_col]

        n_train, n_valid, n_test = len(X_train), len(X_valid), len(X_test)
        pos_ratio_train = float(y_train.mean()) if len(y_train) else 0.0

        # usable check 1: 样本数
        if n_train < self.min_train_samples:
            return ClusterArtifact(
                cluster_id=cluster_id, trainer=None, metrics={},
                feature_importance=None,
                n_train=n_train, n_valid=n_valid, n_test=n_test,
                pos_ratio_train=pos_ratio_train,
                usable=False,
                reason_if_not_usable=f"训练样本数 {n_train} < {self.min_train_samples}",
            )
        # usable check 2: 正样本占比
        if pos_ratio_train < self.min_pos_ratio or pos_ratio_train > self.max_pos_ratio:
            return ClusterArtifact(
                cluster_id=cluster_id, trainer=None, metrics={},
                feature_importance=None,
                n_train=n_train, n_valid=n_valid, n_test=n_test,
                pos_ratio_train=pos_ratio_train,
                usable=False,
                reason_if_not_usable=f"pos_ratio={pos_ratio_train:.3f} 超出 [{self.min_pos_ratio},{self.max_pos_ratio}]",
            )
        # usable check 3: 单类别 (y_train 全 0 或全 1)
        if y_train.nunique() < 2:
            return ClusterArtifact(
                cluster_id=cluster_id, trainer=None, metrics={},
                feature_importance=None,
                n_train=n_train, n_valid=n_valid, n_test=n_test,
                pos_ratio_train=pos_ratio_train,
                usable=False, reason_if_not_usable="y_train 只有一个类别",
            )

        trainer = LimitUpModelTrainer({"params": self.model_params})
        w_train = splits["train"][self.weight_col] if (self.weight_col and self.weight_col in splits["train"].columns) else None
        w_valid = splits["valid"][self.weight_col] if (self.weight_col and self.weight_col in splits["valid"].columns) else None

        trainer.train(
            X_train, y_train, X_valid, y_valid,
            feature_names=list(self.feature_cols),
            sample_weight=w_train, eval_sample_weight=w_valid,
        )

        # 评估
        metrics: Dict[str, Any] = {}
        if n_test > 0 and y_test.nunique() >= 2:
            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test)
            ev = ModelEvaluator()
            metrics = dict(ev.evaluate(y_test.values, y_pred, y_proba))
            # 补充: precision@top10%
            p_pos = y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba[:, 0]
            order = np.argsort(-p_pos)
            top_k = max(1, len(order) // 10)
            metrics["precision_top10pct"] = float(y_test.values[order[:top_k]].mean())
            # 校准: Brier
            metrics["brier_score"] = float(np.mean((p_pos - y_test.values) ** 2))
            # usable_signal_ratio (score > 0.6 的比例)
            metrics["signal_ratio_at_06"] = float(np.mean(p_pos >= 0.6))

        auc_test = float(metrics.get("auc", 0.0))
        usable = auc_test >= self.min_auc_test
        reason = "" if usable else f"auc_test={auc_test:.3f} < {self.min_auc_test}"

        imp = trainer.get_feature_importance() if trainer is not None else None

        return ClusterArtifact(
            cluster_id=cluster_id, trainer=trainer, metrics=metrics,
            feature_importance=imp,
            n_train=n_train, n_valid=n_valid, n_test=n_test,
            pos_ratio_train=pos_ratio_train,
            usable=usable, reason_if_not_usable=reason,
        )
