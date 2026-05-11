"""M-002 模型评估器

计算分类指标 (accuracy / precision / recall / f1) 与 IC / IR 等回归型指标。
对多分类问题，AUC 使用 OvR + macro 平均。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """模型评估器。"""

    # ------------------------------------------------------------------
    @staticmethod
    def calc_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Pearson IC，要求两边都不是常数。"""
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        if y_pred.size == 0 or y_pred.std() == 0 or y_true.std() == 0:
            return float("nan")
        return float(np.corrcoef(y_pred, y_true)[0, 1])

    @staticmethod
    def calc_rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred_rank = pd.Series(y_pred).rank()
        y_true_rank = pd.Series(y_true).rank()
        return ModelEvaluator.calc_ic(y_pred_rank.values, y_true_rank.values)

    @staticmethod
    def calc_ir(ic_series: pd.Series) -> float:
        s = pd.Series(ic_series).dropna()
        if s.empty or s.std() == 0:
            return float("nan")
        return float(s.mean() / s.std())

    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true,
        y_pred,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if len(y_true_arr) == 0:
            return {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
                "ic": float("nan"),
                "confusion_matrix": np.zeros((0, 0), dtype=int),
            }

        labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
        average = "binary" if len(labels) == 2 else "macro"

        out: Dict[str, object] = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels),
            "labels": labels,
        }

        # AUC（需要概率）
        out["auc"] = self._safe_auc(y_true_arr, y_proba, labels)

        # IC（把预测作为打分使用，目标为标签）
        try:
            out["ic"] = self.calc_ic(y_pred_arr.astype(float), y_true_arr.astype(float))
        except Exception:  # noqa: BLE001
            out["ic"] = float("nan")

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_proba, labels) -> float:
        if y_proba is None:
            return float("nan")
        try:
            if len(labels) == 2:
                proba = np.asarray(y_proba)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    proba = proba[:, 1]
                return float(roc_auc_score(y_true, proba))
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels))
        except (ValueError, IndexError):
            return float("nan")

    # ------------------------------------------------------------------
    def format_summary(self, metrics: Dict[str, object]) -> str:
        lines = ["=== model evaluation ==="]
        for k in ("accuracy", "precision", "recall", "f1", "auc", "ic"):
            v = metrics.get(k)
            if isinstance(v, float):
                lines.append(f"  {k:>9}: {v:.4f}")
        return "\n".join(lines)
