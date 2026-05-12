"""模型训练层 (M-001 ~ M-003).

子模块以懒加载方式暴露, 避免在缺失 sklearn / lightgbm 等可选依赖时
就连 ``import src.model`` 都会失败.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["LimitUpModelTrainer", "ModelEvaluator", "FeatureImportanceAnalyzer"]


def __getattr__(name: str):  # pragma: no cover - 简单转发
    if name == "LimitUpModelTrainer":
        from src.model.model_trainer import LimitUpModelTrainer

        return LimitUpModelTrainer
    if name == "ModelEvaluator":
        from src.model.model_evaluator import ModelEvaluator

        return ModelEvaluator
    if name == "FeatureImportanceAnalyzer":
        from src.model.feature_importance import FeatureImportanceAnalyzer

        return FeatureImportanceAnalyzer
    raise AttributeError(f"module 'src.model' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from src.model.feature_importance import FeatureImportanceAnalyzer
    from src.model.model_evaluator import ModelEvaluator
    from src.model.model_trainer import LimitUpModelTrainer
