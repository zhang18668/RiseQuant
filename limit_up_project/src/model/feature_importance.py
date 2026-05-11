"""M-003 特征重要性分析

从训练好的 :class:`LimitUpModelTrainer` 中提取特征重要性并做基础排序/可视化数据。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.model.model_trainer import LimitUpModelTrainer


class FeatureImportanceAnalyzer:
    """特征重要性分析器。"""

    def __init__(self, trainer: LimitUpModelTrainer) -> None:
        self.trainer = trainer

    def importance_df(self, top_n: Optional[int] = None) -> pd.DataFrame:
        s = self.trainer.feature_importance()
        df = s.reset_index()
        df.columns = ["feature", "importance"]
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        if top_n is not None:
            df = df.head(top_n)
        return df

    def top_features(self, top_n: int = 10) -> list:
        return self.importance_df(top_n=top_n)["feature"].tolist()
