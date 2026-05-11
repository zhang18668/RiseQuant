"""模型训练器测试"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.model.model_trainer import LimitUpModelTrainer


class TestLimitUpModelTrainer:
    """模型训练器测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.trainer = LimitUpModelTrainer()

    def test_train_produces_model(self):
        """训练产生模型"""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.choice([0, 1, 2], size=100))

        model = self.trainer.train(X_train, y_train)
        assert model is not None

    def test_predict(self):
        """预测"""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.choice([0, 1, 2], size=100))

        self.trainer.train(X_train, y_train)

        X_test = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
        pred = self.trainer.predict(X_test)

        assert len(pred) == 10
        assert all(p in [0, 1, 2] for p in pred)

    def test_predict_proba(self):
        """预测概率"""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.choice([0, 1, 2], size=100))

        self.trainer.train(X_train, y_train)

        X_test = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
        proba = self.trainer.predict_proba(X_test)

        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_default_config(self):
        """默认配置"""
        trainer = LimitUpModelTrainer()
        assert trainer.config is not None
        assert trainer.config.get("objective") == "multiclass"
