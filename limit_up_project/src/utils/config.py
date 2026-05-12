"""U-003 配置加载工具

放弃了原本基于单例的实现 —— 单例对测试不友好，并且使依赖关系不可见。
本模块改为：
- :class:`Config` 持有一个 ``dict``，提供 ``get``/``get_section``/``to_dict`` 等接口；
- :func:`load_config` 从 YAML 文件加载并返回 :class:`Config` 实例；
- :func:`default_config` 返回方案文档中规定的默认配置。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml


def default_config() -> dict:
    """返回方案文档中的默认配置。"""
    return {
        "data": {
            "tdx_dir": "",
            "cache_dir": "./data/cache",
            "start_date": "2018-01-01",
            "end_date": "2023-12-31",
        },
        "event": {
            "limit_up_threshold": 9.9,
            "exclude_st": True,
            "second_board_days": 5,
            "main_wave_days": 10,
            "main_wave_return": 0.15,
        },
        "feature": {
            "pre_trend_window_1m": 20,
            "pre_trend_window_2m": 40,
            "returns_n": [1, 3, 5, 10, 20],
            "volume_rolling_n": [5, 10],
            "ma_n": [5, 10, 20, 60],
            "volatility_n": [5, 10, 20],
        },
        "label": {
            "second_board_days": 5,
            "main_wave_days": 10,
            "main_wave_return": 0.15,
        },
        "dataset": {
            "test_ratio": 0.2,
            "valid_ratio": 0.1,
            "random_seed": 42,
        },
        "model": {
            "type": "lightgbm",
            "params": {
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "verbose": -1,
            },
        },
        "backtest": {
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "initial_cash": 10_000_000,
            "topk": 10,
            "sell_n": 5,
            "slippage": 0.003,
        },
    }


class Config:
    """轻量配置容器。"""

    _MISSING = object()

    def __init__(self, data: Optional[dict] = None) -> None:
        self._data = deepcopy(data) if data is not None else default_config()

    def get(self, key: str, default: Any = None) -> Any:
        """点号路径读取，例如 ``config.get("event.limit_up_threshold")``。"""
        value: Any = self._data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def require(self, key: str) -> Any:
        v = self.get(key, default=self._MISSING)
        if v is self._MISSING:
            raise KeyError(f"missing config key: {key}")
        return v

    def get_section(self, section: str) -> dict:
        value = self._data.get(section, {})
        return deepcopy(value) if isinstance(value, dict) else {}

    def update(self, patch: dict) -> None:
        _deep_update(self._data, patch)

    def to_dict(self) -> dict:
        return deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        return self.require(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key, default=self._MISSING) is not self._MISSING


def _deep_update(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_config(config_path: Optional[str] = None) -> Config:
    """从 YAML 加载配置；找不到文件则返回默认配置。"""
    if config_path is None:
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / "config" / "default.yaml"
        if candidate.exists():
            config_path = str(candidate)

    if not config_path or not Path(config_path).exists():
        return Config()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    merged = default_config()
    _deep_update(merged, raw)
    return Config(merged)


# ----------------------------------------------------------------------
# 别名: 与 README / scripts 中保持一致
# ----------------------------------------------------------------------
def get_config(config_path: Optional[str] = None) -> Config:
    """:func:`load_config` 的别名."""
    return load_config(config_path)
