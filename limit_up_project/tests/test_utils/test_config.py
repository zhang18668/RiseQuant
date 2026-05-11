"""U-003 Config 测试。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, default_config, load_config


class TestConfig:
    def test_default_config_has_required_sections(self):
        cfg = default_config()
        for section in ("data", "event", "feature", "label", "model", "dataset", "backtest"):
            assert section in cfg

    def test_get_dot_path(self):
        cfg = Config()
        assert cfg.get("event.limit_up_threshold") == 9.9

    def test_get_missing_returns_default(self):
        cfg = Config()
        assert cfg.get("missing.key", default=42) == 42

    def test_require_raises_on_missing(self):
        cfg = Config()
        with pytest.raises(KeyError):
            cfg.require("not.exist")

    def test_load_yaml_overrides_defaults(self, tmp_path: Path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"event": {"limit_up_threshold": 5.0}}), encoding="utf-8")
        cfg = load_config(str(path))
        assert cfg.get("event.limit_up_threshold") == 5.0
        # 其他字段仍来自默认配置
        assert cfg.get("event.second_board_days") == 5

    def test_load_missing_path_returns_default(self, tmp_path: Path):
        cfg = load_config(str(tmp_path / "nope.yaml"))
        assert cfg.get("event.limit_up_threshold") == 9.9

    def test_update_deep_merge(self):
        cfg = Config()
        cfg.update({"event": {"limit_up_threshold": 7.0}})
        assert cfg.get("event.limit_up_threshold") == 7.0
        assert cfg.get("event.second_board_days") == 5
