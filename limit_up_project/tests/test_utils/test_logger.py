"""U-002 logger 测试 (烟雾测试)。"""
from __future__ import annotations

from src.utils.logger import get_logger, setup_logger


def test_get_logger_smoke():
    log = get_logger("unit_test")
    log.info("hello")  # 不抛异常即可


def test_setup_logger_with_dir(tmp_path):
    setup_logger(log_dir=str(tmp_path), log_level="INFO")
    log = get_logger("unit_test")
    log.info("hello dir")
    # 重置 handler 以免影响其他测试
    setup_logger(log_dir=None, log_level="INFO")
