"""U-002 统一日志工具

封装 ``loguru``，提供两个常用入口：
- :func:`setup_logger` 配置全局 handler（控制台 + 可选文件）；
- :func:`get_logger`  返回带名字绑定的 logger，供各模块按需使用。

未配置时 ``loguru`` 仍可正常输出到 stderr，所以业务侧无需强制调用 ``setup_logger``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

_DEFAULT_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
    "{extra[name]}:{function}:{line} - {message}"
)


def setup_logger(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    rotation: str = "00:00",
    retention: str = "30 days",
) -> None:
    """配置全局日志输出。"""
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level=log_level,
        format=_DEFAULT_FORMAT,
        enqueue=False,
    )
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            log_path / "limit_up_{time:YYYY-MM-DD}.log",
            level=log_level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            format=_DEFAULT_FORMAT,
        )


def get_logger(name: str = "limit_up"):
    """返回带 ``name`` 绑定的 logger 实例。"""
    return _loguru_logger.bind(name=name)
