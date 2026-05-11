"""D-003 数据缓存管理

为特征/标签/事件等中间结果提供基于 Parquet 的缓存读写接口。

仅有一个核心抽象 :class:`CacheManager`，每次实例化指定一个目录，
通过 ``key -> 文件名`` 的方式存取 DataFrame，简单可靠。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """基于本地文件系统的 DataFrame 缓存器。"""

    def __init__(self, cache_dir: str, format: str = "parquet") -> None:
        if format not in ("parquet", "csv", "feather", "pickle"):
            raise ValueError(f"unsupported cache format: {format}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

    # ------------------------------------------------------------------
    @staticmethod
    def make_key(*parts: object) -> str:
        """把若干 hashable 拼成稳定的缓存键。"""
        raw = json.dumps([str(p) for p in parts], sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.{self.format}"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def save(self, key: str, df: pd.DataFrame) -> Path:
        path = self._path(key)
        if self.format == "parquet":
            try:
                df.to_parquet(path, index=False)
            except (ImportError, ValueError) as exc:
                logger.warning(f"parquet unavailable ({exc}); fallback to pickle")
                path = path.with_suffix(".pickle")
                df.to_pickle(path)
        elif self.format == "csv":
            df.to_csv(path, index=False)
        elif self.format == "feather":
            df.reset_index(drop=True).to_feather(path)
        else:  # pickle
            df.to_pickle(path)
        logger.debug(f"cache saved: {path}")
        return path

    def load(self, key: str) -> Optional[pd.DataFrame]:
        path = self._path(key)
        if not path.exists():
            alt = path.with_suffix(".pickle")
            if alt.exists():
                return pd.read_pickle(alt)
            return None
        if self.format == "parquet":
            return pd.read_parquet(path)
        if self.format == "csv":
            return pd.read_csv(path)
        if self.format == "feather":
            return pd.read_feather(path)
        return pd.read_pickle(path)

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        count = 0
        for f in self.cache_dir.glob(f"*.{self.format}"):
            f.unlink()
            count += 1
        return count
