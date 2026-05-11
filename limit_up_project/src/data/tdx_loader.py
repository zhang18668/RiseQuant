"""D-001 通达信本地数据加载器

读取通达信 ``.day`` 二进制日线文件。

文件格式
--------
每条记录 32 字节：

================  ===============  ============
偏移              字段              类型
----------------  ---------------  ------------
0..4              date             int32 (YYYYMMDD)
4..8              open*100          int32
8..12             high*100          int32
12..16            low*100           int32
16..20            close*100         int32
20..24            amount            float32  (成交额，元)
24..28            volume            int32    (成交量，手)
28..32            reserved          int32
================  ===============  ============

返回的 ``DataFrame`` 列约定 (满足方案 1.3 数据流):

``date | code | open | high | low | close | volume | turnover | change_pct``
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# 通达信日线文件每条记录 32 字节 (4i 1f 1i 1i)
_RECORD_FMT = "<IIIIIfII"  # date, open, high, low, close, amount, volume, _
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)


def _parse_day_bytes(data: bytes) -> pd.DataFrame:
    """将原始 ``.day`` 字节解析成 DataFrame（不含 code/change_pct 列）。"""
    n = len(data) // _RECORD_SIZE
    if n == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "turnover"])

    arr = np.frombuffer(data[: n * _RECORD_SIZE], dtype=np.dtype(
        [
            ("date", "<u4"),
            ("open", "<u4"),
            ("high", "<u4"),
            ("low", "<u4"),
            ("close", "<u4"),
            ("amount", "<f4"),
            ("volume", "<u4"),
            ("_pad", "<u4"),
        ]
    ))

    df = pd.DataFrame({
        "date": pd.to_datetime(arr["date"].astype(str), format="%Y%m%d"),
        "open": arr["open"].astype(np.float64) / 100.0,
        "high": arr["high"].astype(np.float64) / 100.0,
        "low": arr["low"].astype(np.float64) / 100.0,
        "close": arr["close"].astype(np.float64) / 100.0,
        "volume": arr["volume"].astype(np.float64),
        "turnover": arr["amount"].astype(np.float64),
    })
    return df


class TDXDataLoader:
    """通达信本地数据加载器。"""

    DEFAULT_PATHS: List[str] = [
        r"C:\new_tdx\vipdoc",
        r"C:\TdxW\vipdoc",
        r"C:\Program Files\TdxW\vipdoc",
        r"C:\Program Files (x86)\TdxW\vipdoc",
        r"D:\new_tdx\vipdoc",
        r"D:\TdxW\vipdoc",
    ]

    def __init__(self, data_path: Optional[str] = None) -> None:
        if data_path:
            self.data_path: Optional[Path] = Path(data_path)
        else:
            self.data_path = self._find_tdx_path()

        if self.data_path and self.data_path.exists():
            logger.info(f"TDX data dir: {self.data_path}")
        else:
            logger.warning(f"TDX data dir not found: {self.data_path}")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _find_tdx_path(self) -> Optional[Path]:
        env = os.environ.get("TDX_PATH")
        if env and Path(env).exists():
            return Path(env)
        for p in self.DEFAULT_PATHS:
            if Path(p).exists():
                return Path(p)
        return None

    @staticmethod
    def _infer_market(code: str) -> str:
        """简单推断市场归属。"""
        c = str(code).zfill(6)
        if c.startswith(("60", "68", "9", "5", "11")):
            return "sh"
        return "sz"

    def _day_file(self, code: str, market: Optional[str] = None) -> Optional[Path]:
        if not self.data_path:
            return None
        market = market or self._infer_market(code)
        c = str(code).zfill(6)
        return self.data_path / market / "lday" / f"{market}{c}.day"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_day_file(self, code: str, market: Optional[str] = None) -> pd.DataFrame:
        """加载单只股票的日线数据。

        若文件不存在返回空 DataFrame。返回值列：
        ``date, code, open, high, low, close, volume, turnover, change_pct``
        """
        path = self._day_file(code, market)
        if path is None or not path.exists():
            logger.debug(f"day file missing: {path}")
            return self._empty_frame()

        with open(path, "rb") as f:
            raw = f.read()

        df = _parse_day_bytes(raw)
        if df.empty:
            return self._empty_frame()

        df["code"] = str(code).zfill(6)
        df["change_pct"] = df["close"].pct_change() * 100.0
        df = df[["date", "code", "open", "high", "low", "close", "volume", "turnover", "change_pct"]]
        return df.reset_index(drop=True)

    def load_batch(
        self,
        codes: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量加载若干股票的日线数据并合并。"""
        frames: List[pd.DataFrame] = []
        for code in codes:
            df = self.load_day_file(code)
            if df.empty:
                continue
            if start_date is not None:
                df = df[df["date"] >= pd.to_datetime(start_date)]
            if end_date is not None:
                df = df[df["date"] <= pd.to_datetime(end_date)]
            if not df.empty:
                frames.append(df)
        if not frames:
            return self._empty_frame()
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["code", "date"]).reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "date", "code", "open", "high", "low",
                "close", "volume", "turnover", "change_pct",
            ]
        )

    @staticmethod
    def parse_bytes(data: bytes) -> pd.DataFrame:
        """暴露解析逻辑给测试。"""
        return _parse_day_bytes(data)
