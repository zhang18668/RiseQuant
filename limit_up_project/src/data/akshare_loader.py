"""D-002 AKShare 数据补充

主要用于：
- 获取 ST 股名单 (``is_st`` 字段)；
- 获取交易日历；
- 在没有通达信本地数据时，作为日线数据回退。

由于 ``akshare`` 依赖在线接口，单元测试中应当 mock，绝不要直接联网。
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

try:  # pragma: no cover - 导入异常分支与环境相关
    import akshare as ak  # type: ignore
    _HAS_AKSHARE = True
except Exception:  # noqa: BLE001
    ak = None  # type: ignore
    _HAS_AKSHARE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AKShareLoader:
    """AKShare 数据加载器。"""

    def __init__(self, require: bool = False) -> None:
        if require and not _HAS_AKSHARE:
            raise ImportError("akshare is not installed")
        self.available = _HAS_AKSHARE

    # ------------------------------------------------------------------
    def load_daily(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """加载单只股票的日线（qfq 前复权）。

        返回列与 :class:`TDXDataLoader` 保持一致。
        """
        if not self.available:
            logger.warning("akshare not available; returning empty frame")
            return self._empty()

        try:  # pragma: no cover - 实际网络调用不在单测中触发
            raw = ak.stock_zh_a_hist(
                symbol=str(code).zfill(6),
                period="daily",
                start_date=(start_date or "19900101").replace("-", ""),
                end_date=(end_date or "21000101").replace("-", ""),
                adjust=adjust,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"akshare fetch failed for {code}: {exc}")
            return self._empty()

        return self._normalize(raw, code)

    def get_st_codes(self) -> List[str]:
        """获取当前 ST 股票代码列表。失败时返回空列表。"""
        if not self.available:
            return []
        try:  # pragma: no cover - 网络调用
            df = ak.stock_zh_a_st_em()
            col = "代码" if "代码" in df.columns else df.columns[0]
            return df[col].astype(str).tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to fetch ST list: {exc}")
            return []

    # ------------------------------------------------------------------
    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "date", "code", "open", "high", "low",
                "close", "volume", "turnover", "change_pct",
            ]
        )

    @classmethod
    def _normalize(cls, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """将 akshare 列名标准化为项目约定。"""
        if df is None or df.empty:
            return cls._empty()

        rename_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "turnover",
            "涨跌幅": "change_pct",
        }
        df = df.rename(columns=rename_map).copy()
        df["code"] = str(code).zfill(6)
        df["date"] = pd.to_datetime(df["date"])
        wanted = ["date", "code", "open", "high", "low", "close", "volume", "turnover", "change_pct"]
        for col in wanted:
            if col not in df.columns:
                df[col] = pd.NA
        return df[wanted].reset_index(drop=True)
