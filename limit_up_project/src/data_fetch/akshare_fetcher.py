"""AkshareFetcher — thin adapter over existing src.data.akshare_loader."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.data_fetch.base import DataFetcher

try:
    from src.data.akshare_loader import AkshareLoader
    _HAS_AK = True
except Exception:
    AkshareLoader = None
    _HAS_AK = False


class AkshareFetcher(DataFetcher):
    def __init__(self):
        if not _HAS_AK:
            raise ImportError("akshare not available; install or use TDXFetcher")
        self._loader = AkshareLoader()

    def load_stock_list(self, exchange: str = "sh") -> pd.DataFrame:
        if hasattr(self._loader, "load_stock_list"):
            return self._loader.load_stock_list(exchange)
        raise NotImplementedError("AkshareLoader.load_stock_list not implemented")

    def load_batch(
        self,
        codes: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if hasattr(self._loader, "load_batch"):
            return self._loader.load_batch(codes=list(codes),
                                           start_date=start_date,
                                           end_date=end_date)
        raise NotImplementedError("AkshareLoader.load_batch not implemented")
