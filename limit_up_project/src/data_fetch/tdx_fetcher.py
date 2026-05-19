"""TDXFetcher — thin adapter over existing src.data.tdx_loader.TDXDataLoader.

Keeps the existing TDX parsing code (in src/data/tdx_loader.py) as the
single source of truth, and exposes it through the new DataFetcher API.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.data.tdx_loader import TDXDataLoader
from src.data_fetch.base import DataFetcher


class TDXFetcher(DataFetcher):
    def __init__(self, vipdoc_path: str):
        self._loader = TDXDataLoader(vipdoc_path)
        self.data_path = self._loader.data_path

    def load_stock_list(self, exchange: str = "sh") -> pd.DataFrame:
        return self._loader.load_stock_list(exchange)

    def load_batch(
        self,
        codes: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._loader.load_batch(codes=list(codes),
                                       start_date=start_date,
                                       end_date=end_date)

    def is_main_board(self, code: str) -> bool:
        # delegate to the original loader's logic for full compatibility
        return self._loader.is_main_board(code)
