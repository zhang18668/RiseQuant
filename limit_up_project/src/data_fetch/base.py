"""Abstract DataFetcher interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import pandas as pd


class DataFetcher(ABC):
    """Abstract base. Concrete fetchers implement load_stock_list + load_batch."""

    @abstractmethod
    def load_stock_list(self, exchange: str = "sh") -> pd.DataFrame:
        """Return DataFrame with at least column 'code'."""

    @abstractmethod
    def load_batch(
        self,
        codes: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return concatenated daily bars across codes.

        Required columns: date, code, open, high, low, close, volume,
                          turnover, change_pct.
        """

    def is_main_board(self, code: str) -> bool:
        """Default: main-board if code starts with 60 / 00 / 30 (沪深主板+创业板)."""
        c = str(code).strip()
        return c.startswith(("60", "000", "001", "002", "003", "300", "301"))
