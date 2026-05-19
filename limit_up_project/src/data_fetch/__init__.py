"""Data fetching layer.

Pure-fetch concerns (no cleaning, no validation). Two backends:
- TDXFetcher: local 通达信 .day files
- AkshareFetcher: online via akshare

Each fetcher returns raw OHLCV-style DataFrame with the canonical columns:
    date, code, open, high, low, close, volume, turnover, change_pct
Cleaning + validation is done by ``src.data_clean.DataCleaner``.
"""
from src.data_fetch.base import DataFetcher
from src.data_fetch.tdx_fetcher import TDXFetcher
from src.data_fetch.akshare_fetcher import AkshareFetcher

__all__ = ["DataFetcher", "TDXFetcher", "AkshareFetcher"]
