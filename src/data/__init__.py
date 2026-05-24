"""
src/data — data acquisition, validation, and cache management.

Entry points:
  run_historical(tickers, start, end)          — download + persist full 2015-2023 dataset
  run_live_month(tickers, cutoff_date)          — download + append one live month

Load with anti-leakage cutoff:
  load_adrs_up_to, load_sovereign_bond_up_to,
  load_merval_up_to, load_ccl_up_to, load_macro_up_to
"""
from .cache_manager import (
    CachePayload,
    append_live_month,
    get_last_cached_date,
    historical_cache_exists,
    load_adrs_up_to,
    load_ccl_up_to,
    load_macro_up_to,
    load_merval_up_to,
    load_sovereign_bond_up_to,
    save_historical_cache,
)
from .downloader import download_adrs, download_merval
from .loader import run_historical, run_live_month
from .scraper_al30d import get_sovereign_bond
from .scraper_ccl import get_ccl
from .scraper_macro import get_all_macro, get_embi, get_ipc, get_reservas, get_tc_oficial
from .news_downloader import download_macro_documents, download_news_for_period
from .validator import ValidationReport, validate_all, validate_macro, validate_prices

__all__ = [
    # loader — primary entry points
    "run_historical",
    "run_live_month",
    # cache_manager
    "CachePayload",
    "historical_cache_exists",
    "get_last_cached_date",
    "save_historical_cache",
    "append_live_month",
    "load_adrs_up_to",
    "load_sovereign_bond_up_to",
    "load_merval_up_to",
    "load_ccl_up_to",
    "load_macro_up_to",
    # downloader
    "download_adrs",
    "download_merval",
    # scrapers
    "get_sovereign_bond",
    "get_ccl",
    "get_all_macro",
    "get_ipc",
    "get_embi",
    "get_reservas",
    "get_tc_oficial",
    # news downloader
    "download_news_for_period",
    "download_macro_documents",
    # validator
    "ValidationReport",
    "validate_prices",
    "validate_macro",
    "validate_all",
]
