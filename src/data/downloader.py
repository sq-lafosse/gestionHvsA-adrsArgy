"""
Downloader — yfinance price downloads for ADRs and Merval.

Downloads are per-ticker (not batch) to isolate failures. Tickers that fail
return a NaN series and are never dropped from the output DataFrame.
Weekly frequency, Adj Close prices, tz-naive DatetimeIndex.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MERVAL_TICKER = "^MERV"


def download_adrs(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download weekly Adj Close prices for a list of ADR tickers.

    Returns a DataFrame with tickers as columns and a tz-naive weekly DatetimeIndex.
    Tickers that fail appear as NaN columns (never dropped).
    """
    series_list = [_download_single(t, start, end) for t in tickers]
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    if not df.empty:
        df.index = _normalize_index(df.index)
    _log_coverage(df, tickers)
    return df


def download_merval(start: str, end: str) -> pd.Series:
    """
    Download weekly Merval index prices in ARS.

    Returns a tz-naive Series named 'merval_ars'.
    USD conversion via CCL is the responsibility of src/metrics/.
    """
    s = _download_single(_MERVAL_TICKER, start, end)
    s.name = "merval_ars"
    return s


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _download_single(ticker: str, start: str, end: str) -> pd.Series:
    """Download one ticker at weekly frequency. Returns empty Series on failure."""
    try:
        raw = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval="1wk",
            auto_adjust=True,
            raise_errors=False,
        )
        if raw.empty:
            logger.warning("[%s] yfinance returned empty DataFrame", ticker)
            return pd.Series(name=ticker, dtype=float)
        s = raw["Close"].copy()
        s.name = ticker
        s.index = _normalize_index(s.index)
        return s
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] download failed: %s", ticker, exc)
        return pd.Series(name=ticker, dtype=float)


def _normalize_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and normalize to midnight."""
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _log_coverage(df: pd.DataFrame, expected_tickers: list[str]) -> None:
    """Log per-ticker download coverage at INFO level."""
    for ticker in expected_tickers:
        if ticker not in df.columns or df[ticker].isna().all():
            logger.warning("[%s] no data downloaded", ticker)
        else:
            valid = df[ticker].dropna()
            logger.info(
                "[%s] %d weekly rows (%s → %s)",
                ticker, len(valid),
                valid.index.min().date(), valid.index.max().date(),
            )
