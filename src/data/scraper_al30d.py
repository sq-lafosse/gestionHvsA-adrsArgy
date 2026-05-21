"""
Scraper for AL30D sovereign bond prices in USD.

Instrument chain:
  AY24.BA   (2015-01-01 → 2020-08-31)  — pre-AL30D proxy, same issuer/similar maturity
  AL30D.BA  (2020-09-01 → present)     — actual instrument

Fallback per instrument: yfinance → BYMA open API → last cached value + warning.
The AY24 proxy is documented and justified in the academic paper.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TRANSITION_DATE = pd.Timestamp("2020-09-01")
_AY24_TICKER = "AY24.BA"
_AL30D_TICKER = "AL30D.BA"
_BYMA_URL = "https://open.byma.com.ar/api/prices/bonds"

_PROXY_NOTE = (
    "AL30D was issued 2020-09-04. For 2015-2020 the AY24 bond (same sovereign issuer, "
    "similar maturity profile) is used as proxy. The splice is documented in the paper. "
    "All prices are in USD."
)


def get_sovereign_bond(
    start: str,
    end: str,
    cached_series: pd.Series | None = None,
) -> pd.Series:
    """
    Return weekly sovereign bond USD prices covering [start, end].

    Automatically uses AY24.BA for dates before _TRANSITION_DATE and AL30D.BA
    from _TRANSITION_DATE onward. cached_series is the last-resort fallback.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    segments: list[pd.Series] = []

    if start_ts < _TRANSITION_DATE:
        proxy_end = min(end_ts, _TRANSITION_DATE - pd.Timedelta(days=1))
        seg = _fetch_instrument(_AY24_TICKER, start_ts, proxy_end, cached_series)
        if not seg.empty:
            segments.append(seg)

    if end_ts >= _TRANSITION_DATE:
        live_start = max(start_ts, _TRANSITION_DATE)
        seg = _fetch_instrument(_AL30D_TICKER, live_start, end_ts, cached_series)
        if not seg.empty:
            segments.append(seg)

    if not segments:
        logger.warning("sovereign_bond: all sources failed — returning empty series")
        s = pd.Series(name="sovereign_bond_usd", dtype=float)
    else:
        s = pd.concat(segments).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = "sovereign_bond_usd"

    s.attrs = {
        "source_instrument": [
            {
                "instrument": "AY24",
                "ticker": _AY24_TICKER,
                "role": "proxy",
                "from": "2015-01-01",
                "to": "2020-08-31",
            },
            {
                "instrument": "AL30D",
                "ticker": _AL30D_TICKER,
                "role": "primary",
                "from": "2020-09-01",
                "to": "present",
            },
        ],
        "proxy_note": _PROXY_NOTE,
    }
    return s


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_instrument(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cached_series: pd.Series | None,
) -> pd.Series:
    """Try yfinance → BYMA API → cached fallback for one instrument."""
    s = _try_yfinance(ticker, start, end)
    if not s.empty:
        return s
    s = _try_byma(ticker, start, end)
    if not s.empty:
        return s
    return _use_cached_fallback(ticker, start, end, cached_series)


def _try_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        raw = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1wk",
            auto_adjust=True,
            raise_errors=False,
        )
        if raw.empty:
            return pd.Series(dtype=float)
        s = raw["Close"].copy()
        s.index = (
            s.index.tz_localize(None).normalize()
            if s.index.tz else s.index.normalize()
        )
        return s.resample("W-FRI").last().dropna()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] yfinance failed: %s", ticker, exc)
        return pd.Series(dtype=float)


def _try_byma(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        import requests

        params = {
            "symbol": ticker.replace(".BA", ""),
            "date_from": start.strftime("%Y-%m-%d"),
            "date_to": end.strftime("%Y-%m-%d"),
        }
        resp = requests.get(_BYMA_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        price_col = next(
            (c for c in df.columns if "price" in c.lower() or "close" in c.lower()),
            df.columns[-1],
        )
        s = pd.Series(
            pd.to_numeric(df[price_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col]).normalize(),
        )
        return s.resample("W-FRI").last().dropna()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] BYMA API failed: %s", ticker, exc)
        return pd.Series(dtype=float)


def _use_cached_fallback(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cached_series: pd.Series | None,
) -> pd.Series:
    if cached_series is None or cached_series.empty:
        logger.warning("[%s] all sources failed and no cache available", ticker)
        return pd.Series(dtype=float)

    window = cached_series.loc[start:end]
    if not window.empty:
        logger.warning(
            "[%s] using cached fallback for %s → %s", ticker, start.date(), end.date()
        )
        return window.copy()

    valid = cached_series.dropna()
    if valid.empty:
        logger.warning("[%s] cache has no valid values", ticker)
        return pd.Series(dtype=float)

    # Extend last known value across the requested window
    idx = pd.date_range(start, end, freq="W-FRI")
    logger.warning(
        "[%s] extending last cached value (%.4f) to fill %s → %s",
        ticker, valid.iloc[-1], start.date(), end.date(),
    )
    return pd.Series(valid.iloc[-1], index=idx)
