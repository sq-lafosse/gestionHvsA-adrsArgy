"""
Scraper for CCL (Contado con Liquidación) ARS/USD exchange rate.

Primary:  Ámbito Financiero historical API.
Fallback: GGAL ratio — CCL ≈ (GGAL.BA × 10) / GGAL_ADR.
          GGAL_ADR is passed in from loader.py (already downloaded, no circular import).
          GGAL.BA is downloaded locally via yfinance.
Last resort: cached_series.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_GGAL_ADR_RATIO = 10
_GGAL_BA_TICKER = "GGAL.BA"
_AMBITO_URL = (
    "https://mercados.ambito.com//dolar/contadoconliqui/historico-general/{start}/{end}"
)


def get_ccl(
    start: str,
    end: str,
    ggal_adr: pd.Series | None = None,
    cached_series: pd.Series | None = None,
) -> pd.Series:
    """
    Return weekly CCL (ARS/USD) covering [start, end].

    ggal_adr: GGAL ADR USD price series from loader.py — used for ratio fallback.
    cached_series: last-resort fallback if all live sources fail.
    """
    s = _try_ambito(start, end)
    if not s.empty:
        s.attrs = {"source": "ambito_api"}
        s.name = "ccl"
        return s

    if ggal_adr is not None and not ggal_adr.empty:
        s = _try_ggal_ratio(start, end, ggal_adr)
        if not s.empty:
            s.attrs = {"source": "ggal_ratio"}
            s.name = "ccl"
            return s

    if cached_series is not None and not cached_series.empty:
        window = cached_series.loc[start:end]
        if not window.empty:
            logger.warning("CCL: using cached fallback for %s → %s", start, end)
            s = window.copy()
            s.attrs = {"source": "cache"}
            s.name = "ccl"
            return s

    logger.warning("CCL: all sources failed — returning empty series")
    s = pd.Series(name="ccl", dtype=float)
    s.attrs = {"source": "none"}
    return s


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _try_ambito(start: str, end: str) -> pd.Series:
    try:
        import requests

        start_fmt = pd.Timestamp(start).strftime("%d/%m/%Y")
        end_fmt = pd.Timestamp(end).strftime("%d/%m/%Y")
        url = _AMBITO_URL.format(start=start_fmt, end=end_fmt)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data or len(data) < 2:
            return pd.Series(dtype=float)

        rows = data[1:]  # first row is the header
        dates = pd.to_datetime([r[0] for r in rows], dayfirst=True, errors="coerce")
        values = pd.to_numeric(
            [str(r[1]).replace(",", ".") for r in rows], errors="coerce"
        )
        s = pd.Series(values, index=dates).dropna()
        s.index = s.index.normalize()
        return s.resample("W-FRI").last().dropna()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CCL Ámbito API failed: %s", exc)
        return pd.Series(dtype=float)


def _try_ggal_ratio(start: str, end: str, ggal_adr: pd.Series) -> pd.Series:
    try:
        raw = yf.Ticker(_GGAL_BA_TICKER).history(
            start=start,
            end=end,
            interval="1wk",
            auto_adjust=True,
            raise_errors=False,
        )
        if raw.empty:
            logger.debug("CCL GGAL ratio: GGAL.BA returned empty")
            return pd.Series(dtype=float)

        ggal_ba = raw["Close"].copy()
        ggal_ba.index = (
            ggal_ba.index.tz_localize(None).normalize()
            if ggal_ba.index.tz else ggal_ba.index.normalize()
        )
        ggal_ba = ggal_ba.resample("W-FRI").last()

        adr = ggal_adr.copy()
        adr.index = adr.index.normalize()
        adr = adr.resample("W-FRI").last()

        aligned = pd.concat([ggal_ba.rename("ba"), adr.rename("adr")], axis=1).dropna()
        if aligned.empty:
            logger.debug("CCL GGAL ratio: no overlapping dates after alignment")
            return pd.Series(dtype=float)

        ccl = (aligned["ba"] * _GGAL_ADR_RATIO) / aligned["adr"]
        logger.info("CCL: GGAL ratio fallback used (%d observations)", len(ccl))
        return ccl
    except Exception as exc:  # noqa: BLE001
        logger.debug("CCL GGAL ratio fallback failed: %s", exc)
        return pd.Series(dtype=float)
