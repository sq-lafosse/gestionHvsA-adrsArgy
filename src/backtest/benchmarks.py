"""
Benchmark equity curves for gestionHvsA-adrsArgy.

Three benchmarks, each normalized to 1.0 at the start of the evaluation period:

    ew_bnh        — Equal-weight Buy & Hold across all ADRs (no rebalancing)
    al30d_static  — 100 % AL30D from start (static hold)
    merval_usd    — Merval ARS / CCL (Contado con Liquidación) buy & hold

CCL is the relevant FX for ADR/local arbitrage and for investors operating in
the USD-denominated space of Argentine markets — consistent with the project
benchmark methodology described in CLAUDE.md.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_benchmarks(
    adrs: pd.DataFrame,
    sovereign_bond: pd.Series,
    merval: pd.Series,
    ccl: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """
    Compute equity curves for all three benchmarks over [start_date, end_date].

    Parameters
    ----------
    adrs : pd.DataFrame
        Weekly ADR adjusted-close prices (columns = tickers).
    sovereign_bond : pd.Series
        Weekly AL30D USD prices.
    merval : pd.Series
        Weekly Merval ARS prices.
    ccl : pd.Series
        Weekly CCL (ARS/USD) prices.
    start_date, end_date : pd.Timestamp
        Inclusive bounds of the evaluation window, matching the portfolio period.

    Returns
    -------
    dict with keys "ew_bnh", "al30d_static", "merval_usd".
    Each value is a pd.Series normalized to 1.0 at start_date.
    """
    return {
        "ew_bnh":       _ew_buy_and_hold(adrs, start_date, end_date),
        "al30d_static": _al30d_static(sovereign_bond, start_date, end_date),
        "merval_usd":   _merval_usd(merval, ccl, start_date, end_date),
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _ew_buy_and_hold(
    adrs: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Equal-weight buy & hold across all ADRs.

    Invests equal dollar amounts in each ADR at start_date and holds without
    rebalancing. Portfolio value = mean of individually normalized price series.
    Tickers with NaN on start_date are excluded (no initial position possible).
    """
    prices = adrs.loc[start_date:end_date].ffill().dropna(how="all")

    if prices.empty:
        logger.warning(
            "_ew_buy_and_hold: no ADR data for %s → %s", start_date.date(), end_date.date()
        )
        return pd.Series(dtype=float, name="ew_bnh")

    valid_cols = prices.columns[prices.iloc[0].notna()]
    if valid_cols.empty:
        logger.warning("_ew_buy_and_hold: all tickers NaN at start_date %s", start_date.date())
        return pd.Series(dtype=float, name="ew_bnh")

    if len(valid_cols) < len(prices.columns):
        dropped = set(prices.columns) - set(valid_cols)
        logger.warning(
            "_ew_buy_and_hold: excluded %d tickers with NaN at start: %s", len(dropped), dropped
        )

    prices = prices[valid_cols]
    normalized = prices / prices.iloc[0]
    equity = normalized.mean(axis=1)
    equity.name = "ew_bnh"
    return equity


def _al30d_static(
    sovereign_bond: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """100 % AL30D held from start_date to end_date without rebalancing."""
    prices = sovereign_bond.loc[start_date:end_date].ffill().dropna()

    if prices.empty:
        logger.warning(
            "_al30d_static: no AL30D data for %s → %s", start_date.date(), end_date.date()
        )
        return pd.Series(dtype=float, name="al30d_static")

    equity = prices / prices.iloc[0]
    equity.name = "al30d_static"
    return equity


def _merval_usd(
    merval: pd.Series,
    ccl: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Merval in USD: Merval ARS / CCL (Contado con Liquidación).

    Both series are aligned on their common weekly index. Rows with NaN
    in either series are dropped before normalization.
    """
    merged = (
        pd.DataFrame({
            "merval_ars": merval.loc[start_date:end_date],
            "ccl":        ccl.loc[start_date:end_date],
        })
        .ffill()
        .dropna()
    )

    if merged.empty:
        logger.warning(
            "_merval_usd: no overlapping data for %s → %s", start_date.date(), end_date.date()
        )
        return pd.Series(dtype=float, name="merval_usd")

    merval_usd_prices = merged["merval_ars"] / merged["ccl"]
    equity = merval_usd_prices / merval_usd_prices.iloc[0]
    equity.name = "merval_usd"
    return equity
