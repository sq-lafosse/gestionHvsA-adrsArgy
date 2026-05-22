"""
Features — technical signal computation for the ADR universe.

Signals are computed at two granularities:
  portfolio — equal-weight aggregated series → input for PCA/SVM (src/signals/)
  assets    — per-ticker signals             → input for asset selection (src/allocation/)

Four signals:
  price_to_sma{short} : price / SMA(sma_short) — trend position, continuous
  price_to_sma{long}  : price / SMA(sma_long)  — trend position, continuous
  momentum_{N}        : rolling sum of log-returns over N weeks (total log-return of period)
  realized_vol_{N}    : rolling std of log-returns over N weeks

momentum_{N} is the cumulative sum of log-returns (Jegadeesh & Titman convention),
not the mean. min_periods equals the full window — early rows are NaN by design.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatrix:
    portfolio: pd.DataFrame          # (dates × signals) for PCA/SVM
    assets: pd.DataFrame             # MultiIndex columns (ticker, signal) for allocation
    log_returns_portfolio: pd.Series # underlying equal-weight aggregated log-return
    log_returns_assets: pd.DataFrame # per-ticker log-returns


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_features(
    adrs: pd.DataFrame,
    sma_short: int = 20,
    sma_long: int = 30,
    momentum_window: int = 12,
    vol_window: int = 12,
) -> FeatureMatrix:
    """
    Compute technical signals on the ADR universe.

    Returns signals at portfolio level (equal-weight aggregate) and asset level
    (per-ticker). The first max(sma_long, momentum_window, vol_window) rows contain
    NaN — callers in src/signals/ must drop incomplete rows before training PCA/SVM.

    Parameters
    ----------
    adrs : weekly Adj Close prices, columns = tickers, tz-naive DatetimeIndex.
    sma_short, sma_long : SMA windows in weeks (applied to price levels).
    momentum_window : rolling sum window in weeks for momentum signal.
    vol_window : rolling std window in weeks for realized volatility.
    """
    if adrs.empty:
        logger.warning("compute_features: empty ADR DataFrame — returning empty FeatureMatrix")
        empty = pd.DataFrame()
        return FeatureMatrix(
            portfolio=empty,
            assets=empty,
            log_returns_portfolio=pd.Series(dtype=float),
            log_returns_assets=empty,
        )

    log_returns_assets = np.log(adrs / adrs.shift(1))

    # Equal-weight portfolio log-return: nanmean per row, ignores tickers with NaN that week
    log_returns_portfolio = log_returns_assets.mean(axis=1, skipna=True)
    log_returns_portfolio.name = "portfolio"

    # Synthetic portfolio price index (base=100) for SMA computation
    portfolio_price = _reconstruct_price(log_returns_portfolio)

    portfolio_signals = _compute_signals(
        price=portfolio_price,
        log_returns=log_returns_portfolio,
        sma_short=sma_short,
        sma_long=sma_long,
        momentum_window=momentum_window,
        vol_window=vol_window,
    )

    asset_frames: list[pd.DataFrame] = []
    for ticker in adrs.columns:
        ticker_signals = _compute_signals(
            price=adrs[ticker],
            log_returns=log_returns_assets[ticker],
            sma_short=sma_short,
            sma_long=sma_long,
            momentum_window=momentum_window,
            vol_window=vol_window,
        )
        ticker_signals.columns = pd.MultiIndex.from_product(
            [[ticker], ticker_signals.columns]
        )
        asset_frames.append(ticker_signals)

    assets = pd.concat(asset_frames, axis=1)

    warmup = max(sma_long, momentum_window, vol_window)
    logger.info(
        "compute_features: %d weekly rows | %d tickers | warm-up = %d rows",
        len(adrs), len(adrs.columns), warmup,
    )

    return FeatureMatrix(
        portfolio=portfolio_signals,
        assets=assets,
        log_returns_portfolio=log_returns_portfolio,
        log_returns_assets=log_returns_assets,
    )


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_signals(
    price: pd.Series,
    log_returns: pd.Series,
    sma_short: int,
    sma_long: int,
    momentum_window: int,
    vol_window: int,
) -> pd.DataFrame:
    """Compute all four signals for a single price series."""
    # ffill/fillna(0) handles NaN gaps from union-indexing across tickers with
    # different weekday closes — price holds flat, log-return is 0 on filler dates.
    price_ff = price.ffill()
    lr_ff = log_returns.fillna(0.0)

    sma_s = price_ff.rolling(sma_short, min_periods=sma_short).mean()
    sma_l = price_ff.rolling(sma_long, min_periods=sma_long).mean()

    signals = pd.DataFrame(index=price.index)
    signals[f"price_to_sma{sma_short}"] = price_ff / sma_s
    signals[f"price_to_sma{sma_long}"] = price_ff / sma_l
    signals[f"momentum_{momentum_window}"] = (
        lr_ff.rolling(momentum_window, min_periods=momentum_window).sum()
    )
    signals[f"realized_vol_{vol_window}"] = (
        lr_ff.rolling(vol_window, min_periods=vol_window).std()
    )
    return signals


def _reconstruct_price(log_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Reconstruct a synthetic total-return price index from a log-return series.

    The first period has NaN log-return (no prior price); treated as zero return
    so the index starts at base on the first date.
    """
    cumulative = log_returns.fillna(0.0).cumsum()
    return pd.Series(base * np.exp(cumulative), index=log_returns.index)
