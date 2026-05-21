"""
Walk-forward backtest engine for gestionHvsA-adrsArgy.

Simulates the portfolio with explicit position tracking in pandas.

Note: D18 originally specified vbt.Portfolio.from_weights(), but that method was
removed in vectorbt 1.0.0. The manual simulation below is functionally equivalent,
more auditable for the academic paper, and fully vectorbt-version-independent.
vectorbt is used in src/metrics/ for performance statistics (Sharpe, drawdown, etc.).

Rebalancing rule (D17): triggered only on regime change (Risk-On ↔ Risk-Off).
  - The first month always triggers an initial rebalance (portfolio funding).
  - Subsequent months only rebalance if the regime flips.
  - Between rebalances, positions are held; weights drift with prices.

Execution lag (D22, anti-look-ahead): weights decided at end of month M take
effect at the first weekly close of month M+1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.allocation import AllocationResult

from .benchmarks import compute_benchmarks

logger = logging.getLogger(__name__)

_AL30D = "AL30D"


# ─── Public types ─────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    portfolio: pd.Series                          # equity curve, normalized to 1.0 at first rebalance
    benchmarks: dict[str, pd.Series]             # each normalized to 1.0 at portfolio start date
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    monthly_allocations: dict[pd.Timestamp, AllocationResult],
    adrs: pd.DataFrame,
    sovereign_bond: pd.Series,
    merval: pd.Series,
    ccl: pd.Series,
) -> BacktestResult:
    """
    Run the walk-forward backtest for the given out-of-sample period.

    Parameters
    ----------
    monthly_allocations : dict[pd.Timestamp, AllocationResult]
        Month-end decision date → AllocationResult, one entry per month.
        Keys must be month-end timestamps (e.g., pd.Timestamp("2024-01-31")).
    adrs : pd.DataFrame
        Weekly ADR adjusted-close prices (columns = tickers).
    sovereign_bond : pd.Series
        Weekly AL30D USD prices.
    merval : pd.Series
        Weekly Merval ARS prices (for Merval-in-USD benchmark).
    ccl : pd.Series
        Weekly CCL ARS/USD prices (for Merval-in-USD benchmark).

    Returns
    -------
    BacktestResult with portfolio equity curve, three benchmark curves,
    and the list of rebalance execution dates.
    """
    if not monthly_allocations:
        raise ValueError("monthly_allocations is empty — nothing to backtest.")

    prices = _build_consolidated_prices(adrs, sovereign_bond)

    rebalance_weights, rebalance_dates = _identify_rebalance_events(
        monthly_allocations, prices.index
    )

    if not rebalance_weights:
        raise ValueError(
            "No rebalance events could be scheduled. "
            "Ensure price data covers at least one month after the allocation period."
        )

    equity = _run_manual_simulation(prices, rebalance_weights)

    start_date = equity.index[0]
    end_date = equity.index[-1]
    benchmarks = compute_benchmarks(adrs, sovereign_bond, merval, ccl, start_date, end_date)

    logger.info(
        "Backtest complete: %s → %s | %d weeks | %d rebalances",
        start_date.date(), end_date.date(), len(equity), len(rebalance_dates),
    )

    return BacktestResult(
        portfolio=equity,
        benchmarks=benchmarks,
        rebalance_dates=rebalance_dates,
    )


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_consolidated_prices(
    adrs: pd.DataFrame,
    sovereign_bond: pd.Series,
) -> pd.DataFrame:
    """
    Join ADR prices and AL30D into a single weekly price DataFrame.

    sovereign_bond is renamed to "AL30D" to match AllocationResult.weights keys.
    Gaps are forward-filled so the simulation has continuous price data.
    """
    al30d = sovereign_bond.rename(_AL30D).to_frame()
    prices = adrs.join(al30d, how="outer").ffill()
    return prices


def _identify_rebalance_events(
    monthly_allocations: dict[pd.Timestamp, AllocationResult],
    price_index: pd.DatetimeIndex,
) -> tuple[dict[pd.Timestamp, dict[str, float]], list[pd.Timestamp]]:
    """
    Map each regime-change month to its execution date (first close of M+1).

    A rebalance is triggered on:
    - The first month in the period (portfolio funding — always).
    - Any subsequent month where the regime flips vs. the active regime.

    Returns
    -------
    rebalance_weights : execution_date → weights dict
    rebalance_dates   : ordered list of execution dates
    """
    months = sorted(monthly_allocations.keys())
    rebalance_weights: dict[pd.Timestamp, dict[str, float]] = {}
    rebalance_dates: list[pd.Timestamp] = []
    prev_regime: str | None = None

    for month_end in months:
        allocation = monthly_allocations[month_end]
        is_first = prev_regime is None
        regime_changed = allocation.regime != prev_regime

        if is_first or regime_changed:
            next_month_start = (month_end + pd.DateOffset(days=1)).normalize()
            candidates = price_index[price_index >= next_month_start]

            if candidates.empty:
                logger.warning(
                    "No price data after %s — rebalance for %s skipped",
                    next_month_start.date(), month_end.strftime("%Y-%m"),
                )
                prev_regime = allocation.regime
                continue

            exec_date = candidates[0]
            rebalance_weights[exec_date] = allocation.weights
            rebalance_dates.append(exec_date)

            logger.info(
                "Rebalance: decision=%s | %s → %s | execution=%s",
                month_end.strftime("%Y-%m"),
                prev_regime or "initial",
                allocation.regime,
                exec_date.date(),
            )

        prev_regime = allocation.regime

    return rebalance_weights, rebalance_dates


def _run_manual_simulation(
    prices: pd.DataFrame,
    rebalance_weights: dict[pd.Timestamp, dict[str, float]],
) -> pd.Series:
    """
    Simulate the portfolio with explicit position tracking.

    At each rebalance date: sell all current holdings and reinvest total
    portfolio value according to target weights (close-of-day execution).
    Between rebalances: hold positions — weights drift with prices.

    The equity curve starts at 1.0 on the first rebalance date (portfolio funding)
    and is recorded at each weekly close thereafter.

    Tickers with NaN or zero prices on a rebalance date receive zero units;
    their intended allocation remains in cash and is absorbed into rounding.
    """
    rebalance_set = set(rebalance_weights.keys())
    tickers = prices.columns.tolist()
    sim_index = prices.index[prices.index >= min(rebalance_set)]

    units = pd.Series(0.0, index=tickers)
    cash = 0.0
    initialized = False
    equity: dict[pd.Timestamp, float] = {}

    for date in sim_index:
        current_prices = prices.loc[date]

        if not initialized:
            portfolio_value = 1.0
        else:
            holdings_value = (units * current_prices).fillna(0.0).sum()
            portfolio_value = cash + holdings_value

        equity[date] = portfolio_value

        if date in rebalance_set:
            target_weights = rebalance_weights[date]
            new_units = pd.Series(0.0, index=tickers)

            for ticker in tickers:
                w = target_weights.get(ticker, 0.0)
                p = current_prices.get(ticker)
                if w > 0 and p is not None and not pd.isna(p) and p > 0:
                    new_units[ticker] = (portfolio_value * w) / p

            units = new_units
            invested = (units * current_prices).fillna(0.0).sum()
            cash = portfolio_value - invested
            initialized = True

    result = pd.Series(equity)
    if result.empty:
        raise ValueError("Portfolio simulation produced no results — check price data coverage.")

    normalized = (result / result.iloc[0]).rename("portfolio")
    return normalized
