"""
src/backtest — walk-forward backtest engine for gestionHvsA-adrsArgy.

Public API:
    run_backtest    — simulate portfolio + benchmarks, return BacktestResult
    BacktestResult  — dataclass: portfolio (pd.Series), benchmarks (dict[str, pd.Series]),
                      rebalance_dates (list[pd.Timestamp])
"""
from .engine import BacktestResult, run_backtest

__all__ = ["run_backtest", "BacktestResult"]
