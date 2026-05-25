"""
Performance metrics for gestionHvsA-adrsArgy.

Computes portfolio and benchmark metrics from a BacktestResult.

Design decisions (D23–D27):
  D23: Pure pandas — no vectorbt dependency for metric computation.
  D24: Risk-free rate = 0.0 (annualized). Standard assumption for EM research;
       documented as a model assumption in the academic paper.
  D25: EW B&H ADRs ("ew_bnh") as market proxy for beta, Treynor, Jensen.
  D26: MetricsResult dataclass with portfolio dict, benchmarks dict, summary_df.
  D27: Annualization factor = 52 (weekly data); volatility scaled by sqrt(52).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtest import BacktestResult

logger = logging.getLogger(__name__)

_ANNUALIZATION = 52
_RF = 0.0

_METRIC_KEYS = [
    "total_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "calmar",
    "beta",
    "treynor",
    "jensen",
]

_BENCHMARK_LABELS = {
    "ew_bnh":            "EW B&H ADRs",
    "merval_usd":        "Merval USD",
    "fima_acciones_usd": "FIMA Acciones USD",
}


# ─── Public types ─────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    portfolio:   dict[str, float]                # metric_name → value
    benchmarks:  dict[str, dict[str, float]]     # series_name → {metric_name → value}
    summary_df:  pd.DataFrame                    # pivot: rows=series, cols=metrics


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_metrics(result: BacktestResult) -> MetricsResult:
    """
    Compute all performance metrics for the portfolio and each benchmark.

    Parameters
    ----------
    result : BacktestResult
        Output of run_backtest(); contains equity curve, benchmarks, rebalance dates.

    Returns
    -------
    MetricsResult with portfolio metrics, per-benchmark metrics, and a summary DataFrame.
    """
    market = result.benchmarks.get("ew_bnh", pd.Series(dtype=float))

    portfolio_metrics = _compute_series_metrics(result.portfolio, market)
    logger.info(
        "Portfolio — total_return=%.2f%%  ann_return=%.2f%%  sharpe=%.2f  max_dd=%.2f%%",
        portfolio_metrics["total_return"] * 100,
        portfolio_metrics["ann_return"] * 100,
        portfolio_metrics["sharpe"],
        portfolio_metrics["max_drawdown"] * 100,
    )

    benchmark_metrics: dict[str, dict[str, float]] = {}
    for name, equity in result.benchmarks.items():
        bm = _compute_series_metrics(equity, market)
        benchmark_metrics[name] = bm
        logger.info(
            "%-16s — total_return=%.2f%%  ann_return=%.2f%%  sharpe=%.2f",
            _BENCHMARK_LABELS.get(name, name),
            bm["total_return"] * 100,
            bm["ann_return"] * 100,
            bm["sharpe"],
        )

    all_series: dict[str, dict[str, float]] = {
        "portfolio": portfolio_metrics,
        **benchmark_metrics,
    }
    summary_df = pd.DataFrame(all_series).T[_METRIC_KEYS]
    summary_df.index.name = "series"

    return MetricsResult(
        portfolio=portfolio_metrics,
        benchmarks=benchmark_metrics,
        summary_df=summary_df,
    )


def save_metrics_csv(metrics_result: MetricsResult, output_dir: Path | str) -> Path:
    """
    Export MetricsResult.summary_df as performance_metrics.csv.

    Percentage columns (returns, vol, drawdown) are formatted as "x.xx%".
    Ratio columns (Sharpe, beta, etc.) are formatted to 4 decimal places.
    NaN values are written as "—".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = metrics_result.summary_df.copy().astype(object)

    pct_cols   = ["total_return", "ann_return", "ann_vol", "max_drawdown"]
    ratio_cols = ["sharpe", "calmar", "beta", "treynor", "jensen"]

    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: f"{float(x):.2%}" if _is_finite(x) else "—"
            )
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: f"{float(x):.4f}" if _is_finite(x) else "—"
            )

    path = output_dir / "performance_metrics.csv"
    df.to_csv(path)
    logger.info("Saved: %s", path)
    return path


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_series_metrics(
    equity: pd.Series,
    market_equity: pd.Series,
) -> dict[str, float]:
    """
    Compute all metrics for a single equity curve relative to market_equity.

    Returns a dict with keys matching _METRIC_KEYS; NaN for uncomputable values.
    """
    nan_all = {k: float("nan") for k in _METRIC_KEYS}

    equity = equity.dropna()
    if len(equity) < 2:
        return nan_all

    returns  = equity.pct_change().dropna()
    n_weeks  = len(equity)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann_return   = float((1 + total_return) ** (_ANNUALIZATION / n_weeks) - 1)
    ann_vol      = float(returns.std()) * math.sqrt(_ANNUALIZATION)

    sharpe = (ann_return - _RF) / ann_vol if ann_vol > 1e-12 else float("nan")

    drawdown    = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar      = ann_return / abs(max_drawdown) if max_drawdown < -1e-9 else float("nan")

    beta, treynor, jensen = _compute_relative_metrics(
        equity, returns, ann_return, market_equity
    )

    return {
        "total_return": total_return,
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_drawdown,
        "calmar":       calmar,
        "beta":         beta,
        "treynor":      treynor,
        "jensen":       jensen,
    }


def _compute_relative_metrics(
    equity: pd.Series,
    returns: pd.Series,
    ann_return: float,
    market_equity: pd.Series,
) -> tuple[float, float, float]:
    """
    Compute beta, Treynor ratio, and Jensen's alpha vs. market_equity (D25: ew_bnh).

    Returns (beta, treynor, jensen). Any value that cannot be computed is NaN.
    """
    nan3 = (float("nan"), float("nan"), float("nan"))

    if market_equity.empty or len(market_equity) < 2:
        return nan3

    mkt_returns = market_equity.pct_change().dropna()
    aligned = pd.DataFrame({"p": returns, "m": mkt_returns}).dropna()

    if len(aligned) < 2:
        return nan3

    mkt_var = float(aligned["m"].var())
    if mkt_var < 1e-12:
        return nan3

    beta = float(aligned["p"].cov(aligned["m"])) / mkt_var
    treynor = (ann_return - _RF) / beta if abs(beta) > 1e-9 else float("nan")

    # Market annualized return over the same evaluation window as equity
    mkt_slice = market_equity.reindex(equity.index).ffill().dropna()
    if len(mkt_slice) >= 2:
        mkt_total = float(mkt_slice.iloc[-1] / mkt_slice.iloc[0] - 1)
        mkt_ann   = (1 + mkt_total) ** (_ANNUALIZATION / len(mkt_slice)) - 1
        jensen    = (ann_return - _RF) - beta * (mkt_ann - _RF)
    else:
        jensen = float("nan")

    return beta, treynor, jensen


def _is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
