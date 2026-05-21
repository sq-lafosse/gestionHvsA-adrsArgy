"""
src/metrics — performance metrics and visualizations for gestionHvsA-adrsArgy.

Public API:
    compute_metrics     — Sharpe, max drawdown, Treynor, Jensen, Calmar, total/ann return
    save_metrics_csv    — export MetricsResult.summary_df as performance_metrics.csv
    MetricsResult       — dataclass: portfolio dict, benchmarks dict, summary_df
    plot_equity_curves  — equity curves PNG/PDF (portfolio + 3 benchmarks)
    plot_drawdown       — drawdown chart PNG/PDF with max drawdown annotation
"""
from .metrics import MetricsResult, compute_metrics, save_metrics_csv
from .visualizer import plot_drawdown, plot_equity_curves

__all__ = [
    "compute_metrics",
    "save_metrics_csv",
    "MetricsResult",
    "plot_equity_curves",
    "plot_drawdown",
]
