"""
Matplotlib figures for gestionHvsA-adrsArgy academic paper.

D28: Two PNG/PDF figures per run:
    equity_curves.png — portfolio + 3 benchmarks, each normalized to 1.0 at start.
                        Vertical dashed lines mark rebalance execution dates.
    drawdown.png      — portfolio drawdown as filled area; max drawdown annotated.

Academic style: clean axes, no chartjunk, DPI 300 for publication inclusion.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.backtest import BacktestResult

logger = logging.getLogger(__name__)

_PORTFOLIO_COLOR = "#1a2e6b"
_BENCHMARK_COLORS = {
    "ew_bnh":       "#e07b26",
    "al30d_static": "#2a9d8f",
    "merval_usd":   "#c1121f",
}
_BENCHMARK_LABELS = {
    "ew_bnh":       "EW B&H ADRs",
    "al30d_static": "AL30D (static)",
    "merval_usd":   "Merval USD",
}


# ─── Public API ───────────────────────────────────────────────────────────────

def plot_equity_curves(result: BacktestResult, output_dir: Path | str) -> Path:
    """
    Plot portfolio equity curve vs. three benchmarks.

    Legend entries include total return. Vertical dashed lines show rebalance dates.
    Saves equity_curves.png (DPI 300) and equity_curves.pdf to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_academic_style(ax)

    for name, equity in result.benchmarks.items():
        if equity.empty:
            continue
        total = equity.iloc[-1] - 1
        ax.plot(
            equity.index, equity.values,
            label=f"{_BENCHMARK_LABELS.get(name, name)} ({total:+.1%})",
            color=_BENCHMARK_COLORS.get(name, "#888888"),
            linewidth=1.4,
            alpha=0.85,
            zorder=2,
        )

    total_port = result.portfolio.iloc[-1] - 1
    ax.plot(
        result.portfolio.index, result.portfolio.values,
        label=f"Portfolio ({total_port:+.1%})",
        color=_PORTFOLIO_COLOR,
        linewidth=2.2,
        zorder=3,
    )

    for date in result.rebalance_dates:
        ax.axvline(date, color="#aaaaaa", linewidth=0.6, linestyle="--", alpha=0.6, zorder=1)

    ax.axhline(1.0, color="#cccccc", linewidth=0.5, zorder=0)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalized Value (start = 1.0)", fontsize=11)
    ax.set_title("Portfolio vs. Benchmarks — Equity Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    fig.tight_layout()
    path = output_dir / "equity_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s (+ .pdf)", path)
    return path


def plot_drawdown(result: BacktestResult, output_dir: Path | str) -> Path:
    """
    Plot portfolio drawdown as a filled area below zero.

    The maximum drawdown date and magnitude are annotated on the chart.
    Saves drawdown.png (DPI 300) and drawdown.pdf to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equity   = result.portfolio.dropna()
    drawdown = equity / equity.cummax() - 1
    max_dd   = float(drawdown.min())
    max_dd_date = drawdown.idxmin()

    fig, ax = plt.subplots(figsize=(12, 4))
    _apply_academic_style(ax)

    ax.fill_between(
        drawdown.index, drawdown.values, 0,
        color=_PORTFOLIO_COLOR, alpha=0.55,
        label=f"Drawdown (max: {max_dd:.1%})",
    )
    ax.plot(drawdown.index, drawdown.values, color=_PORTFOLIO_COLOR, linewidth=1.0)
    ax.axhline(0.0, color="#333333", linewidth=0.7)

    if max_dd < -1e-4:
        ax.scatter([max_dd_date], [max_dd], color="#c1121f", s=25, zorder=4)
        ax.annotate(
            f"max dd: {max_dd:.1%}",
            xy=(max_dd_date, max_dd),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=9,
            color="#c1121f",
            ha="center",
        )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.set_title("Portfolio Drawdown", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    path = output_dir / "drawdown.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s (+ .pdf)", path)
    return path


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _apply_academic_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", labelsize=9, colors="#444444")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
