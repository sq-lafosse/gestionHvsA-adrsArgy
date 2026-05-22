#!/usr/bin/env python3
"""
main.py — Monthly Tactical Asset Allocation pipeline.

Modes (--mode):
  historical : download & cache 2015-2023 data, train PCA+SVM regime classifier.
               Runs once — subsequent calls are no-ops (cache guard).
  live       : walk-forward month-by-month from live_start.
               --month YYYY-MM sets the last month to process (default: current month).

Configuration is read from config/ (assets.yaml, periods.yaml, settings.yaml).
Outputs are written to results/{year}/ for each year in the live period.
"""
from __future__ import annotations

import argparse
import logging
import sys
from calendar import monthrange
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.allocation import AllocationResult, compute_weights
from src.backtest import BacktestResult, run_backtest
from src.data import (
    load_adrs_up_to,
    load_ccl_up_to,
    load_macro_up_to,
    load_merval_up_to,
    load_sovereign_bond_up_to,
    run_historical,
    run_live_month,
)
from src.features import compute_features
from src.metrics import compute_metrics, plot_drawdown, plot_equity_curves, save_metrics_csv
from src.nlp import compute_monthly_sentiment
from src.signals import predict_regime, train_regime_classifier
from src.utils import ensure_dir, save_snapshot, setup_logging, timer

logger = logging.getLogger(__name__)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monthly Tactical Asset Allocation — gestionHvsA-adrsArgy"
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live"],
        required=True,
        help="historical: download + train.  live: walk-forward pipeline.",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        metavar="YYYY-MM",
        help="Last month to process in live mode (default: current month).",
    )
    return parser.parse_args()


# ─── Config ───────────────────────────────────────────────────────────────────

def _load_config() -> tuple[dict, dict, dict]:
    """Load assets.yaml, periods.yaml, settings.yaml from config/."""
    config_dir = Path("config")
    with (config_dir / "assets.yaml").open(encoding="utf-8") as f:
        assets = yaml.safe_load(f)
    with (config_dir / "periods.yaml").open(encoding="utf-8") as f:
        periods = yaml.safe_load(f)
    with (config_dir / "settings.yaml").open(encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return assets, periods, settings


# ─── Month utilities ──────────────────────────────────────────────────────────

def _month_end(year: int, month: int) -> pd.Timestamp:
    last_day = monthrange(year, month)[1]
    return pd.Timestamp(f"{year}-{month:02d}-{last_day}")


def _live_months(live_start: str, month_override: str | None) -> list[pd.Timestamp]:
    """Return list of month-start Timestamps from live_start to live_end (inclusive)."""
    start = pd.Timestamp(live_start)
    if month_override:
        end = pd.Timestamp(month_override)
    else:
        now = datetime.now()
        end = pd.Timestamp(f"{now.year}-{now.month:02d}-01")
    return list(pd.date_range(start=start, end=end, freq="MS"))


# ─── Historical mode ──────────────────────────────────────────────────────────

def run_mode_historical(assets: dict, periods: dict, settings: dict) -> None:
    tickers: list[str] = assets["adrs"]
    in_sample_end = periods["in_sample_end"]
    model_path = Path(settings["model_path"])

    start_str = pd.Timestamp(periods["in_sample_start"]).strftime("%Y-%m-%d")
    end_str   = pd.Timestamp(in_sample_end).strftime("%Y-%m-%d")

    with timer("historical data download"):
        run_historical(tickers, start=start_str, end=end_str)

    logger.info("Loading in-sample data for model training...")
    cutoff = pd.Timestamp(in_sample_end) + pd.offsets.MonthEnd(0)
    adrs  = load_adrs_up_to(cutoff)
    ccl   = load_ccl_up_to(cutoff)
    macro = load_macro_up_to(cutoff)

    if adrs.empty:
        logger.error("ADR cache is empty after historical download — aborting.")
        sys.exit(1)

    with timer("feature computation (in-sample)"):
        features = compute_features(
            adrs,
            sma_short=settings["sma_short"],
            sma_long=settings["sma_long"],
            momentum_window=settings.get("momentum_window", 12),
            vol_window=settings.get("vol_window", 12),
        )

    with timer("regime classifier training"):
        meta = train_regime_classifier(
            portfolio_features=features.portfolio,
            macro=macro,
            ccl=ccl,
            model_path=model_path,
        )

    logger.info(
        "Historical mode complete | model=%s | %d samples | %d PCA components (%.1f%% var)",
        model_path,
        meta["n_train_samples"],
        meta["n_components_selected"],
        meta.get("cumulative_variance_explained", 0.0) * 100,
    )


# ─── Live mode ────────────────────────────────────────────────────────────────

def run_mode_live(
    assets: dict,
    periods: dict,
    settings: dict,
    month_override: str | None,
) -> None:
    tickers: list[str]   = assets["adrs"]
    news_base_dir: str   = settings["news_base_dir"]
    model_path           = Path(settings["model_path"])
    results_base         = Path(settings["results_base_dir"])
    sma_short: int       = settings["sma_short"]
    sma_long: int        = settings["sma_long"]
    momentum_window: int = settings.get("momentum_window", 12)
    vol_window: int      = settings.get("vol_window", 12)

    months = _live_months(periods["live_start"], month_override)
    if not months:
        logger.error("No months to process — check live_start and --month argument.")
        sys.exit(1)

    logger.info(
        "Live mode: %d months | %s → %s",
        len(months),
        months[0].strftime("%Y-%m"),
        months[-1].strftime("%Y-%m"),
    )

    monthly_allocations: dict[pd.Timestamp, AllocationResult] = {}
    last_allocation: AllocationResult | None = None
    successful_months = 0

    for month_start in months:
        year  = month_start.year
        month = month_start.month
        me    = _month_end(year, month)

        try:
            with timer(f"live {year}-{month:02d}"):
                _process_live_month(
                    year=year, month=month, month_end=me,
                    tickers=tickers,
                    news_base_dir=news_base_dir,
                    model_path=model_path,
                    sma_short=sma_short, sma_long=sma_long,
                    momentum_window=momentum_window, vol_window=vol_window,
                    monthly_allocations=monthly_allocations,
                )
            last_allocation = monthly_allocations[me]
            successful_months += 1

        except Exception as exc:
            logger.error(
                "Month %d-%02d failed (%s: %s) — applying carry-forward.",
                year, month, type(exc).__name__, exc,
            )
            if last_allocation is not None:
                monthly_allocations[me] = last_allocation
                logger.warning(
                    "Carried forward %s allocation (p=%.3f) to %d-%02d.",
                    last_allocation.regime, last_allocation.probability, year, month,
                )
            else:
                logger.error(
                    "No prior allocation to carry forward — month %d-%02d skipped entirely.",
                    year, month,
                )

    if successful_months == 0:
        raise RuntimeError(
            "Live pipeline: no month completed successfully. "
            "Check logs above for per-month errors."
        )

    logger.info(
        "Live loop complete: %d/%d months succeeded.", successful_months, len(months)
    )

    # ─── Backtest ───────────────────────────────────────────────────────────

    last_me = _month_end(months[-1].year, months[-1].month)

    with timer("backtest"):
        adrs_bt   = load_adrs_up_to(last_me)
        sov_bt    = load_sovereign_bond_up_to(last_me)
        merval_bt = load_merval_up_to(last_me)
        ccl_bt    = load_ccl_up_to(last_me)

        result = run_backtest(
            monthly_allocations=monthly_allocations,
            adrs=adrs_bt,
            sovereign_bond=sov_bt,
            merval=merval_bt,
            ccl=ccl_bt,
        )

    # ─── Per-year outputs ────────────────────────────────────────────────────

    years_in_period = sorted({m.year for m in months})

    for year in years_in_period:
        _save_year_outputs(
            year=year,
            full_result=result,
            results_base=results_base,
            settings=settings,
            periods=periods,
            assets=assets,
            monthly_allocations=monthly_allocations,
        )

    logger.info("Live mode complete. All outputs written to %s/", results_base)


def _process_live_month(
    year: int,
    month: int,
    month_end: pd.Timestamp,
    tickers: list[str],
    news_base_dir: str,
    model_path: Path,
    sma_short: int,
    sma_long: int,
    momentum_window: int,
    vol_window: int,
    monthly_allocations: dict[pd.Timestamp, AllocationResult],
) -> None:
    """
    Run the full pipeline for one live month and store the AllocationResult.

    Anti-leakage: all data loads are capped at month_end so the model sees
    only information that would have been available on that calendar date.
    """
    run_live_month(tickers, cutoff_date=month_end)

    adrs  = load_adrs_up_to(month_end)
    ccl   = load_ccl_up_to(month_end)
    macro = load_macro_up_to(month_end)

    if adrs.empty:
        raise ValueError(f"ADR data empty for cutoff {month_end.date()}")

    features = compute_features(
        adrs,
        sma_short=sma_short,
        sma_long=sma_long,
        momentum_window=momentum_window,
        vol_window=vol_window,
    )

    nlp_score = compute_monthly_sentiment(year, month, base_dir=news_base_dir)

    regime_result = predict_regime(
        portfolio_features=features.portfolio,
        macro=macro,
        ccl=ccl,
        model_path=model_path,
        nlp_score=nlp_score,
    )

    allocation = compute_weights(
        regime_result=regime_result,
        assets_features=features.assets,
    )

    monthly_allocations[month_end] = allocation

    logger.info(
        "%d-%02d | %s (p=%.3f) | nlp=%.4f | active=%d excluded=%d",
        year, month,
        allocation.regime, allocation.probability,
        nlp_score,
        len(allocation.active_adrs), len(allocation.excluded_adrs),
    )


def _save_year_outputs(
    year: int,
    full_result: BacktestResult,
    results_base: Path,
    settings: dict,
    periods: dict,
    assets: dict,
    monthly_allocations: dict[pd.Timestamp, AllocationResult],
) -> None:
    """
    Slice BacktestResult to one calendar year, compute metrics, and write all outputs.

    Per D40: portfolio and benchmarks are re-normalized to 1.0 at the start of the
    year slice so within-year comparisons are fair across periods.
    Rebalance dates are filtered to the year so equity_curves.png shows only the
    vertical lines relevant to that period.
    """
    year_mask      = full_result.portfolio.index.year == year
    portfolio_year = full_result.portfolio.loc[year_mask]

    if portfolio_year.empty:
        logger.warning("No portfolio data for year %d — skipping outputs.", year)
        return

    portfolio_year = portfolio_year / portfolio_year.iloc[0]

    bench_year: dict[str, pd.Series] = {}
    for name, series in full_result.benchmarks.items():
        if series.empty:
            bench_year[name] = series
            continue
        s = series.loc[series.index.year == year]
        bench_year[name] = (s / s.iloc[0]) if not s.empty else s

    rebalance_year = [d for d in full_result.rebalance_dates if d.year == year]

    year_result = BacktestResult(
        portfolio=portfolio_year,
        benchmarks=bench_year,
        rebalance_dates=rebalance_year,
    )

    output_dir  = ensure_dir(results_base / str(year))
    figures_dir = ensure_dir(output_dir / "figures")

    with timer(f"metrics + outputs {year}"):
        metrics = compute_metrics(year_result)
        save_metrics_csv(metrics, output_dir)
        plot_equity_curves(year_result, figures_dir)
        plot_drawdown(year_result, figures_dir)
        _save_weights_csv(monthly_allocations, year, output_dir)

    n_months_year = sum(1 for ts in monthly_allocations if ts.year == year)
    save_snapshot(
        output_dir=output_dir,
        params={
            "sma_short":       settings["sma_short"],
            "sma_long":        settings["sma_long"],
            "momentum_window": settings.get("momentum_window", 12),
            "vol_window":      settings.get("vol_window", 12),
            "model_path":      settings["model_path"],
            "news_base_dir":   settings["news_base_dir"],
        },
        metadata={
            "year":            year,
            "in_sample_start": periods["in_sample_start"],
            "in_sample_end":   periods["in_sample_end"],
            "live_start":      periods["live_start"],
            "adrs":            assets["adrs"],
            "n_months":        n_months_year,
            "n_rebalances":    len(rebalance_year),
        },
    )

    logger.info(
        "Year %d outputs saved → %s | rebalances=%d | total_return=%.2f%%",
        year, output_dir, len(rebalance_year),
        metrics.portfolio.get("total_return", 0.0) * 100,
    )


def _save_weights_csv(
    monthly_allocations: dict[pd.Timestamp, AllocationResult],
    year: int,
    output_dir: Path,
) -> None:
    """Write portfolio_weights.csv for the given calendar year."""
    rows = []
    for ts, alloc in sorted(monthly_allocations.items()):
        if ts.year != year:
            continue
        row: dict = {"month": ts.strftime("%Y-%m"), "regime": alloc.regime}
        row.update(alloc.weights)
        rows.append(row)

    if not rows:
        logger.warning("No allocations for year %d — portfolio_weights.csv not written.", year)
        return

    df = pd.DataFrame(rows).set_index("month")
    path = output_dir / "portfolio_weights.csv"
    df.to_csv(path, float_format="%.6f")
    logger.info("portfolio_weights.csv → %s (%d rows)", path, len(df))


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    assets, periods, settings = _load_config()

    setup_logging(
        level=settings.get("log_level", "INFO"),
        log_dir=settings.get("log_dir", "logs"),
    )

    logger.info("=== gestionHvsA-adrsArgy | mode=%s ===", args.mode)

    if args.mode == "historical":
        run_mode_historical(assets, periods, settings)
    elif args.mode == "live":
        run_mode_live(assets, periods, settings, month_override=args.month)


if __name__ == "__main__":
    main()
