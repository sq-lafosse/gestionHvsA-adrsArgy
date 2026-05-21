"""
Loader — orchestrator for src/data/.

Responsibilities:
- Historical mode: download full 2015→2023 dataset, validate, persist (runs once)
- Live mode: download one month up to cutoff_date, validate, append to cache
- Primary guard against re-writing an immutable historical cache
- Anti-leakage: live mode clips every download to <= cutoff_date
"""
from __future__ import annotations

import logging

import pandas as pd

from . import cache_manager, downloader, scraper_al30d, scraper_ccl, scraper_macro, validator
from .cache_manager import CachePayload

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_historical(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2023-12-31",
) -> None:
    """
    Download and persist the full historical dataset (2015→2023). Runs once only.

    If the historical cache already exists, logs a warning and returns immediately
    without touching any data (primary guard; secondary guard is in cache_manager).
    Aborts without writing if validation finds any entirely-NaN series.
    """
    if cache_manager.historical_cache_exists():
        logger.warning(
            "Historical cache already exists — skipping download. "
            "Delete data/cache/.cache_manifest.json manually to force a re-download."
        )
        return

    logger.info(
        "Starting historical download: %s → %s | tickers: %s",
        start, end, tickers,
    )

    adrs = downloader.download_adrs(tickers, start, end)
    merval = downloader.download_merval(start, end)

    ggal_adr: pd.Series | None = adrs["GGAL"] if "GGAL" in adrs.columns else None
    if ggal_adr is None:
        logger.warning("GGAL not in ADR download — CCL GGAL-ratio fallback unavailable")

    sovereign_bond = scraper_al30d.get_sovereign_bond(start, end)
    ccl = scraper_ccl.get_ccl(start, end, ggal_adr=ggal_adr)
    macro = scraper_macro.get_all_macro(start, end)

    report = validator.validate_all(adrs, sovereign_bond, merval, ccl, macro)
    if not report.is_valid:
        logger.error(
            "Validation failed — historical cache NOT written. %d error(s):",
            len(report.errors),
        )
        for err in report.errors:
            logger.error("  %s", err)
        return

    if report.warnings:
        logger.warning(
            "%d validation warning(s) — cache will still be written. "
            "See coverage report above for details.",
            len(report.warnings),
        )

    cache_manager.save_historical_cache(
        CachePayload(
            adrs=adrs,
            sovereign_bond=sovereign_bond,
            merval=merval,
            ccl=ccl,
            macro=macro,
        )
    )
    logger.info("Historical cache written successfully.")


def run_live_month(
    tickers: list[str],
    cutoff_date: str | pd.Timestamp,
) -> None:
    """
    Download and append one month of live data up to cutoff_date.

    cutoff_date must be the last calendar date of the month being processed
    (e.g., "2024-01-31"). Every download is clipped to <= cutoff_date to
    prevent data leakage.

    Raises:
        RuntimeError: if the historical cache does not exist yet.
    """
    if not cache_manager.historical_cache_exists():
        raise RuntimeError(
            "Cannot run live mode: historical cache has not been written yet. "
            "Run run_historical() first."
        )

    cutoff_ts = pd.Timestamp(cutoff_date).normalize()
    last_cached = cache_manager.get_last_cached_date()

    start_ts = (
        last_cached + pd.Timedelta(days=1)
        if last_cached is not None
        else pd.Timestamp("2024-01-01")
    )
    start = start_ts.strftime("%Y-%m-%d")
    end = cutoff_ts.strftime("%Y-%m-%d")

    logger.info(
        "Starting live-month download: %s → %s (cutoff: %s) | tickers: %s",
        start, end, cutoff_ts.date(), tickers,
    )

    adrs = downloader.download_adrs(tickers, start, end)
    merval = downloader.download_merval(start, end)

    ggal_adr: pd.Series | None = adrs["GGAL"] if "GGAL" in adrs.columns else None
    if ggal_adr is None:
        logger.warning("GGAL not in live ADR download — CCL GGAL-ratio fallback unavailable")

    # Pass cached series so scrapers can fall back to last known value if live sources fail
    cached_sovereign = cache_manager.load_sovereign_bond_up_to(cutoff_ts)
    sovereign_bond = scraper_al30d.get_sovereign_bond(
        start, end, cached_series=cached_sovereign
    )

    cached_ccl = cache_manager.load_ccl_up_to(cutoff_ts)
    ccl = scraper_ccl.get_ccl(
        start, end, ggal_adr=ggal_adr, cached_series=cached_ccl
    )

    cached_macro = cache_manager.load_macro_up_to(cutoff_ts)
    macro = scraper_macro.get_all_macro(start, end, cached_df=cached_macro)

    report = validator.validate_all(adrs, sovereign_bond, merval, ccl, macro)
    if not report.is_valid:
        logger.error(
            "Live-month validation failed — data NOT appended. %d error(s):",
            len(report.errors),
        )
        for err in report.errors:
            logger.error("  %s", err)
        return

    if report.warnings:
        logger.warning(
            "%d validation warning(s) — data will still be appended.",
            len(report.warnings),
        )

    cache_manager.append_live_month(
        CachePayload(
            adrs=adrs,
            sovereign_bond=sovereign_bond,
            merval=merval,
            ccl=ccl,
            macro=macro,
        )
    )
    logger.info("Live month appended successfully. Cutoff: %s", cutoff_ts.date())
