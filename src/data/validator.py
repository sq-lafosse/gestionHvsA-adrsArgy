"""
Validator for the gestionHvsA-adrsArgy data pipeline.

Pure detection and reporting — never modifies data, never raises exceptions.
Logs a formatted coverage table at INFO level on every validate_* call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NOMINAL_START = pd.Timestamp("2015-01-01")
_WEEKLY_GAP_DAYS = 10
_MONTHLY_GAP_DAYS = 45
_OUTLIER_ZSCORE = 5.0


@dataclass
class ValidationReport:
    is_valid: bool                                    # False if any error is present
    errors: list[str] = field(default_factory=list)  # critical: 100% NaN series
    warnings: list[str] = field(default_factory=list) # non-critical: late start, interior gaps, outliers
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-asset stats
    gaps: dict[str, list[str]] = field(default_factory=dict)           # gap descriptions per asset
    outliers: dict[str, list[str]] = field(default_factory=dict)       # outlier dates per asset


# ─── Public API ───────────────────────────────────────────────────────────────

def validate_prices(
    adrs: pd.DataFrame,
    sovereign_bond: pd.Series,
    merval: pd.Series,
    ccl: pd.Series,
    gap_threshold_days: int = _WEEKLY_GAP_DAYS,
    outlier_zscore: float = _OUTLIER_ZSCORE,
    nominal_start: str | pd.Timestamp = _NOMINAL_START,
) -> ValidationReport:
    """
    Validate weekly price data: ADRs, sovereign bond, Merval, CCL.

    Detects gaps, NaN (late start vs interior), and return outliers.
    Logs a coverage table at INFO. Never modifies data.
    """
    nominal_ts = pd.Timestamp(nominal_start)
    errors: list[str] = []
    warnings: list[str] = []
    coverage: dict[str, dict] = {}
    gaps: dict[str, list[str]] = {}
    outliers: dict[str, list[str]] = {}

    for ticker in adrs.columns:
        cov, errs, warns, asset_gaps, asset_outliers = _validate_series(
            adrs[ticker], ticker, nominal_ts, gap_threshold_days, outlier_zscore
        )
        coverage[ticker] = cov
        errors.extend(errs)
        warnings.extend(warns)
        if asset_gaps:
            gaps[ticker] = asset_gaps
        if asset_outliers:
            outliers[ticker] = asset_outliers

    for name, series in [
        ("sovereign_bond", sovereign_bond),
        ("merval", merval),
        ("ccl", ccl),
    ]:
        cov, errs, warns, asset_gaps, asset_outliers = _validate_series(
            series, name, nominal_ts, gap_threshold_days, outlier_zscore
        )
        coverage[name] = cov
        errors.extend(errs)
        warnings.extend(warns)
        if asset_gaps:
            gaps[name] = asset_gaps
        if asset_outliers:
            outliers[name] = asset_outliers

    _log_coverage(coverage, section="prices")

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        coverage=coverage,
        gaps=gaps,
        outliers=outliers,
    )


def validate_macro(
    macro: pd.DataFrame,
    gap_threshold_days: int = _MONTHLY_GAP_DAYS,
    nominal_start: str | pd.Timestamp = _NOMINAL_START,
) -> ValidationReport:
    """
    Validate monthly macro DataFrame.

    Outlier detection is skipped — extreme values are real in the Argentine macro context.
    Logs a coverage table at INFO. Never modifies data.
    """
    nominal_ts = pd.Timestamp(nominal_start)
    errors: list[str] = []
    warnings: list[str] = []
    coverage: dict[str, dict] = {}
    gaps: dict[str, list[str]] = {}

    for col in macro.columns:
        cov, errs, warns, col_gaps, _ = _validate_series(
            macro[col], col, nominal_ts, gap_threshold_days, outlier_zscore=None
        )
        coverage[col] = cov
        errors.extend(errs)
        warnings.extend(warns)
        if col_gaps:
            gaps[col] = col_gaps

    _log_coverage(coverage, section="macro")

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        coverage=coverage,
        gaps=gaps,
        outliers={},
    )


def validate_all(
    adrs: pd.DataFrame,
    sovereign_bond: pd.Series,
    merval: pd.Series,
    ccl: pd.Series,
    macro: pd.DataFrame,
    weekly_gap_days: int = _WEEKLY_GAP_DAYS,
    monthly_gap_days: int = _MONTHLY_GAP_DAYS,
    outlier_zscore: float = _OUTLIER_ZSCORE,
    nominal_start: str | pd.Timestamp = _NOMINAL_START,
) -> ValidationReport:
    """Validate all cache components. Merges results from validate_prices and validate_macro."""
    prices_report = validate_prices(
        adrs, sovereign_bond, merval, ccl,
        gap_threshold_days=weekly_gap_days,
        outlier_zscore=outlier_zscore,
        nominal_start=nominal_start,
    )
    macro_report = validate_macro(
        macro,
        gap_threshold_days=monthly_gap_days,
        nominal_start=nominal_start,
    )
    return ValidationReport(
        is_valid=prices_report.is_valid and macro_report.is_valid,
        errors=prices_report.errors + macro_report.errors,
        warnings=prices_report.warnings + macro_report.warnings,
        coverage={**prices_report.coverage, **macro_report.coverage},
        gaps={**prices_report.gaps, **macro_report.gaps},
        outliers=prices_report.outliers,
    )


# ─── Core validation ──────────────────────────────────────────────────────────

def _validate_series(
    series: pd.Series,
    name: str,
    nominal_start: pd.Timestamp,
    gap_threshold_days: int,
    outlier_zscore: float | None,
) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """
    Validate a single Series. Returns (coverage, errors, warnings, gaps, outliers).
    outlier_zscore=None skips outlier detection.
    """
    errors: list[str] = []
    warnings: list[str] = []
    gap_list: list[str] = []
    outlier_list: list[str] = []

    total_rows = len(series)
    nan_count = int(series.isna().sum())

    if nan_count == total_rows:
        errors.append(f"[{name}] series is entirely NaN ({total_rows} rows)")
        return (
            {
                "first_date": None,
                "last_date": None,
                "total_rows": total_rows,
                "nan_count": nan_count,
                "nan_pct": 100.0,
                "late_start": True,
                "days_behind_nominal": None,
            },
            errors,
            warnings,
            gap_list,
            outlier_list,
        )

    valid = series.dropna()
    first_date = valid.index.min()
    last_date = valid.index.max()
    nan_pct = round(nan_count / total_rows * 100, 2) if total_rows > 0 else 0.0

    # Late start
    late_start = first_date > nominal_start
    days_behind = int((first_date - nominal_start).days) if late_start else 0
    if late_start:
        warnings.append(
            f"[{name}] late start: first valid date {first_date.date()} "
            f"is {days_behind} days after nominal start {nominal_start.date()}"
        )

    # Interior NaN (between first and last valid date, not counting leading NaN)
    interior = series.loc[first_date:last_date]
    interior_nan_mask = interior.isna()
    interior_nan_count = int(interior_nan_mask.sum())
    if interior_nan_count > 0:
        interior_nan_dates = interior.index[interior_nan_mask]
        warnings.append(
            f"[{name}] {interior_nan_count} interior NaN between "
            f"{first_date.date()} and {last_date.date()}"
        )
        for d in interior_nan_dates:
            warnings.append(f"[{name}]   interior NaN on {pd.Timestamp(d).date()}")

    # Gaps in the valid (non-NaN) observations
    valid_sorted = valid.index.sort_values()
    for i in range(1, len(valid_sorted)):
        delta = valid_sorted[i] - valid_sorted[i - 1]
        if delta.days > gap_threshold_days:
            gap_str = (
                f"{valid_sorted[i - 1].date()} -> {valid_sorted[i].date()} "
                f"({delta.days}d)"
            )
            gap_list.append(gap_str)
            warnings.append(f"[{name}] gap: {gap_str}")

    # Outliers on log returns (skipped for macro)
    if outlier_zscore is not None and len(valid) > 2:
        log_returns = np.log(valid / valid.shift(1)).dropna()
        std = log_returns.std()
        if std > 0:
            zscores = (log_returns - log_returns.mean()) / std
            for dt, z in zscores[zscores.abs() > outlier_zscore].items():
                entry = f"{pd.Timestamp(dt).date()} (z={z:.2f})"
                outlier_list.append(entry)
                warnings.append(f"[{name}] outlier return: {entry}")

    coverage = {
        "first_date": str(first_date.date()),
        "last_date": str(last_date.date()),
        "total_rows": total_rows,
        "nan_count": nan_count,
        "nan_pct": nan_pct,
        "late_start": late_start,
        "days_behind_nominal": days_behind,
    }
    return coverage, errors, warnings, gap_list, outlier_list


# ─── Coverage log ─────────────────────────────────────────────────────────────

def _log_coverage(coverage: dict[str, dict], section: str) -> None:
    """Log a formatted coverage table at INFO level."""
    col_w = 20
    header = (
        f"{'asset':<{col_w}}  {'first_date':<12}  {'last_date':<12}"
        f"  {'rows':>5}  {'nan':>5}  {'nan%':>6}  {'status':<10}"
    )
    sep = "-" * len(header)
    logger.info("Coverage report [%s]", section)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for asset, s in coverage.items():
        if s["first_date"] is None:
            logger.error(
                "%-*s  %-12s  %-12s  %5s  %5s  %6s  %-10s",
                col_w, asset, "MISSING", "MISSING",
                s["total_rows"], s["nan_count"], "100.00%", "ERROR",
            )
        else:
            status = f"LATE +{s['days_behind_nominal']}d" if s["late_start"] else "OK"
            logger.info(
                "%-*s  %-12s  %-12s  %5d  %5d  %5.2f%%  %-10s",
                col_w, asset,
                s["first_date"], s["last_date"],
                s["total_rows"], s["nan_count"], s["nan_pct"],
                status,
            )
    logger.info(sep)
