"""
Allocation engine for gestionHvsA-adrsArgy.

Translates a monthly regime decision and per-asset technical signals
into a portfolio weight vector.

Regime rules:
    Risk-On  → equal weight across ADRs with positive signal, AL30D = 0 %
    Risk-Off → 80 % equal weight across ADRs with positive signal + 20 % AL30D

Asset inclusion rule (applied every month, both regimes):
    price_to_sma30 > 1.0  →  ADR is active (positive signal)
    price_to_sma30 ≤ 1.0  →  ADR is excluded this month (can return next month)

Edge case — all ADRs excluded:
    Risk-On  + all excluded → 100 % AL30D  ("double-filter defensive")
    Risk-Off + all excluded → 100 % AL30D  (standard rule, no ADR weight)

Weights are fractions in [0.0, 1.0] that sum to exactly 1.0.
AL30D is always present as a key in the weights dict (0.0 in Risk-On unless defensive).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_AL30D = "AL30D"
_RISK_ON = "risk_on"
_RISK_OFF = "risk_off"
_INCLUSION_SIGNAL = "price_to_sma30"
_INCLUSION_THRESHOLD = 1.0
_RISK_OFF_ADR_FRACTION = 0.8
_RISK_OFF_AL30D_FRACTION = 0.2


# ─── Public types ─────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    weights: dict[str, float]   # ticker → weight in [0.0, 1.0], sums to 1.0
    active_adrs: list[str]      # ADRs with price_to_sma30 > 1.0 this month
    excluded_adrs: list[str]    # ADRs excluded this month (signal ≤ 1.0 or no data)
    regime: str                 # "risk_on" or "risk_off"
    probability: float          # SVM posterior from predict_regime()


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_weights(
    regime_result: dict[str, Any],
    assets_features: pd.DataFrame,
) -> AllocationResult:
    """
    Compute monthly portfolio weights from regime and per-asset signals.

    Parameters
    ----------
    regime_result : dict
        Output of predict_regime(): {"regime": "risk_on"|"risk_off", "probability": float}
    assets_features : pd.DataFrame
        FeatureMatrix.assets — weekly prices, MultiIndex columns (ticker, signal_name).
        Required signal: price_to_sma30 at MultiIndex level 1.

    Returns
    -------
    AllocationResult with weights summing to 1.0.
    """
    regime = regime_result["regime"]
    probability = float(regime_result["probability"])

    all_adrs = _get_all_adrs(assets_features)
    sma30_series = _extract_monthly_sma30(assets_features)

    active_adrs = sorted(t for t in sma30_series.index if sma30_series[t] > _INCLUSION_THRESHOLD)
    excluded_adrs = sorted(t for t in all_adrs if t not in active_adrs)

    weights = _apply_regime_rules(regime, active_adrs, all_adrs)

    assert abs(sum(weights.values()) - 1.0) < 1e-9, (
        f"Weight sum={sum(weights.values()):.10f} deviates from 1.0 — logic error."
    )

    result = AllocationResult(
        weights=weights,
        active_adrs=active_adrs,
        excluded_adrs=excluded_adrs,
        regime=regime,
        probability=probability,
    )

    _log_allocation(result, sma30_series)
    return result


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_all_adrs(assets_features: pd.DataFrame) -> list[str]:
    """Return sorted list of ADR tickers from the MultiIndex level 0."""
    if assets_features.empty:
        raise ValueError(
            "assets_features is empty. Pass FeatureMatrix.assets for the target month."
        )
    if not isinstance(assets_features.columns, pd.MultiIndex):
        raise ValueError(
            "assets_features must have MultiIndex columns (ticker, signal_name). "
            "Pass FeatureMatrix.assets directly — do not flatten the columns."
        )
    return sorted(assets_features.columns.get_level_values(0).unique().tolist())


def _extract_monthly_sma30(assets_features: pd.DataFrame) -> pd.Series:
    """
    Extract the last monthly price_to_sma30 value for each ticker.

    Resamples weekly signals to monthly (last observation of the month) and
    returns the most recent month's row. Tickers with NaN are dropped — they
    are treated as excluded (no data this month) by compute_weights().
    """
    try:
        sma30_weekly = assets_features.xs(_INCLUSION_SIGNAL, axis=1, level=1)
    except KeyError:
        raise ValueError(
            f"'{_INCLUSION_SIGNAL}' not found in assets_features at MultiIndex level 1. "
            "Ensure compute_features() was called with sma_long=30."
        )

    sma30_monthly = sma30_weekly.resample("ME").last()

    if sma30_monthly.empty:
        raise ValueError(
            "assets_features produced empty monthly data after resampling. "
            "Ensure the DataFrame covers at least one full calendar month."
        )

    last_row = sma30_monthly.iloc[-1].dropna()

    if last_row.empty:
        raise ValueError(
            "All tickers have NaN price_to_sma30 for the last available month. "
            "Check that assets_features covers the target period."
        )

    return last_row


def _apply_regime_rules(
    regime: str,
    active_adrs: list[str],
    all_adrs: list[str],
) -> dict[str, float]:
    """
    Build the weight dict according to regime rules.

    Returns a complete dict covering every ADR in all_adrs plus AL30D.
    Non-active tickers have weight 0.0.
    """
    weights: dict[str, float] = {ticker: 0.0 for ticker in all_adrs}
    weights[_AL30D] = 0.0

    if regime == _RISK_ON:
        if not active_adrs:
            weights[_AL30D] = 1.0
        else:
            adr_w = 1.0 / len(active_adrs)
            for ticker in active_adrs:
                weights[ticker] = adr_w

    elif regime == _RISK_OFF:
        if not active_adrs:
            weights[_AL30D] = 1.0
        else:
            adr_w = _RISK_OFF_ADR_FRACTION / len(active_adrs)
            for ticker in active_adrs:
                weights[ticker] = adr_w
            weights[_AL30D] = _RISK_OFF_AL30D_FRACTION

    else:
        raise ValueError(
            f"Unknown regime '{regime}'. Expected '{_RISK_ON}' or '{_RISK_OFF}'."
        )

    return weights


def _log_allocation(result: AllocationResult, sma30_series: pd.Series) -> None:
    """
    Emit one summary log line for the monthly allocation decision.

    Active ADRs are annotated with their price_to_sma30 value.
    Excluded ADRs show their signal value or 'no-data' if absent from sma30_series.
    """
    active_detail = [f"{t}={sma30_series[t]:.3f}" for t in result.active_adrs]

    excluded_detail = [
        f"{t}={sma30_series[t]:.3f}" if t in sma30_series.index else f"{t}=no-data"
        for t in result.excluded_adrs
    ]

    defensive = result.regime == _RISK_ON and not result.active_adrs
    mode = "double-filter defensive" if defensive else result.regime

    nonzero_weights = {t: w for t, w in result.weights.items() if w > 0.0}
    weight_summary = " ".join(f"{t}={w:.4f}" for t, w in nonzero_weights.items())

    logger.info(
        "compute_weights: %s (p=%.3f) | active=[%s] | excluded=[%s] | %s",
        mode,
        result.probability,
        ", ".join(active_detail) if active_detail else "none",
        ", ".join(excluded_detail) if excluded_detail else "none",
        weight_summary if weight_summary else f"{_AL30D}=1.0000",
    )
