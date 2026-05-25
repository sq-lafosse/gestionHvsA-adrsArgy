"""
Allocation engine for gestionHvsA-adrsArgy.

Computes monthly portfolio weights using a 5-variable dynamic formula
loaded from config/settings.yaml (allocation_weights section):

    peso_i = signal_weight      × price_to_sma30_i      (normalizado)
           + vol_weight         × (1 / realized_vol_i)   (normalizado)
           + corr_weight        × (1 - avg_corr_i)       (normalizado)
           + nlp_company_weight × nlp_empresa_i
           + nlp_macro_weight   × nlp_macro

Inclusion rule (applied first):
    Risk-On  → all ADRs with price_to_sma30 > 1.0 are candidates
    Risk-Off → top 5 ADRs by composite score

Edge case — no ADR passes the inclusion filter:
    Falls back to equal weight across the full universe.
    No fixed income (D8: portfolio 100 % equity).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_RISK_ON  = "risk_on"
_RISK_OFF = "risk_off"
_INCLUSION_SIGNAL    = "price_to_sma30"
_INCLUSION_THRESHOLD = 1.0
_RISK_OFF_TOP_N      = 5
_SETTINGS_PATH       = Path("config/settings.yaml")

_DEFAULT_WEIGHTS = {
    "signal_weight":      0.40,
    "vol_weight":         0.30,
    "corr_weight":        0.10,
    "nlp_company_weight": 0.10,
    "nlp_macro_weight":   0.10,
}


# ─── Public types ─────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    weights: dict[str, float]   # ticker → weight in [0.0, 1.0], sums to 1.0
    active_adrs: list[str]      # ADRs selected this month (passed inclusion filter)
    excluded_adrs: list[str]    # ADRs excluded this month
    regime: str                 # "risk_on" or "risk_off"
    probability: float          # SVM posterior from predict_regime()


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_weights(
    regime_result: dict[str, Any],
    assets_features: pd.DataFrame,
    nlp_result: dict[str, Any] | None = None,
) -> AllocationResult:
    """
    Compute monthly portfolio weights using the 5-variable dynamic formula.

    Parameters
    ----------
    regime_result : dict
        Output of predict_regime(): {"regime": str, "probability": float}
    assets_features : pd.DataFrame
        FeatureMatrix.assets — weekly prices, MultiIndex columns (ticker, signal_name).
        Required signals at level 1: price_to_sma30, realized_vol_12.
        Required level 0: price (adjusted close) for correlation computation.
    nlp_result : dict | None
        Output of compute_monthly_sentiment():
        {"macro_score": float, "company_scores": {ticker: float}}
        Pass None to use 0.0 for both NLP components.

    Returns
    -------
    AllocationResult with weights summing to 1.0.
    """
    regime     = regime_result["regime"]
    probability = float(regime_result["probability"])

    formula_weights = _load_formula_weights()

    all_adrs    = _get_all_adrs(assets_features)
    sma30_series = _extract_signal(assets_features, _INCLUSION_SIGNAL)
    vol_series   = _extract_signal(assets_features, "realized_vol_12")
    price_series = _extract_prices(assets_features)

    # Step 1 — inclusion filter
    positive_adrs = sorted(
        t for t in sma30_series.index if sma30_series[t] > _INCLUSION_THRESHOLD
    )

    if regime == _RISK_OFF:
        candidates = _select_top_n(
            positive_adrs, sma30_series, vol_series, price_series,
            nlp_result, formula_weights, _RISK_OFF_TOP_N,
        )
    else:
        candidates = positive_adrs

    if not candidates:
        logger.warning(
            "compute_weights (%s): no candidates — falling back to EW across "
            "full universe (%d tickers).",
            regime, len(all_adrs),
        )
        candidates = all_adrs

    active_adrs   = sorted(candidates)
    excluded_adrs = sorted(t for t in all_adrs if t not in active_adrs)

    # Step 2 — dynamic formula
    raw_scores = _compute_scores(
        active_adrs, sma30_series, vol_series, price_series,
        nlp_result, formula_weights,
    )

    weights = _normalize_scores(raw_scores, all_adrs)

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
    _log_allocation(result, sma30_series, raw_scores)
    return result


# ─── Formula components ───────────────────────────────────────────────────────

def _compute_scores(
    active_adrs: list[str],
    sma30_series: pd.Series,
    vol_series: pd.Series,
    price_series: pd.DataFrame,
    nlp_result: dict[str, Any] | None,
    fw: dict[str, float],
) -> dict[str, float]:
    """
    Compute raw composite score for each active ADR using the 5-variable formula.

    All sub-components are min-max normalized to [0, 1] across active_adrs
    before applying the formula weights, so each dimension contributes
    proportionally regardless of its raw scale.
    """
    if not active_adrs:
        return {}

    signal_raw  = pd.Series({t: sma30_series.get(t, 1.0)   for t in active_adrs})
    vol_inv_raw = pd.Series({t: 1.0 / max(vol_series.get(t, 1.0), 1e-9)
                              for t in active_adrs})
    corr_inv_raw = _compute_correlation_inverse(active_adrs, price_series)

    macro_score   = float(nlp_result["macro_score"])   if nlp_result else 0.0
    company_scores = nlp_result.get("company_scores", {}) if nlp_result else {}

    # Normalize each dimension independently to [0, 1]
    signal_norm   = _minmax(signal_raw)
    vol_norm      = _minmax(vol_inv_raw)
    corr_norm     = _minmax(corr_inv_raw)

    # NLP scores are already in [-1, 1] → shift to [0, 1]
    macro_norm = (macro_score + 1.0) / 2.0

    scores: dict[str, float] = {}
    for ticker in active_adrs:
        nlp_co = company_scores.get(ticker, 0.0)
        nlp_co_norm = (nlp_co + 1.0) / 2.0

        scores[ticker] = (
            fw["signal_weight"]      * signal_norm.get(ticker, 0.5)
            + fw["vol_weight"]       * vol_norm.get(ticker, 0.5)
            + fw["corr_weight"]      * corr_norm.get(ticker, 0.5)
            + fw["nlp_company_weight"] * nlp_co_norm
            + fw["nlp_macro_weight"] * macro_norm
        )

    return scores


def _select_top_n(
    positive_adrs: list[str],
    sma30_series: pd.Series,
    vol_series: pd.Series,
    price_series: pd.DataFrame,
    nlp_result: dict[str, Any] | None,
    fw: dict[str, float],
    n: int,
) -> list[str]:
    """
    For Risk-Off: restrict the candidate set to the top-n ADRs by composite score.

    If fewer than n ADRs have positive signal, returns all positive ADRs.
    """
    if len(positive_adrs) <= n:
        return positive_adrs

    scores = _compute_scores(
        positive_adrs, sma30_series, vol_series, price_series, nlp_result, fw
    )
    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
    return ranked[:n]


def _compute_correlation_inverse(
    active_adrs: list[str],
    price_series: pd.DataFrame,
) -> pd.Series:
    """
    Compute correlation_inverse_i = 1 - avg(|corr(i, j)|) for j ≠ i
    over the last 12 weeks of price data.

    A ticker that moves independently of the rest receives a higher value
    (better diversification contribution).
    Returns a Series with a value for each active ADR.
    Falls back to 0.5 for any ticker with insufficient price history.
    """
    if len(active_adrs) == 1:
        return pd.Series({active_adrs[0]: 1.0})

    available = [t for t in active_adrs if t in price_series.columns]
    if not available:
        return pd.Series({t: 0.5 for t in active_adrs})

    prices_12w = price_series[available].iloc[-12:]
    if len(prices_12w) < 2:
        return pd.Series({t: 0.5 for t in active_adrs})

    corr_matrix = prices_12w.pct_change().dropna().corr().abs()

    result: dict[str, float] = {}
    for ticker in active_adrs:
        if ticker not in corr_matrix.columns:
            result[ticker] = 0.5
            continue
        others = [t for t in corr_matrix.columns if t != ticker]
        if not others:
            result[ticker] = 1.0
        else:
            avg_corr = float(corr_matrix.loc[ticker, others].mean())
            result[ticker] = 1.0 - avg_corr

    return pd.Series(result)


# ─── Normalization ────────────────────────────────────────────────────────────

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Returns 0.5 for all-equal series."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _normalize_scores(
    raw_scores: dict[str, float],
    all_adrs: list[str],
) -> dict[str, float]:
    """
    Normalize raw composite scores to portfolio weights summing to 1.0.

    Non-active tickers receive 0.0 weight.
    """
    weights: dict[str, float] = {t: 0.0 for t in all_adrs}
    total = sum(raw_scores.values())

    if total < 1e-12:
        n = len(raw_scores)
        for ticker in raw_scores:
            weights[ticker] = 1.0 / n if n > 0 else 0.0
        return weights

    for ticker, score in raw_scores.items():
        weights[ticker] = score / total

    return weights


# ─── Data extraction ──────────────────────────────────────────────────────────

def _get_all_adrs(assets_features: pd.DataFrame) -> list[str]:
    if assets_features.empty:
        raise ValueError("assets_features is empty.")
    if not isinstance(assets_features.columns, pd.MultiIndex):
        raise ValueError(
            "assets_features must have MultiIndex columns (ticker, signal_name)."
        )
    return sorted(assets_features.columns.get_level_values(0).unique().tolist())


def _extract_signal(assets_features: pd.DataFrame, signal: str) -> pd.Series:
    """
    Extract the last monthly value of a named signal for each ticker.

    Resamples weekly → monthly (last obs), returns the most recent month's row.
    Tickers with NaN are dropped.
    """
    try:
        weekly = assets_features.xs(signal, axis=1, level=1)
    except KeyError:
        logger.warning("Signal '%s' not found in assets_features — using empty Series", signal)
        return pd.Series(dtype=float)

    monthly = weekly.resample("ME").last()
    if monthly.empty:
        return pd.Series(dtype=float)

    return monthly.iloc[-1].dropna()


def _extract_prices(assets_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the weekly price_to_sma30 series (full history) as a price proxy
    for the 12-week return correlation computation.

    pct_change() on price_to_sma30 approximates actual returns over short windows
    because the SMA changes slowly relative to price. Falls back to empty DataFrame
    if the signal is not available.
    """
    try:
        return assets_features.xs(_INCLUSION_SIGNAL, axis=1, level=1)
    except KeyError:
        logger.warning("'%s' not found — correlation component set to 0.5", _INCLUSION_SIGNAL)
        return pd.DataFrame()


# ─── Config loading ───────────────────────────────────────────────────────────

def _load_formula_weights() -> dict[str, float]:
    """Load allocation_weights from settings.yaml; fall back to defaults."""
    try:
        with open(_SETTINGS_PATH) as f:
            cfg = yaml.safe_load(f)
        w = cfg.get("allocation_weights", {})
        merged = {**_DEFAULT_WEIGHTS, **w}
        total = sum(merged.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "allocation_weights sum=%.4f ≠ 1.0 in %s — using defaults",
                total, _SETTINGS_PATH,
            )
            return _DEFAULT_WEIGHTS
        return merged
    except Exception as exc:
        logger.warning("Could not load %s: %s — using defaults", _SETTINGS_PATH, exc)
        return _DEFAULT_WEIGHTS


# ─── Logging ─────────────────────────────────────────────────────────────────

def _log_allocation(
    result: AllocationResult,
    sma30_series: pd.Series,
    raw_scores: dict[str, float],
) -> None:
    active_detail = [
        f"{t}(sma={sma30_series.get(t, float('nan')):.3f},score={raw_scores.get(t, 0):.4f})"
        for t in result.active_adrs
    ]
    excluded_detail = [
        f"{t}={sma30_series[t]:.3f}" if t in sma30_series.index else f"{t}=no-data"
        for t in result.excluded_adrs
    ]
    nonzero = {t: w for t, w in result.weights.items() if w > 0.0}
    weight_str = " ".join(f"{t}={w:.4f}" for t, w in nonzero.items())

    logger.info(
        "compute_weights: %s (p=%.3f) | active=[%s] | excluded=[%s] | %s",
        result.regime,
        result.probability,
        ", ".join(active_detail) if active_detail else "none",
        ", ".join(excluded_detail) if excluded_detail else "none",
        weight_str,
    )
