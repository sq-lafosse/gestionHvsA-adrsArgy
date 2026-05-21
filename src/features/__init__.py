"""
src/features — technical signal computation for the ADR universe.

Entry point:
  compute_features(adrs, sma_short, sma_long, momentum_window, vol_window)
  → FeatureMatrix(portfolio, assets, log_returns_portfolio, log_returns_assets)
"""
from .features import FeatureMatrix, compute_features

__all__ = ["FeatureMatrix", "compute_features"]
