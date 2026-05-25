"""
src/nlp — FinBERT monthly sentiment scoring for gestionHvsA-adrsArgy.

Public API:
    compute_monthly_sentiment(year, month, base_dir, _pipeline)
        → {"macro_score": float, "company_scores": {ticker: float}}

    macro_score      : aggregated FinBERT score from BCRA/INDEC/FOMC documents [-1, 1]
    company_scores   : per-ticker FinBERT score from SEC 6-K/20-F filings [-1, 1]
    Returns 0.0 for any missing or empty document group.
"""
from .nlp import compute_monthly_sentiment

__all__ = ["compute_monthly_sentiment"]
