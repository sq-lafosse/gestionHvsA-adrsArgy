"""
src/nlp — FinBERT monthly sentiment scoring for gestionHvsA-adrsArgy.

Public API:
    compute_monthly_sentiment(year, month, base_dir, _pipeline) → float
        Aggregated sentiment score in [-1, 1] for the given month's news corpus.
        Returns 0.0 (neutral) if the directory is absent or empty.
        Pass the result directly as nlp_score to predict_regime().
"""
from .nlp import compute_monthly_sentiment

__all__ = ["compute_monthly_sentiment"]
