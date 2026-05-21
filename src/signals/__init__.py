"""
src/signals — regime classification for gestionHvsA-adrsArgy.

Public API:
    train_regime_classifier  — fit Pipeline(StandardScaler → PCA → SVC) on in-sample data
                               and serialize to data/cache/models/regime_pipeline.joblib
    predict_regime           — load pipeline, predict current month's regime
                               returns {"regime": "risk_on" | "risk_off", "probability": float}
    FEATURE_COLS             — ordered list of 8 feature column names (fixed contract)
"""
from .regime import FEATURE_COLS, predict_regime, train_regime_classifier

__all__ = ["train_regime_classifier", "predict_regime", "FEATURE_COLS"]
