"""
Regime classification for gestionHvsA-adrsArgy.

Classifies each month as Risk-On (1) or Risk-Off (0) using a sklearn Pipeline:
    StandardScaler → PCA (≥ 95 % variance explained) → SVC (RBF, probability=True)

Training: in-sample period 2015-01 → 2023-12 (caller must pass pre-filtered data).
Live use: load the serialized pipeline from data/cache/models/ — never re-fit.

Label rule (consistent across training and live prediction):
    portfolio price_to_sma30 > 1.0  →  Risk-On  (1)
    portfolio price_to_sma30 ≤ 1.0  →  Risk-Off (0)

Feature vector — 8 columns, fixed order:
    [0] price_to_sma20    weekly portfolio signal, resampled to month-end last
    [1] price_to_sma30    weekly portfolio signal, resampled to month-end last
    [2] momentum_12       weekly portfolio signal, resampled to month-end last
    [3] realized_vol_12   weekly portfolio signal, resampled to month-end last
    [4] embi              monthly — Argentina EMBI spread (bps)
    [5] ccl_variation     monthly — CCL month-over-month % change
    [6] reservas          monthly — BCRA international reserves (USD millions)
    [7] nlp_score         monthly — FinBERT sentiment score (0.0 until src/nlp/ built)

When src/nlp/ is integrated, callers pass the monthly FinBERT score via predict_regime();
the stored pipeline and this module's interface do not change.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(os.getenv("DATA_CACHE_PATH", "data/cache"))
_DEFAULT_MODEL_PATH = _CACHE_ROOT / "models" / "regime_pipeline.joblib"

FEATURE_COLS: list[str] = [
    "price_to_sma20",
    "price_to_sma30",
    "momentum_12",
    "realized_vol_12",
    "embi",
    "ccl_variation",
    "reservas",
    "nlp_score",
]

_RISK_ON = "risk_on"
_RISK_OFF = "risk_off"
_INT_TO_LABEL: dict[int, str] = {1: _RISK_ON, 0: _RISK_OFF}

_MIN_TRAIN_SAMPLES = 12


# ─── Public API ───────────────────────────────────────────────────────────────

def train_regime_classifier(
    portfolio_features: pd.DataFrame,
    macro: pd.DataFrame,
    ccl: pd.Series,
    model_path: Path | str = _DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """
    Train and persist the regime classification Pipeline on in-sample data.

    Callers must pass data pre-filtered to the in-sample period (2015-01 → 2023-12).
    Anti-leakage is the caller's responsibility — this function applies no date cutoff.

    Parameters
    ----------
    portfolio_features : pd.DataFrame
        Weekly portfolio-level signals from FeatureMatrix.portfolio.
        Required columns: price_to_sma20, price_to_sma30, momentum_12, realized_vol_12.
    macro : pd.DataFrame
        Monthly macro DataFrame from cache_manager.load_macro_up_to().
        Required columns: embi, reservas.
    ccl : pd.Series
        Weekly CCL (ARS/USD) series from cache_manager.load_ccl_up_to().
    model_path : Path | str
        Destination for the serialized Pipeline (.joblib) and metadata sidecar (.meta.json).

    Returns
    -------
    dict with keys: n_train_samples, n_components_selected, class_distribution,
                    explained_variance_ratio, cumulative_variance_explained,
                    feature_cols, model_path, trained_at.
    """
    model_path = Path(model_path)

    X = _build_monthly_feature_matrix(portfolio_features, macro, ccl)
    X = X.dropna()


    if len(X) < _MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Training requires at least {_MIN_TRAIN_SAMPLES} complete monthly observations "
            f"after dropping NaN rows. Got {len(X)}. "
            "Ensure inputs cover the full in-sample period (2015-01 → 2023-12)."
        )

    y = _build_labels(X)

    X_arr = X[FEATURE_COLS].values
    y_arr = y.values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])

    pipeline.fit(X_arr, y_arr)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    pca_step: PCA = pipeline.named_steps["pca"]
    n_components = int(pca_step.n_components_)
    evr = pca_step.explained_variance_ratio_.tolist()

    unique, counts = np.unique(y_arr, return_counts=True)
    class_dist = {_INT_TO_LABEL[int(k)]: int(v) for k, v in zip(unique, counts)}

    metadata: dict[str, Any] = {
        "n_train_samples": len(X),
        "n_components_selected": n_components,
        "class_distribution": class_dist,
        "explained_variance_ratio": evr,
        "cumulative_variance_explained": float(sum(evr)),
        "feature_cols": FEATURE_COLS,
        "model_path": str(model_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    _save_meta(metadata, model_path)

    logger.info(
        "Regime classifier trained: %d samples | %d PCA components (%.1f%% variance) | "
        "Risk-On=%d  Risk-Off=%d | saved → %s",
        len(X), n_components, sum(evr) * 100,
        class_dist.get(_RISK_ON, 0), class_dist.get(_RISK_OFF, 0),
        model_path,
    )

    return metadata


def predict_regime(
    portfolio_features: pd.DataFrame,
    macro: pd.DataFrame,
    ccl: pd.Series,
    model_path: Path | str = _DEFAULT_MODEL_PATH,
    nlp_score: float = 0.0,
) -> dict[str, Any]:
    """
    Predict regime for the most recent month in the provided data.

    Loads the pipeline serialized by train_regime_classifier() — no refitting occurs.
    The NLP score occupies slot [7] in the feature vector; pass the FinBERT monthly
    score when src/nlp/ is integrated. Until then the default 0.0 is used.

    Parameters
    ----------
    portfolio_features : pd.DataFrame
        Weekly portfolio-level signals up to and including the target month.
    macro : pd.DataFrame
        Monthly macro data up to and including the target month.
    ccl : pd.Series
        Weekly CCL (ARS/USD) up to and including the target month.
        Must include at least one prior month so ccl_variation can be computed.
    model_path : Path | str
        Path to the serialized joblib Pipeline.
    nlp_score : float
        Monthly FinBERT sentiment score. Defaults to 0.0 (placeholder).

    Returns
    -------
    dict: {"regime": "risk_on" | "risk_off", "probability": float}
          probability is the SVM posterior for the predicted class.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"Regime pipeline not found at '{model_path}'. "
            "Run train_regime_classifier() first (historical mode)."
        )

    pipeline: Pipeline = joblib.load(model_path)

    monthly = _build_monthly_feature_matrix(portfolio_features, macro, ccl, nlp_score)
    monthly = monthly.dropna()

    if monthly.empty:
        raise ValueError(
            "Feature matrix is empty after dropping NaN rows. "
            "Ensure portfolio_features, macro, and ccl cover the target month, "
            "and that ccl includes at least one prior month for ccl_variation."
        )

    X_last = monthly[FEATURE_COLS].iloc[[-1]].values
    target_month = monthly.index[-1]

    pred = int(pipeline.predict(X_last)[0])
    proba = pipeline.predict_proba(X_last)[0]
    prob_for_pred = float(proba[pred])

    regime = _INT_TO_LABEL[pred]

    logger.info(
        "Regime prediction: %s (p=%.3f) | month=%s | nlp_score=%.4f",
        regime, prob_for_pred, target_month.strftime("%Y-%m"), nlp_score,
    )

    return {"regime": regime, "probability": prob_for_pred}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_monthly_feature_matrix(
    portfolio_features: pd.DataFrame,
    macro: pd.DataFrame,
    ccl: pd.Series,
    nlp_score: float = 0.0,
) -> pd.DataFrame:
    """
    Construct the 8-column monthly feature matrix for PCA/SVM input.

    Technical signals (weekly) are resampled to month-end last observation.
    CCL variation is month-over-month % change of the monthly CCL close —
    the first month is always NaN and is dropped by the caller via dropna().
    Macro columns embi and reservas are merged on month-end index.
    nlp_score fills slot [7] as a scalar constant across all rows in the batch.
    """
    tech_monthly = portfolio_features.resample("ME").last()

    ccl_monthly = ccl.resample("ME").last()
    ccl_var = ccl_monthly.pct_change().rename("ccl_variation")

    macro_sel = _extract_macro_cols(macro)

    combined = tech_monthly.join(ccl_var, how="left")
    combined = combined.join(macro_sel, how="left")
    combined["nlp_score"] = nlp_score

    for col in FEATURE_COLS:
        if col not in combined.columns:
            logger.warning(
                "_build_monthly_feature_matrix: column '%s' absent — filling with NaN", col
            )
            combined[col] = np.nan

    return combined[FEATURE_COLS]


def _extract_macro_cols(macro: pd.DataFrame) -> pd.DataFrame:
    """Select embi and reservas from the macro DataFrame; NaN-fill any absent column."""
    needed = ["embi", "reservas"]
    df = pd.DataFrame(index=macro.index)
    for col in needed:
        if col in macro.columns:
            df[col] = macro[col]
        else:
            logger.warning("_extract_macro_cols: '%s' absent from macro — filling with NaN", col)
            df[col] = np.nan
    return df


def _build_labels(monthly_features: pd.DataFrame) -> pd.Series:
    """
    Binary regime labels derived from the monthly price_to_sma30 signal.

    price_to_sma30 > 1.0  →  1 (Risk-On)   portfolio above its 30-week SMA
    price_to_sma30 ≤ 1.0  →  0 (Risk-Off)  portfolio at or below its 30-week SMA
    """
    if "price_to_sma30" not in monthly_features.columns:
        raise ValueError(
            "'price_to_sma30' missing from feature matrix. "
            "Ensure portfolio_features was computed with sma_long=30."
        )
    labels = (monthly_features["price_to_sma30"] > 1.0).astype(int)
    labels.name = "regime"
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    logger.info(
        "_build_labels: Risk-On=%d (%.1f%%)  Risk-Off=%d (%.1f%%)",
        n1, n1 / len(labels) * 100, n0, n0 / len(labels) * 100,
    )
    return labels


def _save_meta(metadata: dict[str, Any], model_path: Path) -> None:
    meta_path = model_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug("Regime pipeline metadata saved → %s", meta_path)
