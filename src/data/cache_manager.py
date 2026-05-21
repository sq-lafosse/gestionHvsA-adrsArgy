"""
Cache manager for the gestionHvsA-adrsArgy data pipeline.

Responsibilities:
- Persist downloaded data to Parquet + JSON sidecar format
- Enforce historical cache immutability via double guard
- Provide strict temporal cutoffs on all load operations (anti-leakage)
- Append live monthly data without touching historical cache
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(os.getenv("DATA_CACHE_PATH", "data/cache"))
_PRICES_DIR = _CACHE_ROOT / "prices"
_MACRO_DIR = _CACHE_ROOT / "macro"
_MANIFEST_PATH = _CACHE_ROOT / ".cache_manifest.json"
_MANIFEST_VERSION = "1.0"

_ADR_PATH = _PRICES_DIR / "adrs.parquet"
_SOVEREIGN_PATH = _PRICES_DIR / "sovereign_bond.parquet"
_MERVAL_PATH = _PRICES_DIR / "merval.parquet"
_CCL_PATH = _PRICES_DIR / "ccl.parquet"
_MACRO_PATH = _MACRO_DIR / "macro.parquet"


@dataclass
class CachePayload:
    adrs: pd.DataFrame    # weekly Adj Close, columns = tickers, DatetimeIndex (tz-naive)
    sovereign_bond: pd.Series  # weekly USD prices, name="sovereign_bond_usd"
    merval: pd.Series          # weekly ARS prices, name="merval_ars"
    ccl: pd.Series             # weekly ARS/USD, name="ccl"
    macro: pd.DataFrame        # monthly macro variables, DatetimeIndex (tz-naive, month-end)


# ─── Manifest ────────────────────────────────────────────────────────────────

def historical_cache_exists() -> bool:
    """Returns True if the historical cache has been fully written and is immutable."""
    if not _MANIFEST_PATH.exists():
        return False
    try:
        return bool(_read_manifest().get("historical_complete", False))
    except (json.JSONDecodeError, KeyError):
        logger.warning("Manifest file is corrupted or unreadable: %s", _MANIFEST_PATH)
        return False


def get_last_cached_date() -> pd.Timestamp | None:
    """Returns the last live date recorded in the manifest, or None if no live appends yet."""
    if not _MANIFEST_PATH.exists():
        return None
    try:
        raw = _read_manifest().get("last_live_date")
        return pd.Timestamp(raw) if raw else None
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _read_manifest() -> dict:
    with _MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest: dict) -> None:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


# ─── Parquet + sidecar I/O ────────────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def _save_meta(attrs: dict, path: Path) -> None:
    meta_path = path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(attrs, f, indent=2, default=str)


def _load_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _series_to_df(series: pd.Series) -> pd.DataFrame:
    return series.to_frame(name=series.name or "value")


def _df_to_series(df: pd.DataFrame) -> pd.Series:
    col = df.columns[0]
    s = df[col].copy()
    s.name = col
    return s


def _cutoff_ts(cutoff: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(cutoff).normalize()


# ─── Historical cache ─────────────────────────────────────────────────────────

def save_historical_cache(payload: CachePayload) -> None:
    """
    Persist the full historical dataset (2015→2023) to Parquet.

    Raises RuntimeError if historical cache already exists — defense in depth
    beyond the guard in loader.py.
    """
    if historical_cache_exists():
        raise RuntimeError(
            "Historical cache already exists and is immutable. "
            "Delete the manifest manually to force a re-download (not recommended)."
        )

    logger.info("Writing historical cache to %s", _CACHE_ROOT)

    _save_parquet(payload.adrs, _ADR_PATH)
    _save_meta(payload.adrs.attrs, _ADR_PATH)

    _save_parquet(_series_to_df(payload.sovereign_bond), _SOVEREIGN_PATH)
    _save_meta(payload.sovereign_bond.attrs, _SOVEREIGN_PATH)

    _save_parquet(_series_to_df(payload.merval), _MERVAL_PATH)
    _save_meta(payload.merval.attrs, _MERVAL_PATH)

    _save_parquet(_series_to_df(payload.ccl), _CCL_PATH)
    _save_meta(payload.ccl.attrs, _CCL_PATH)

    _save_parquet(payload.macro, _MACRO_PATH)
    _save_meta(payload.macro.attrs, _MACRO_PATH)

    row_counts = {
        "adrs": len(payload.adrs),
        "sovereign_bond": len(payload.sovereign_bond),
        "merval": len(payload.merval),
        "ccl": len(payload.ccl),
        "macro": len(payload.macro),
    }
    _write_manifest({
        "historical_complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": _MANIFEST_VERSION,
        "row_counts": row_counts,
        "last_live_date": None,
    })
    logger.info("Historical cache written. Row counts: %s", row_counts)


# ─── Live append ──────────────────────────────────────────────────────────────

def append_live_month(payload: CachePayload) -> None:
    """
    Append one month of live data to the existing cache.

    Raises:
        RuntimeError: if historical cache does not exist yet.
        ValueError: if any component's earliest new date is not strictly after
                    the last live date recorded in the manifest.
    """
    if not historical_cache_exists():
        raise RuntimeError(
            "Cannot append live data: historical cache has not been written yet."
        )

    _validate_live_payload_order(payload)

    logger.info("Appending live month to cache")

    _append_parquet(_ADR_PATH, payload.adrs)
    _append_parquet(_SOVEREIGN_PATH, _series_to_df(payload.sovereign_bond))
    _append_parquet(_MERVAL_PATH, _series_to_df(payload.merval))
    _append_parquet(_CCL_PATH, _series_to_df(payload.ccl))
    _append_parquet(_MACRO_PATH, payload.macro)

    new_max = max(
        payload.adrs.index.max(),
        payload.sovereign_bond.index.max(),
        payload.merval.index.max(),
        payload.ccl.index.max(),
        payload.macro.index.max(),
    )
    manifest = _read_manifest()
    manifest["last_live_date"] = pd.Timestamp(new_max).isoformat()
    _write_manifest(manifest)
    logger.info("Live month appended. New last cached date: %s", pd.Timestamp(new_max).date())


def _append_parquet(path: Path, new_df: pd.DataFrame) -> None:
    existing = _load_parquet(path)
    combined = pd.concat([existing, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    _save_parquet(combined, path)


def _validate_live_payload_order(payload: CachePayload) -> None:
    """Raise ValueError if any payload component overlaps with already-cached live data."""
    last_date = get_last_cached_date()
    if last_date is None:
        return  # First live append — no prior live date to check against

    components: dict[str, pd.DatetimeIndex] = {
        "adrs": payload.adrs.index,
        "sovereign_bond": payload.sovereign_bond.index,
        "merval": payload.merval.index,
        "ccl": payload.ccl.index,
        "macro": payload.macro.index,
    }
    for name, index in components.items():
        new_min = pd.Timestamp(index.min()).normalize()
        if new_min <= last_date.normalize():
            raise ValueError(
                f"Live append rejected for '{name}': earliest new date {new_min.date()} "
                f"is not strictly after last cached date {last_date.date()}. "
                "Months must be appended in strict chronological order."
            )


# ─── Load with anti-leakage cutoff ───────────────────────────────────────────

def load_adrs_up_to(cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Load ADR prices up to and including cutoff date."""
    if not _ADR_PATH.exists():
        logger.warning("ADR cache not found: %s", _ADR_PATH)
        return pd.DataFrame()
    df = _load_parquet(_ADR_PATH)
    result = df.loc[: _cutoff_ts(cutoff)].copy()
    result.attrs = _load_meta(_ADR_PATH)
    return result


def load_sovereign_bond_up_to(cutoff: str | pd.Timestamp) -> pd.Series:
    """Load sovereign bond prices up to and including cutoff date."""
    if not _SOVEREIGN_PATH.exists():
        logger.warning("Sovereign bond cache not found: %s", _SOVEREIGN_PATH)
        return pd.Series(name="sovereign_bond_usd", dtype=float)
    df = _load_parquet(_SOVEREIGN_PATH)
    s = _df_to_series(df.loc[: _cutoff_ts(cutoff)])
    s.attrs = _load_meta(_SOVEREIGN_PATH)
    return s


def load_merval_up_to(cutoff: str | pd.Timestamp) -> pd.Series:
    """Load Merval prices (ARS) up to and including cutoff date."""
    if not _MERVAL_PATH.exists():
        logger.warning("Merval cache not found: %s", _MERVAL_PATH)
        return pd.Series(name="merval_ars", dtype=float)
    df = _load_parquet(_MERVAL_PATH)
    s = _df_to_series(df.loc[: _cutoff_ts(cutoff)])
    s.attrs = _load_meta(_MERVAL_PATH)
    return s


def load_ccl_up_to(cutoff: str | pd.Timestamp) -> pd.Series:
    """Load CCL (ARS/USD) up to and including cutoff date."""
    if not _CCL_PATH.exists():
        logger.warning("CCL cache not found: %s", _CCL_PATH)
        return pd.Series(name="ccl", dtype=float)
    df = _load_parquet(_CCL_PATH)
    s = _df_to_series(df.loc[: _cutoff_ts(cutoff)])
    s.attrs = _load_meta(_CCL_PATH)
    return s


def load_macro_up_to(cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """
    Load macro variables up to and including cutoff date (monthly frequency).

    The cutoff is snapped to month-end so callers can pass any day within the
    target month (e.g., "2024-01-15" includes all months through January 2024).
    The caller is responsible for ensuring the cutoff does not create leakage
    (macro data for month M is typically published in month M+1).
    """
    if not _MACRO_PATH.exists():
        logger.warning("Macro cache not found: %s", _MACRO_PATH)
        return pd.DataFrame()
    df = _load_parquet(_MACRO_PATH)
    cutoff_me = pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)
    result = df.loc[:cutoff_me].copy()
    result.attrs = _load_meta(_MACRO_PATH)
    return result
