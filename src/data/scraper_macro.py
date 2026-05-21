"""
Scraper for Argentine macro variables (monthly frequency, month-end DatetimeIndex).

Variables:
  ipc        — monthly CPI level (INDEC official empalme back to 2003 via datos.gob.ar)
  embi       — Argentina EMBI risk spread in bps (Ámbito)
  reservas   — BCRA international reserves in USD millions (datos.gob.ar → BCRA API)
  tc_oficial — official ARS/USD exchange rate (datos.gob.ar → BCRA API)

If a variable fails entirely, its column is NaN — never raises, only warns.
IPC empalme covers the 2007-2015 INDEC intervention period; document in paper.
"""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

_DATOS_GOB_URL = (
    "https://apis.datos.gob.ar/series/api/series/"
    "?ids={series_id}&limit=5000&start_date={start}&end_date={end}&format=json"
)
_BCRA_URL = (
    "https://api.bcra.gob.ar/estadisticas/v3.0/datosvariable/{var_id}/{start}/{end}"
)
_AMBITO_EMBI_URL = (
    "https://mercados.ambito.com//riesgopais/historico-general/{start}/{end}"
)

_SERIES_IPC = "148.3_INIVELGENERAL_DICI_M_26"
_SERIES_RESERVAS = "174.1_IR_2012_0_15"
_SERIES_TC = "168.1_T_CAMBIOR_D_0_0_26"

_BCRA_RESERVAS_ID = 1
_BCRA_TC_ID = 4


# ─── Public API ───────────────────────────────────────────────────────────────

def get_all_macro(
    start: str,
    end: str,
    cached_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Download all macro variables and return a monthly DataFrame at month-end frequency.

    Variables that fail are NaN columns. cached_df is passed per-variable as fallback.
    """
    cached: dict[str, pd.Series] = {}
    if cached_df is not None and not cached_df.empty:
        for col in cached_df.columns:
            cached[col] = cached_df[col]

    fetchers: dict[str, Callable[[], pd.Series]] = {
        "ipc":       lambda: get_ipc(start, end, cached.get("ipc")),
        "embi":      lambda: get_embi(start, end, cached.get("embi")),
        "reservas":  lambda: get_reservas(start, end, cached.get("reservas")),
        "tc_oficial": lambda: get_tc_oficial(start, end, cached.get("tc_oficial")),
    }

    series_map: dict[str, pd.Series] = {}
    for name, fetch in fetchers.items():
        try:
            series_map[name] = fetch()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[macro/%s] unexpected error: %s — using NaN series", name, exc)
            series_map[name] = pd.Series(name=name, dtype=float)

    if not series_map:
        return pd.DataFrame()

    df = pd.concat(series_map.values(), axis=1)
    df.columns = list(series_map.keys())
    return df.sort_index()


def get_ipc(
    start: str,
    end: str,
    cached_series: pd.Series | None = None,
) -> pd.Series:
    """Monthly CPI level (INDEC empalme). datos.gob.ar only — no BCRA fallback."""
    s = _try_datos_gob(start, end, _SERIES_IPC, "ipc")
    if not s.empty:
        return s
    return _use_cached_fallback("ipc", start, end, cached_series)


def get_embi(
    start: str,
    end: str,
    cached_series: pd.Series | None = None,
) -> pd.Series:
    """Argentina EMBI spread in basis points. Ámbito primary, cache fallback."""
    s = _try_ambito_embi(start, end)
    if not s.empty:
        return s
    return _use_cached_fallback("embi", start, end, cached_series)


def get_reservas(
    start: str,
    end: str,
    cached_series: pd.Series | None = None,
) -> pd.Series:
    """BCRA international reserves (USD millions). datos.gob.ar → BCRA API → cache."""
    s = _try_datos_gob(start, end, _SERIES_RESERVAS, "reservas")
    if not s.empty:
        return s
    s = _try_bcra_api(start, end, _BCRA_RESERVAS_ID, "reservas")
    if not s.empty:
        return s
    return _use_cached_fallback("reservas", start, end, cached_series)


def get_tc_oficial(
    start: str,
    end: str,
    cached_series: pd.Series | None = None,
) -> pd.Series:
    """Official ARS/USD exchange rate. datos.gob.ar → BCRA API → cache."""
    s = _try_datos_gob(start, end, _SERIES_TC, "tc_oficial")
    if not s.empty:
        return s
    s = _try_bcra_api(start, end, _BCRA_TC_ID, "tc_oficial")
    if not s.empty:
        return s
    return _use_cached_fallback("tc_oficial", start, end, cached_series)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _to_month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index + pd.offsets.MonthEnd(0)


def _try_datos_gob(
    start: str,
    end: str,
    series_id: str,
    name: str,
) -> pd.Series:
    try:
        import requests

        url = _DATOS_GOB_URL.format(series_id=series_id, start=start, end=end)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            return pd.Series(name=name, dtype=float)

        dates = pd.to_datetime([r[0] for r in rows], errors="coerce")
        values = pd.to_numeric([r[1] for r in rows], errors="coerce")
        s = pd.Series(values, index=_to_month_end(dates), name=name).dropna()
        logger.info("[macro/%s] datos.gob.ar: %d rows", name, len(s))
        return s
    except Exception as exc:  # noqa: BLE001
        logger.debug("[macro/%s] datos.gob.ar failed: %s", name, exc)
        return pd.Series(name=name, dtype=float)


def _try_ambito_embi(start: str, end: str) -> pd.Series:
    try:
        import requests

        start_fmt = pd.Timestamp(start).strftime("%d/%m/%Y")
        end_fmt = pd.Timestamp(end).strftime("%d/%m/%Y")
        url = _AMBITO_EMBI_URL.format(start=start_fmt, end=end_fmt)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data or len(data) < 2:
            return pd.Series(name="embi", dtype=float)

        rows = data[1:]
        dates = pd.to_datetime([r[0] for r in rows], dayfirst=True, errors="coerce")
        values = pd.to_numeric(
            [str(r[1]).replace(",", ".") for r in rows], errors="coerce"
        )
        s = pd.Series(values, index=dates.normalize(), name="embi").dropna()
        s = s.resample("ME").last().dropna()
        s.index = _to_month_end(s.index)
        logger.info("[macro/embi] Ámbito: %d monthly observations", len(s))
        return s
    except Exception as exc:  # noqa: BLE001
        logger.debug("[macro/embi] Ámbito API failed: %s", exc)
        return pd.Series(name="embi", dtype=float)


def _try_bcra_api(
    start: str,
    end: str,
    var_id: int,
    name: str,
) -> pd.Series:
    try:
        import requests

        url = _BCRA_URL.format(
            var_id=var_id,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=pd.Timestamp(end).strftime("%Y-%m-%d"),
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return pd.Series(name=name, dtype=float)

        dates = pd.to_datetime([r["fecha"] for r in results], errors="coerce")
        values = pd.to_numeric([r["valor"] for r in results], errors="coerce")
        s = pd.Series(values, index=dates.normalize(), name=name).dropna()
        s = s.resample("ME").last().dropna()
        s.index = _to_month_end(s.index)
        logger.info("[macro/%s] BCRA API: %d monthly observations", name, len(s))
        return s
    except Exception as exc:  # noqa: BLE001
        logger.debug("[macro/%s] BCRA API failed: %s", name, exc)
        return pd.Series(name=name, dtype=float)


def _use_cached_fallback(
    name: str,
    start: str,
    end: str,
    cached_series: pd.Series | None,
) -> pd.Series:
    if cached_series is None or cached_series.empty:
        logger.warning("[macro/%s] all sources failed and no cache — NaN series", name)
        return pd.Series(name=name, dtype=float)

    window = cached_series.loc[start:end]
    if not window.empty:
        logger.warning("[macro/%s] using cached fallback for %s → %s", name, start, end)
        return window.copy()

    valid = cached_series.dropna()
    if valid.empty:
        logger.warning("[macro/%s] cache has no valid values", name)
        return pd.Series(name=name, dtype=float)

    idx = pd.date_range(
        pd.Timestamp(start) + pd.offsets.MonthEnd(0),
        pd.Timestamp(end) + pd.offsets.MonthEnd(0),
        freq="ME",
    )
    idx = _to_month_end(idx)
    logger.warning(
        "[macro/%s] extending last cached value (%.4f) to fill %s → %s",
        name, valid.iloc[-1], start, end,
    )
    return pd.Series(valid.iloc[-1], index=idx, name=name)
