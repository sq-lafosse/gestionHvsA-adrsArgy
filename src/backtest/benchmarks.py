"""
Benchmark equity curves for gestionHvsA-adrsArgy.

Two active benchmarks plus one pending, each normalized to 1.0 at the start of
the evaluation period:

    ew_bnh              — Equal-weight Buy & Hold across all ADRs (no rebalancing)
    merval_usd          — Merval ARS / CCL (Contado con Liquidación) buy & hold
    fima_acciones_usd   — FIMA Acciones FCI (fondo_id=21, clase_id=21) in USD via CCL
                          (returns empty Series — CAFCI API requires Bearer JWT)

AL30D was removed from the universe (D8: portfolio 100 % equity; historical data
not available in regulated sources equivalent to SEC).

CCL is the relevant FX for ADR/local arbitrage and for investors operating in
the USD-denominated space of Argentine markets — consistent with the project
benchmark methodology described in CLAUDE.md.

FIMA Acciones scraping strategy (Opción A → Opción B → empty Series):
  A) GET api.cafci.org.ar/estadisticas/informacion/diaria/?fecha=YYYY-MM-DD
     (one request per week; vcp field filtered for fondo_id=21)
  B) GET api.cafci.org.ar/fondo/21/clase/21/rendimiento/{start}/{end}
     (compound daily yields from 1.0 to reconstruct VCP index)
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CAFCI_BASE = "https://api.cafci.org.ar"
_FIMA_FONDO = 21
_FIMA_CLASE = 21
_REQUEST_TIMEOUT = 10


def compute_benchmarks(
    adrs: pd.DataFrame,
    merval: pd.Series,
    ccl: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """
    Compute equity curves for all active benchmarks over [start_date, end_date].

    Parameters
    ----------
    adrs : pd.DataFrame
        Weekly ADR adjusted-close prices (columns = tickers).
    merval : pd.Series
        Weekly Merval ARS prices.
    ccl : pd.Series
        Weekly CCL (ARS/USD) prices.
    start_date, end_date : pd.Timestamp
        Inclusive bounds of the evaluation window, matching the portfolio period.

    Returns
    -------
    dict with keys "ew_bnh", "merval_usd", "fima_acciones_usd".
    Each value is a pd.Series normalized to 1.0 at start_date.
    AL30D removed (D8: portfolio 100 % equity).
    """
    return {
        "ew_bnh":            _ew_buy_and_hold(adrs, start_date, end_date),
        "merval_usd":        _merval_usd(merval, ccl, start_date, end_date),
        "fima_acciones_usd": fima_acciones_usd(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            ccl,
        ),
    }


# ─── FIMA Acciones benchmark ──────────────────────────────────────────────────

def fima_acciones_usd(start: str, end: str, ccl: pd.Series) -> pd.Series:
    """
    Descarga VCP histórico de FIMA Acciones (fondo_id=21, clase_id=21)
    via API Cafci, convierte a USD dividiendo por CCL semanal,
    normaliza a 1.0 en la fecha de inicio del rango.
    Retorna pd.Series semanal con nombre "fima_acciones_usd".

    # FIMA Acciones: datos históricos no disponibles públicamente.
    # La API de CAFCI requiere Bearer token de usuario registrado.
    # Función lista para activar si se obtiene el CSV manualmente
    # en data/cache/fima_acciones_vcp.csv
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    vcp = _fetch_fima_vcp_daily(start_ts, end_ts)
    if vcp is None:
        logger.info("fima_acciones_usd: Opción A falló — intentando fallback Opción B")
        vcp = _fetch_fima_vcp_from_rendimiento(start_ts, end_ts)
    if vcp is None or vcp.empty:
        logger.warning("fima_acciones_usd: ambas opciones fallaron — retornando serie vacía")
        return pd.Series(dtype=float, name="fima_acciones_usd")

    vcp_weekly = vcp.resample("W-FRI").last().dropna()

    ccl_weekly = ccl.resample("W-FRI").last()
    merged = pd.DataFrame({"vcp": vcp_weekly, "ccl": ccl_weekly}).ffill().dropna()

    if merged.empty:
        logger.warning("fima_acciones_usd: sin solapamiento con CCL — retornando serie vacía")
        return pd.Series(dtype=float, name="fima_acciones_usd")

    fima_usd = (merged["vcp"] / merged["ccl"]).loc[start_ts:end_ts]

    if fima_usd.empty or fima_usd.iloc[0] == 0:
        return pd.Series(dtype=float, name="fima_acciones_usd")

    equity = (fima_usd / fima_usd.iloc[0]).rename("fima_acciones_usd")
    return equity


def _fetch_fima_vcp_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """
    Opción A: poll /estadisticas/informacion/diaria/?fecha=YYYY-MM-DD once per week.

    Returns daily VCP ARS series (indexed by date), or None if >50 % of
    requests fail or the endpoint clearly does not accept the fecha param.
    """
    weekly_dates = pd.date_range(start=start, end=end, freq="W-MON")
    if weekly_dates.empty:
        return None

    records: dict[pd.Timestamp, float] = {}
    failed = 0

    for dt in weekly_dates:
        fecha = dt.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                f"{_CAFCI_BASE}/estadisticas/informacion/diaria/",
                params={"fecha": fecha},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code == 404:
                logger.warning("fima daily: 404 — endpoint no acepta ?fecha, usando fallback")
                return None
            if resp.status_code != 200:
                failed += 1
                continue

            data  = resp.json()
            items = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                failed += 1
                continue

            for item in items:
                fondo_id = item.get("fondo") or item.get("fondoId") or item.get("fondo_id")
                if str(fondo_id) == str(_FIMA_FONDO):
                    raw_vcp = item.get("vcp") or item.get("cuotaparte")
                    if raw_vcp is not None:
                        records[dt] = float(raw_vcp)
                        break

        except Exception as exc:
            logger.debug("fima daily fetch failed for %s: %s", fecha, exc)
            failed += 1

    total = len(weekly_dates)
    if failed > total * 0.5:
        logger.warning(
            "fima daily: %d/%d requests failed — endpoint puede no soportar ?fecha",
            failed, total,
        )
        return None

    if not records:
        logger.warning(
            "fima daily: requests OK pero fondo_id=%d no encontrado en respuestas", _FIMA_FONDO
        )
        return None

    logger.info("fima daily (Opción A): %d puntos VCP descargados para fondo_id=%d", len(records), _FIMA_FONDO)
    return pd.Series(records, name="vcp_ars").sort_index()


def _fetch_fima_vcp_from_rendimiento(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """
    Opción B: reconstruct VCP index by compounding daily yields from
    GET /fondo/21/clase/21/rendimiento/{start}/{end}.

    Returns a VCP index starting at 1.0 on the first available date, or None on failure.
    """
    url = (
        f"{_CAFCI_BASE}/fondo/{_FIMA_FONDO}/clase/{_FIMA_CLASE}"
        f"/rendimiento/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            logger.warning("fima rendimiento: HTTP %d — %s", resp.status_code, url)
            return None

        data = resp.json()

        # Defensive parse: try several common nested structures
        items: list | None = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            inner = data.get("data", data)
            if isinstance(inner, list):
                items = inner
            elif isinstance(inner, dict):
                for key in ("rendimiento", "data", "items", "results"):
                    candidate = inner.get(key)
                    if isinstance(candidate, list):
                        items = candidate
                        break

        if not items:
            logger.warning(
                "fima rendimiento: estructura de respuesta inesperada — primeros 300 chars: %s",
                str(data)[:300],
            )
            return None

        records: dict[pd.Timestamp, float] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            fecha_raw = item.get("fecha") or item.get("date") or item.get("dia")
            rend_raw  = item.get("rendimiento") or item.get("yield") or item.get("variacion")
            if fecha_raw is None or rend_raw is None:
                continue
            try:
                records[pd.Timestamp(fecha_raw)] = float(rend_raw)
            except (ValueError, TypeError):
                continue

        if not records:
            logger.warning("fima rendimiento: sin registros parseables en la respuesta")
            return None

        daily_yields = pd.Series(records, name="yield_pct").sort_index()
        vcp_reconstructed = (1 + daily_yields / 100).cumprod()

        logger.info(
            "fima rendimiento (Opción B): VCP reconstruido con %d días", len(vcp_reconstructed)
        )
        return vcp_reconstructed.rename("vcp_reconstructed")

    except Exception as exc:
        logger.warning("fima rendimiento: excepción — %s", exc)
        return None


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _ew_buy_and_hold(
    adrs: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Equal-weight buy & hold across all ADRs.

    Invests equal dollar amounts in each ADR at start_date and holds without
    rebalancing. Portfolio value = mean of individually normalized price series.
    Tickers with NaN on start_date are excluded (no initial position possible).
    """
    prices = adrs.loc[start_date:end_date].ffill().dropna(how="all")

    if prices.empty:
        logger.warning(
            "_ew_buy_and_hold: no ADR data for %s → %s", start_date.date(), end_date.date()
        )
        return pd.Series(dtype=float, name="ew_bnh")

    valid_cols = prices.columns[prices.iloc[0].notna()]
    if valid_cols.empty:
        logger.warning("_ew_buy_and_hold: all tickers NaN at start_date %s", start_date.date())
        return pd.Series(dtype=float, name="ew_bnh")

    if len(valid_cols) < len(prices.columns):
        dropped = set(prices.columns) - set(valid_cols)
        logger.warning(
            "_ew_buy_and_hold: excluded %d tickers with NaN at start: %s", len(dropped), dropped
        )

    prices = prices[valid_cols]
    normalized = prices / prices.iloc[0]
    equity = normalized.mean(axis=1)
    equity.name = "ew_bnh"
    return equity



def _merval_usd(
    merval: pd.Series,
    ccl: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Merval in USD: Merval ARS / CCL (Contado con Liquidación).

    Both series are aligned on their common weekly index. Rows with NaN
    in either series are dropped before normalization.
    """
    merged = (
        pd.DataFrame({
            "merval_ars": merval.loc[start_date:end_date],
            "ccl":        ccl.loc[start_date:end_date],
        })
        .ffill()
        .dropna()
    )

    if merged.empty:
        logger.warning(
            "_merval_usd: no overlapping data for %s → %s", start_date.date(), end_date.date()
        )
        return pd.Series(dtype=float, name="merval_usd")

    merval_usd_prices = merged["merval_ars"] / merged["ccl"]
    equity = merval_usd_prices / merval_usd_prices.iloc[0]
    equity.name = "merval_usd"
    return equity
