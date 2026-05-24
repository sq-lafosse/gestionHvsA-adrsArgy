"""
SEC EDGAR news downloader for gestionHvsA-adrsArgy.

Downloads 6-K and 20-F filings for all 10 Argentine ADRs over the live period.
Output: data/news/{year}/{month}/sec_{ticker}_{date}_{acc_tail}.{ext}
  - .txt  for HTML/SGML/text documents
  - .pdf  for native PDF documents

EDGAR rate limit: 0.5 s between requests (policy).
User-Agent: "gestionHvsA research@example.com" (required by EDGAR).

Skip-if-exists: re-runs are idempotent — already-downloaded files are not
re-fetched.  All errors are logged as warnings; the function never raises.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_EDGAR_ARCHIVES    = "https://www.sec.gov/Archives/edgar/data"
_RATE_LIMIT        = 0.5   # seconds between every HTTP request
_TIMEOUT           = 30    # seconds per request

_FORM_TYPES = {"6-K", "20-F"}

_ADR_CIKS: dict[str, str] = {
    "YPF":   "0000904851",
    "GGAL":  "0001114700",
    "BMA":   "0001347426",
    "PAM":   "0001469395",
    "TGS":   "0000931427",
    "CEPU":  "0001717161",
    "EDN":   "0001395213",
    "LOMA":  "0001711375",
    "CRESY": "0001034957",
    "IRS":   "0000933267",
}

_HEADERS_DATA = {
    "User-Agent":      "gestionHvsA research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host":            "data.sec.gov",
}
_HEADERS_WWW = {
    "User-Agent":      "gestionHvsA research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host":            "www.sec.gov",
}

# ── Macro sources ─────────────────────────────────────────────────────────────

_MONTHS_ES = ["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"]

_BCRA_BASE = "https://www.bcra.gob.ar"
_FED_BASE  = "https://www.federalreserve.gov"

_HEADERS_BCRA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/pdf,*/*",
}
_HEADERS_FED = {
    "User-Agent": "gestionHvsA research@example.com",
    "Accept":     "text/html,*/*",
}

# Exact FOMC meeting dates as released by the Federal Reserve (2024–2025)
_FOMC_DATES = [
    "20240131", "20240320", "20240501", "20240612",
    "20240731", "20240918", "20241107", "20241218",
    "20250129", "20250319", "20250507", "20250618",
    "20250730", "20250917", "20251105", "20251210",
]


# ─── Public API ───────────────────────────────────────────────────────────────

def download_macro_documents(start: str, end: str) -> dict[str, dict[str, int]]:
    """
    Download BCRA and Fed macro documents for each month in [start, end].

    Sources (approved):
      bcra_monetario — BCRA Informe Monetario Mensual (PDF). Available from
                       2024-06 onward; 2024-01..05 are 404 and skipped.
                       Published ~2 weeks after month-close → saved in M+1 folder.
      bcra_rem       — BCRA Relevamiento de Expectativas de Mercado (PDF).
                       Published last business day of the covered month → same-month folder.
      fed_fomc       — Federal Reserve FOMC minutes page (HTML saved as .txt).
                       16 hardcoded meeting dates for 2024–2025.

    Parameters
    ----------
    start, end : str
        Format 'YYYY-MM'. Both inclusive (coverage month).

    Returns
    -------
    dict mapping 'YYYY-MM' → {'bcra_monetario': N, 'bcra_rem': N, 'fed_fomc': N}
    for every month in [start, end]. Counts are by coverage month.
    All errors are logged as warnings; the function never raises.
    """
    start_ts = pd.Timestamp(start + "-01")
    end_ts   = pd.Timestamp(end   + "-01") + pd.offsets.MonthEnd(0)
    months   = _month_range(start_ts, end_ts)

    counts: dict[str, dict[str, int]] = {
        m: {"bcra_monetario": 0, "bcra_rem": 0, "fed_fomc": 0}
        for m in months
    }

    for month_key in months:
        ts    = pd.Timestamp(month_key + "-01")
        year  = ts.year
        month = ts.month
        mes   = _MONTHS_ES[month - 1]
        yy    = str(year)[-2:]

        # BCRA Monetario — published in M+1, saved there
        pub_ts  = ts + pd.offsets.MonthBegin(1)
        pub_dir = Path("data/news") / pub_ts.strftime("%Y") / pub_ts.strftime("%m")
        if _download_bcra_monetario(mes, yy, str(year), pub_dir):
            counts[month_key]["bcra_monetario"] += 1

        # BCRA REM — published same month
        rem_dir = Path("data/news") / f"{year}" / f"{month:02d}"
        if _download_bcra_rem(mes, str(year), rem_dir):
            counts[month_key]["bcra_rem"] += 1

    # FOMC — fixed dates, count under coverage month
    for date_str in _FOMC_DATES:
        fomc_ts = pd.Timestamp(date_str)
        if not (start_ts <= fomc_ts <= end_ts):
            continue
        month_key = fomc_ts.strftime("%Y-%m")
        fomc_dir  = Path("data/news") / fomc_ts.strftime("%Y") / fomc_ts.strftime("%m")
        if _download_fomc(date_str, fomc_dir):
            counts[month_key]["fed_fomc"] += 1

    return counts


def download_news_for_period(start: str, end: str) -> dict[str, dict[str, int]]:
    """
    Download SEC 6-K and 20-F filings for all ADRs within [start, end].

    Parameters
    ----------
    start, end : str
        Format 'YYYY-MM'. Both inclusive.

    Returns
    -------
    dict mapping 'YYYY-MM' → {'sec_filings': N} for every month in the range.
    All months are present in the result even if N == 0.
    """
    start_ts = pd.Timestamp(start + "-01")
    end_ts   = pd.Timestamp(end   + "-01") + pd.offsets.MonthEnd(0)

    monthly_counts: dict[str, int] = defaultdict(int)

    for ticker, cik in _ADR_CIKS.items():
        filings = _fetch_submission_filings(ticker, cik, start_ts, end_ts)
        for filing in filings:
            filing_date = pd.Timestamp(filing["date"])
            year  = filing_date.strftime("%Y")
            month = filing_date.strftime("%m")
            key   = f"{year}-{month}"

            saved = _download_filing(
                ticker=ticker,
                cik=cik,
                accession=filing["accession"],
                primary_doc=filing["primary_doc"],
                filing_date=filing_date,
                out_dir=Path("data/news") / year / month,
            )
            if saved:
                monthly_counts[key] += 1

    return {
        m: {"sec_filings": monthly_counts.get(m, 0)}
        for m in _month_range(start_ts, end_ts)
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_submission_filings(
    ticker: str,
    cik: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict]:
    """
    Fetch all 6-K / 20-F filings for one ticker from EDGAR submissions JSON.

    EDGAR paginates older filings into auxiliary JSON files listed under
    `filings.files`.  We fetch those too so that early-2024 filings are not
    missed if the company has a large filing history.

    Returns list of dicts: {date, accession, primary_doc, form}.
    Only filings whose date falls within [start_ts, end_ts] are returned.
    """
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{_EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"

    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, headers=_HEADERS_DATA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("%s: EDGAR submissions HTTP %d — skipping", ticker, resp.status_code)
            return []
        data = resp.json()
    except Exception as exc:
        logger.warning("%s: failed to fetch EDGAR submissions — %s", ticker, exc)
        return []

    filings: list[dict] = []

    # Primary (most-recent) filings block
    filings += _parse_filings_block(data.get("filings", {}).get("recent", {}))

    # Older paginated blocks
    for page in data.get("filings", {}).get("files", []):
        page_name = page.get("name", "")
        if page_name:
            filings += _fetch_older_filings_page(ticker, page_name)

    # Filter to the requested date range
    filtered = [
        f for f in filings
        if start_ts <= pd.Timestamp(f["date"]) <= end_ts
    ]

    logger.info(
        "%s: %d 6-K/20-F filings in range (of %d total)",
        ticker, len(filtered), len(filings),
    )
    return filtered


def _parse_filings_block(block: dict) -> list[dict]:
    """Convert the parallel-array EDGAR filings block into a list of dicts."""
    forms        = block.get("form", [])
    dates        = block.get("filingDate", [])
    accessions   = block.get("accessionNumber", [])
    primary_docs = block.get("primaryDocument", [])

    return [
        {"form": form, "date": date, "accession": acc, "primary_doc": doc}
        for form, date, acc, doc in zip(forms, dates, accessions, primary_docs)
        if form in _FORM_TYPES
    ]


def _fetch_older_filings_page(ticker: str, page_name: str) -> list[dict]:
    """Fetch one pagination page of older filings from data.sec.gov/submissions/."""
    url = f"{_EDGAR_SUBMISSIONS}/{page_name}"
    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, headers=_HEADERS_DATA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("%s: older filings page %s HTTP %d", ticker, page_name, resp.status_code)
            return []
        return _parse_filings_block(resp.json())
    except Exception as exc:
        logger.debug("%s: older filings page fetch failed — %s", ticker, exc)
        return []


def _download_filing(
    ticker: str,
    cik: str,
    accession: str,
    primary_doc: str,
    filing_date: pd.Timestamp,
    out_dir: Path,
) -> bool:
    """
    Download the primary document of one SEC filing.

    File is skipped if it already exists on disk.
    Returns True if the file is available on disk after the call.
    """
    ext = ".pdf" if primary_doc.lower().endswith(".pdf") else ".txt"
    # Last 7 chars of accession (without dashes) make it unique within ticker+date
    acc_tail = accession.replace("-", "")[-7:]
    filename = f"sec_{ticker}_{filing_date.strftime('%Y-%m-%d')}_{acc_tail}{ext}"
    out_path = out_dir / filename

    if out_path.exists():
        logger.debug("%s: already on disk — %s", ticker, filename)
        return True

    cik_num    = cik.lstrip("0")
    acc_nodash = accession.replace("-", "")
    url = f"{_EDGAR_ARCHIVES}/{cik_num}/{acc_nodash}/{primary_doc}"

    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, headers=_HEADERS_WWW, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(
                "%s: HTTP %d downloading %s", ticker, resp.status_code, primary_doc
            )
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        logger.info("%s: saved %s (%d bytes)", ticker, filename, len(resp.content))
        return True

    except Exception as exc:
        logger.warning("%s: exception downloading %s — %s", ticker, primary_doc, exc)
        return False


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Return ['YYYY-MM', ...] for every month from start to end inclusive."""
    months  = []
    current = start.replace(day=1)
    end_m   = end.replace(day=1)
    while current <= end_m:
        months.append(current.strftime("%Y-%m"))
        current += pd.offsets.MonthBegin(1)
    return months


# ─── Macro source helpers ─────────────────────────────────────────────────────

def _download_bcra_monetario(mes: str, yy: str, year: str, out_dir: Path) -> bool:
    """
    Download BCRA Informe Monetario Mensual for the given month.

    mes  : Spanish abbreviation (e.g. 'ene')
    yy   : 2-digit year (e.g. '24')
    year : 4-digit year string for the filename (e.g. '2024')
    out_dir : publication-month folder (M+1)

    Returns True if file is on disk after the call.
    Logs a warning and returns False for 2024-01..05 (404 on BCRA server).
    """
    filename = f"bcra_monetario_{mes}_{year}.pdf"
    out_path = out_dir / filename
    if out_path.exists():
        logger.debug("bcra_monetario: already on disk — %s", filename)
        return True

    url = (
        f"{_BCRA_BASE}/Pdfs/PublicacionesEstadisticas/"
        f"informe-monetario-mensual-{mes}-{yy}.pdf"
    )
    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, headers=_HEADERS_BCRA, timeout=_TIMEOUT)
        if resp.status_code == 404:
            logger.warning("bcra_monetario: 404 para %s-%s (no publicado)", mes, year)
            return False
        if resp.status_code != 200:
            logger.warning("bcra_monetario: HTTP %d para %s-%s", resp.status_code, mes, year)
            return False
        if resp.content[:4] != b"%PDF":
            logger.warning("bcra_monetario: respuesta no es PDF para %s-%s", mes, year)
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        logger.info("bcra_monetario: saved %s (%d bytes)", filename, len(resp.content))
        return True

    except Exception as exc:
        logger.warning("bcra_monetario: excepción para %s-%s — %s", mes, year, exc)
        return False


def _download_bcra_rem(mes: str, year: str, out_dir: Path) -> bool:
    """
    Download BCRA REM (Relevamiento de Expectativas de Mercado).

    Uses variant A URL pattern only (confirmed 24/24 months via verification).

    Returns True if file is on disk after the call.
    """
    filename = f"bcra_rem_{mes}_{year}.pdf"
    out_path = out_dir / filename
    if out_path.exists():
        logger.debug("bcra_rem: already on disk — %s", filename)
        return True

    url = (
        f"{_BCRA_BASE}/Pdfs/PublicacionesEstadisticas/"
        f"relevamiento-expectativas-mercado-{mes}-{year}.pdf"
    )
    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, headers=_HEADERS_BCRA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("bcra_rem: HTTP %d para %s-%s", resp.status_code, mes, year)
            return False
        if resp.content[:4] != b"%PDF":
            logger.warning("bcra_rem: respuesta no es PDF para %s-%s", mes, year)
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        logger.info("bcra_rem: saved %s (%d bytes)", filename, len(resp.content))
        return True

    except Exception as exc:
        logger.warning("bcra_rem: excepción para %s-%s — %s", mes, year, exc)
        return False


def _download_fomc(date_str: str, out_dir: Path) -> bool:
    """
    Download one FOMC minutes page as .txt (HTML content).

    date_str : YYYYMMDD (e.g. '20240131')
    out_dir  : meeting-month folder

    Returns True if file is on disk after the call.
    """
    filename = f"fed_fomc_{date_str}.txt"
    out_path = out_dir / filename
    if out_path.exists():
        logger.debug("fed_fomc: already on disk — %s", filename)
        return True

    url = f"{_FED_BASE}/monetarypolicy/fomcminutes{date_str}.htm"
    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, headers=_HEADERS_FED, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("fed_fomc: HTTP %d para %s", resp.status_code, date_str)
            return False
        if b"html" not in resp.content[:64].lower():
            logger.warning("fed_fomc: respuesta inesperada para %s", date_str)
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        logger.info("fed_fomc: saved %s (%d bytes)", filename, len(resp.content))
        return True

    except Exception as exc:
        logger.warning("fed_fomc: excepción para %s — %s", date_str, exc)
        return False
