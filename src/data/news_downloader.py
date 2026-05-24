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


# ─── Public API ───────────────────────────────────────────────────────────────

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
