"""
NLP sentiment scoring for gestionHvsA-adrsArgy.

Reads monthly documents from data/news/{year}/{month:02d}/ and returns
two separate FinBERT scores using ProsusAI/finbert:

    macro_score      — aggregated score from macro documents (BCRA, INDEC, FOMC)
    company_scores   — per-ticker score from SEC 6-K/20-F filings

Partition rule (from filename):
    Company doc: filename contains one of the 10 known ticker symbols.
                 e.g. sec_YPF_2024-01-15_abc1234.txt
    Macro doc:   all others (bcra_*, indec_*, fed_fomc_*)

Design decisions (D30–D36):
  D30: pdfminer.six for PDF extraction; open() for TXT (utf-8, latin-1 fallback).
  D31: transformers.pipeline("text-classification", "ProsusAI/finbert").
  D32: Tokenize → chunk at 450 tokens with 50-token overlap → score each chunk.
  D33: score per chunk = P(positive) – P(negative); doc score = mean over chunks.
  D34: Lazy singleton at module level; injectable via _pipeline for tests.
  D35: Missing or empty directory → 0.0 scores (neutral placeholder).
  D36: Single nlp.py + __init__.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 450
_CHUNK_STEP = 400

_TICKERS = {"YPF", "GGAL", "BMA", "PAM", "TGS", "CEPU", "EDN", "LOMA", "CRESY", "IRS"}

_pipeline_instance: Any = None


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_monthly_sentiment(
    year: int,
    month: int,
    base_dir: Path | str = "data/news",
    _pipeline: Any = None,
) -> dict[str, Any]:
    """
    Compute FinBERT sentiment scores for a calendar month, split by document type.

    Documents are partitioned by filename:
      - Company docs: filename contains a known ticker (e.g. 'sec_YPF_...')
      - Macro docs:   all others (BCRA, INDEC Informa, Fed FOMC)

    Parameters
    ----------
    year, month : calendar period to score
    base_dir    : root of the news corpus (default: "data/news")
    _pipeline   : pre-loaded transformers pipeline override (for tests)

    Returns
    -------
    {
        "macro_score": float,            # [-1, 1] — same for all assets
        "company_scores": {              # [-1, 1] per ticker
            "YPF": float, "GGAL": float, ...  (all 10 tickers)
        }
    }
    Returns 0.0 for any missing or empty group.
    """
    news_dir = Path(base_dir) / str(year) / f"{month:02d}"

    if not news_dir.exists():
        logger.warning(
            "News directory not found: %s — returning 0.0 for all scores", news_dir
        )
        return _neutral_result()

    macro_texts, company_texts = _partition_documents(news_dir)

    if not macro_texts and not any(company_texts.values()):
        logger.warning(
            "No readable documents in %s — returning 0.0 for all scores", news_dir
        )
        return _neutral_result()

    pipe = _get_pipeline(_pipeline)

    macro_score = _score_document_list(macro_texts, pipe, label=f"{year}-{month:02d} macro")

    company_scores: dict[str, float] = {}
    for ticker in _TICKERS:
        docs = company_texts.get(ticker, [])
        if docs:
            company_scores[ticker] = _score_document_list(
                docs, pipe, label=f"{year}-{month:02d} {ticker}"
            )
        else:
            logger.debug(
                "No documents for %s in %d-%02d — company_score=0.0", ticker, year, month
            )
            company_scores[ticker] = 0.0

    logger.info(
        "Monthly sentiment %d-%02d — macro=%.4f | company=[%s]",
        year, month, macro_score,
        " ".join(f"{t}={v:.3f}" for t, v in sorted(company_scores.items())),
    )
    return {"macro_score": macro_score, "company_scores": company_scores}


# ─── Partition ────────────────────────────────────────────────────────────────

def _partition_documents(
    news_dir: Path,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Load and partition all .txt and .pdf files from news_dir.

    Returns
    -------
    macro_texts   : list of document strings (macro docs)
    company_texts : dict ticker → list of document strings
    """
    macro_texts: list[str] = []
    company_texts: dict[str, list[str]] = {t: [] for t in _TICKERS}

    for path in sorted(news_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in (".txt", ".pdf"):
            continue

        try:
            text = _extract_txt(path) if suffix == ".txt" else _extract_pdf(path)
        except Exception as exc:
            logger.warning("Skipping %s — extraction failed: %s", path.name, exc)
            continue

        cleaned = text.strip()
        if not cleaned:
            continue

        name_upper = path.name.upper()
        ticker_match = next((t for t in _TICKERS if t in name_upper), None)
        if ticker_match:
            company_texts[ticker_match].append(cleaned)
        else:
            macro_texts.append(cleaned)

    macro_count = len(macro_texts)
    company_count = sum(len(v) for v in company_texts.values())
    logger.info(
        "Loaded from %s — macro=%d docs, company=%d docs across %d tickers",
        news_dir, macro_count, company_count,
        sum(1 for v in company_texts.values() if v),
    )
    return macro_texts, company_texts


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _score_document_list(docs: list[str], pipe: Any, label: str = "") -> float:
    """Score a list of documents and return their mean score."""
    if not docs:
        return 0.0
    scores = [_score_document(text, pipe) for text in docs]
    mean = sum(scores) / len(scores)
    if label:
        logger.debug("%s — docs=%d mean=%.4f", label, len(scores), mean)
    return mean


def _score_document(text: str, pipe: Any) -> float:
    """Score a single document: chunk → score each chunk → mean."""
    chunks = _chunk_tokens(text, pipe.tokenizer)
    if not chunks:
        return 0.0
    scores = [_score_chunk(chunk, pipe) for chunk in chunks]
    return sum(scores) / len(scores)


def _chunk_tokens(text: str, tokenizer: Any) -> list[str]:
    """
    Tokenize text and split into overlapping segments of at most _CHUNK_SIZE tokens.

    Uses the FinBERT tokenizer to ensure chunk boundaries respect subword tokens.
    Each segment is decoded back to a string for pipeline input.
    """
    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: list[str] = []
    for start in range(0, len(token_ids), _CHUNK_STEP):
        end = min(start + _CHUNK_SIZE, len(token_ids))
        decoded = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        if decoded.strip():
            chunks.append(decoded)
    return chunks


def _score_chunk(text: str, pipe: Any) -> float:
    """
    Run FinBERT on one text chunk and return P(positive) – P(negative).

    top_k=None requests all three label scores (positive / negative / neutral).
    Neutral probability is discarded; the result falls in (-1, 1).
    """
    result = pipe(text, truncation=True, max_length=512, top_k=None)
    scores = {item["label"]: item["score"] for item in result}
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


# ─── Model loading ────────────────────────────────────────────────────────────

def _get_pipeline(override: Any = None) -> Any:
    global _pipeline_instance
    if override is not None:
        return override
    if _pipeline_instance is None:
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        logger.info("Loading FinBERT (ProsusAI/finbert) on CPU — first call...")
        _pipeline_instance = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,
        )
        logger.info("FinBERT loaded.")
    return _pipeline_instance


# ─── Text extraction ──────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)[:5000]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path.name} as utf-8 or latin-1")


def _extract_pdf(path: Path) -> str:
    from pdfminer.high_level import extract_text  # noqa: PLC0415
    return extract_text(str(path))[:5000]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _neutral_result() -> dict[str, Any]:
    return {
        "macro_score":    0.0,
        "company_scores": {t: 0.0 for t in _TICKERS},
    }
