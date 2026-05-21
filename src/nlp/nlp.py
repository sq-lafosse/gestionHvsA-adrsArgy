"""
NLP sentiment scoring for gestionHvsA-adrsArgy.

Reads monthly news files (PDF and TXT) from data/news/{year}/{month:02d}/
and returns a sentiment score in [-1, 1] using ProsusAI/finbert.

Design decisions (D30–D36):
  D30: pdfminer.six for PDF extraction; open() for TXT (utf-8, latin-1 fallback).
  D31: transformers.pipeline("text-classification", "ProsusAI/finbert").
  D32: Tokenize → chunk at 450 tokens with 50-token overlap → score each chunk.
  D33: score per chunk = P(positive) – P(negative); monthly score = mean over docs.
  D34: Lazy singleton at module level; injectable via _pipeline for tests.
  D35: Missing or empty news dir → return 0.0 + logger.warning (neutral placeholder).
  D36: Single nlp.py + __init__.py.

Both pdfminer and transformers are imported lazily so the module loads even if
those packages are not yet installed; failures surface at first actual call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 450   # max tokens per segment fed to FinBERT
_CHUNK_STEP = 400   # stride between segments (overlap = 450 - 400 = 50 tokens)

_pipeline_instance: Any = None


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_monthly_sentiment(
    year: int,
    month: int,
    base_dir: Path | str = "data/news",
    _pipeline: Any = None,
) -> float:
    """
    Compute the aggregated FinBERT sentiment score for a calendar month.

    Reads all .txt and .pdf files in {base_dir}/{year}/{month:02d}/.
    Each document is tokenized and split into overlapping 450-token segments;
    the segment scores are averaged to produce a per-document score, then all
    document scores are averaged to produce the monthly score.

    Returns a float in [-1, 1]: positive = bullish, negative = bearish.
    Returns 0.0 (neutral) when the directory is absent or contains no readable
    files — consistent with the placeholder in predict_regime().

    Parameters
    ----------
    year, month : calendar period to score
    base_dir    : root of the news corpus (default: "data/news")
    _pipeline   : pre-loaded transformers pipeline override (for tests)

    Returns
    -------
    float in [-1, 1]
    """
    news_dir = Path(base_dir) / str(year) / f"{month:02d}"

    if not news_dir.exists():
        logger.warning(
            "News directory not found: %s — returning 0.0 (neutral)", news_dir
        )
        return 0.0

    documents = _load_documents(news_dir)

    if not documents:
        logger.warning(
            "No readable documents in %s — returning 0.0 (neutral)", news_dir
        )
        return 0.0

    pipe = _get_pipeline(_pipeline)
    doc_scores: list[float] = []

    for i, text in enumerate(documents, start=1):
        score = _score_document(text, pipe)
        doc_scores.append(score)
        logger.debug("Doc %d/%d — score=%.4f", i, len(documents), score)

    monthly_score = sum(doc_scores) / len(doc_scores)
    logger.info(
        "Monthly sentiment %d-%02d — score=%.4f  docs=%d",
        year, month, monthly_score, len(doc_scores),
    )
    return monthly_score


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

def _load_documents(news_dir: Path) -> list[str]:
    """Load all .txt and .pdf files from news_dir; skip unreadable files."""
    texts: list[str] = []
    for path in sorted(news_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".txt":
                text = _extract_txt(path)
            elif suffix == ".pdf":
                text = _extract_pdf(path)
            else:
                continue
        except Exception as exc:
            logger.warning("Skipping %s — extraction failed: %s", path.name, exc)
            continue
        cleaned = text.strip()
        if cleaned:
            texts.append(cleaned)
    logger.info("Loaded %d document(s) from %s", len(texts), news_dir)
    return texts


def _extract_txt(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path.name} as utf-8 or latin-1")


def _extract_pdf(path: Path) -> str:
    from pdfminer.high_level import extract_text  # noqa: PLC0415
    return extract_text(str(path))


# ─── Scoring ──────────────────────────────────────────────────────────────────

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
