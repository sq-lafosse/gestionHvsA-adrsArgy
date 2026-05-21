"""
src/utils — logging, I/O helpers, and pipeline timer for gestionHvsA-adrsArgy.

Conventions:
  - setup_logging() configures root logger: console + file handlers simultaneously.
  - Log file: logs/run_{UTC_timestamp}.log (relative to project root by default).
  - Snapshots: snapshot_{timestamp}.json — reproducibility record for each run.
  - timer(): context manager that logs stage name and elapsed seconds.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ─── Public API ───────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_dir: Path | str = "logs",
) -> Path:
    """
    Configure root logger with console and file handlers.

    Both handlers share the same format. The log file is created at
    {log_dir}/run_{timestamp}.log where timestamp is UTC at call time.
    Calling this function more than once adds a new file handler each time
    (one per run); the console handler is added only once.

    Parameters
    ----------
    level   : logging level name ("DEBUG", "INFO", "WARNING", ...)
    log_dir : directory for log files (relative to project root or absolute)

    Returns
    -------
    Path to the log file created in this call.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Add console handler only once (FileHandler is a subclass of StreamHandler,
    # so use exact type check to avoid counting file handlers as console handlers).
    console_exists = any(type(h) is logging.StreamHandler for h in root.handlers)
    if not console_exists:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logger.info(
        "Logging initialized — level=%s  file=%s", level.upper(), log_path
    )
    return log_path


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if absent. Returns the resolved Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_snapshot(
    output_dir: Path | str,
    params: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """
    Write a reproducibility snapshot as snapshot_{timestamp}.json.

    Always includes a 'run_at' field (UTC ISO-8601). Non-serializable values
    (pd.Timestamp, Path, numpy scalars, etc.) are coerced to str via default=str.

    Parameters
    ----------
    output_dir : destination directory (created if absent)
    params     : model/pipeline parameters used in this run
    metadata   : runtime information (data versions, record counts, etc.)

    Returns
    -------
    Path to the saved JSON file.
    """
    output_dir = ensure_dir(output_dir)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"snapshot_{timestamp}.json"

    payload: dict[str, Any] = {
        "run_at":   datetime.now(tz=timezone.utc).isoformat(),
        "params":   params,
        "metadata": metadata,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)

    logger.info("Snapshot saved: %s", path)
    return path


@contextmanager
def timer(stage_name: str) -> Generator[None, None, None]:
    """
    Context manager that logs start and elapsed time for a pipeline stage.

    Usage:
        with timer("feature computation"):
            result = compute_features(...)
    """
    logger.info("[START] %s", stage_name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("[DONE]  %s — %.2fs", stage_name, elapsed)
