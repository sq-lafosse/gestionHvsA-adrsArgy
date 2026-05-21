"""
src/utils — logging, I/O helpers, and pipeline timer for gestionHvsA-adrsArgy.

Public API:
    setup_logging  — configure root logger: console + file handlers simultaneously
    ensure_dir     — mkdir -p, returns Path
    save_snapshot  — write snapshot_{timestamp}.json for reproducibility
    timer          — context manager that logs stage name and elapsed time
"""
from .utils import ensure_dir, save_snapshot, setup_logging, timer

__all__ = ["setup_logging", "ensure_dir", "save_snapshot", "timer"]
