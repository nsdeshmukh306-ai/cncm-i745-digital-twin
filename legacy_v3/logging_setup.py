"""
Centralised logging configuration for the CNCM I-745 Digital Twin (v4.0.0).

A single rotating file handler (logs/digital_twin.log, 10 MB × 5 backups) is
shared by every layer module and the FastAPI backend so that all components
write to one structured log. Importing this module is idempotent: the handler
is attached exactly once regardless of how many modules call ``get_logger``.

Log format
----------
    [%(asctime)s] [%(levelname)s] [%(name)s] — %(message)s

Reference
---------
Standard library ``logging`` with ``logging.handlers.RotatingFileHandler``
(Python Software Foundation, Python 3.12 documentation).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
LOG_DIR    = BASE_DIR / "logs"
LOG_FILE   = LOG_DIR / "digital_twin.log"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] — %(message)s"
DATE_FMT   = "%Y-%m-%d %H:%M:%S"
MAX_BYTES  = 10 * 1024 * 1024          # 10 MB rotation threshold
BACKUP_CNT = 5

ROOT_NAME = "digital_twin"             # all project loggers live under this name
_configured = False


def _configure_root() -> logging.Logger:
    """Attach the rotating file handler to the project root logger once."""
    global _configured
    root = logging.getLogger(ROOT_NAME)
    if _configured:
        return root

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root.setLevel(logging.INFO)

    # Guard against duplicate handlers if this runs under reloaders / re-imports.
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler(
            LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_CNT, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT))
        root.addHandler(fh)

    # Do not double-emit through the global root logger.
    root.propagate = False
    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger (e.g. ``digital_twin.api.main``)."""
    _configure_root()
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def tail_log(n: int = 100) -> list[str]:
    """Return the last ``n`` lines of the shared log file (newest last)."""
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()[-n:]
