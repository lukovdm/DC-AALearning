"""Unified project logger with per-line elapsed time.

Usage:
    from lsharpsat.logger import setup_logger, get_logger
    log = setup_logger(level="INFO")
    log.info("Starting up")

Features:
- Console logging by default
- Optional file logging (append or overwrite)
- Elapsed time (+delta seconds) since previous log message per handler
- Idempotent setup (safe to call multiple times)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import sys
import time
from typing import Optional

__all__ = ["setup_logger", "get_logger"]


@dataclass(slots=True)
class _DeltaState:
    last_time: float | None = None


class _DeltaFormatter(logging.Formatter):
    default_fmt = "%(asctime)s | %(levelname)-8s | %(delta)+.3fs | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    def __init__(self, fmt: Optional[str] = None):
        super().__init__(fmt or self.default_fmt, datefmt=self.datefmt)
        self._state = _DeltaState()

    def format(self, record: logging.LogRecord) -> str:
        now = record.created
        delta = 0.0 if self._state.last_time is None else now - self._state.last_time
        self._state.last_time = now
        record.delta = delta  # type: ignore[attr-defined]
        try:
            return super().format(record)
        finally:
            if hasattr(record, "delta"):
                delattr(record, "delta")


def setup_logger(
    name: str = "lsharpsat",
    level: int | str = logging.INFO,
    log_file: str | None = None,
    overwrite: bool = False,
    propagate: bool = False,
) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger(name)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = propagate

    if not getattr(logger, "_unified_logger_configured", False):
        # Clear existing handlers (avoid duplicates when re-running notebooks, etc.)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(_DeltaFormatter())
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        logger._unified_logger_configured = True  # type: ignore[attr-defined]
    else:
        for h in logger.handlers:
            h.setLevel(level)

    if log_file:
        abs_path = os.path.abspath(log_file)
        need_new = True
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == abs_path:  # type: ignore[attr-defined]
                need_new = False
                break
        if need_new:
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            mode = "w" if overwrite else "a"
            fh = logging.FileHandler(abs_path, mode=mode, encoding="utf-8")
            fh.setFormatter(_DeltaFormatter())
            fh.setLevel(level)
            logger.addHandler(fh)
            logger.debug(f"Added file handler at {abs_path} (mode={mode})")

    return logger


def get_logger(name: str = "lsharpsat") -> logging.Logger:
    return logging.getLogger(name)


if __name__ == "__main__":  # simple smoke test
    log = setup_logger(level="DEBUG")
    log.info(f"Logger initialized")
    time.sleep(0.1)
    log.debug(f"Second line")
    time.sleep(0.05)
    log.warning(f"Example warning")
