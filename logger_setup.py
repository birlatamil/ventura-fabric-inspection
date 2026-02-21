"""
logger_setup.py
===============
Structured, rotating logging for the Fabric Width Measurement System.
- JSON-formatted file logs for machine ingestion
- Human-readable console output for development
"""

import logging
import logging.handlers
import json
import os
import time
from typing import Optional


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        obj = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "ms": int(record.msecs),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            obj.update(record.extra)
        return json.dumps(obj, ensure_ascii=False)


class _ColorConsoleFormatter(logging.Formatter):
    """ANSI-colored console formatter for human readability."""

    _COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"
    _FMT = "{color}[{levelname:8s}]{reset} {asctime} {name}: {message}"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = self._COLORS.get(record.levelname, "")
        fmt = self._FMT.format(
            color=color,
            levelname=record.levelname,
            reset=self._RESET,
            asctime=self.formatTime(record, datefmt="%H:%M:%S"),
            name=record.name,
            message=record.getMessage(),
        )
        if record.exc_info:
            fmt += "\n" + self.formatException(record.exc_info)
        return fmt


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "fabric_width.log",
    level: str = "INFO",
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 30,
    console: bool = True,
) -> logging.Logger:
    """
    Configure the root logger with rotating JSON file + optional console output.

    Returns the root logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any pre-existing handlers (idempotent re-call)
    root.handlers.clear()

    # ── Rotating JSON file handler ─────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_JSONFormatter())
    root.addHandler(file_handler)

    # ── Console handler ────────────────────────────────────────────────────
    if console:
        con_handler = logging.StreamHandler()
        con_handler.setLevel(numeric_level)
        con_handler.setFormatter(_ColorConsoleFormatter())
        root.addHandler(con_handler)

    root.info(
        "Logging initialised",
        extra={"log_path": log_path, "level": level},
    )
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger."""
    return logging.getLogger(name)
