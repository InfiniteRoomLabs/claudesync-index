"""
Structured logging via structlog routed through stdlib logging so we can
multiplex handlers (stderr + rotating file).

Stderr format:
  human   ANSI-colored single-line, default when stderr is a TTY
  json    ndjson, default otherwise
Selected by $LOG_FORMAT.

File output (separate from stderr):
  Always JSONL regardless of stderr format.
  Path resolved by configure(log_file=...) > $CSINDEX_LOG_FILE > default.
  Rotates at 10MB, keeps 5 backups.

Levels: debug | info | warn | error.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def _select_format() -> str:
    fmt = os.environ.get("LOG_FORMAT")
    if fmt:
        return fmt
    return "human" if sys.stderr.isatty() else "json"


def _select_level(level: str | None) -> str:
    level = (level or os.environ.get("LOG_LEVEL") or "info").upper()
    if level == "WARN":
        level = "WARNING"
    return level


def configure(
    level: str | None = None,
    fmt: str | None = None,
    *,
    log_file: str | Path | None = None,
    no_log_file: bool = False,
) -> None:
    """Set up stderr + optional file handler. Idempotent — safe to call twice."""
    level_str = _select_level(level)
    fmt = fmt or _select_format()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    # Stderr renderer.
    if fmt == "json":
        stderr_renderer = structlog.processors.JSONRenderer()
    else:
        stderr_renderer = structlog.dev.ConsoleRenderer(
            colors=not bool(os.environ.get("NO_COLOR")),
            event_key="event",
        )

    root = logging.getLogger()
    # Drop any handlers we (or pytest) installed previously so reconfigure doesn't double-emit.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level_str))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=stderr_renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    root.addHandler(stderr_handler)

    if not no_log_file:
        path = _resolve_log_path(log_file)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=10_000_000, backupCount=5, encoding="utf-8",
            )
            file_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    processor=structlog.processors.JSONRenderer(),
                    foreign_pre_chain=shared_processors,
                )
            )
            root.addHandler(file_handler)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level_str)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def _resolve_log_path(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("CSINDEX_LOG_FILE")
    if env:
        return Path(env)
    # Default path lives next to the export root if known.
    export_root = os.environ.get("CSINDEX_ROOT")
    if export_root:
        return Path(export_root) / ".reindex.log.jsonl"
    return None


def get(component: str) -> structlog.stdlib.BoundLogger:
    """Get a logger bound to a component name.

    Call this inside functions, not at module top-level, so the wrapper class
    captured at bind time reflects the current configure() call.
    """
    return structlog.get_logger().bind(component=component)
