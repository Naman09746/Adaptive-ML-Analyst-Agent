# ama2/backend/app/utils/logging.py

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog

    HAS_STRUCTLOG = True
except Exception:  # pragma: no cover - optional dependency fallback
    structlog = None
    HAS_STRUCTLOG = False


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )

    if not HAS_STRUCTLOG:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_contextvars(**kwargs: Any) -> None:
    if HAS_STRUCTLOG:
        structlog.contextvars.bind_contextvars(**kwargs)


def get_logger(name: str):
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
