"""Structured logging helpers with per-request trace IDs.

Usage in endpoints:
    req_id = new_request_id()
    log.info("code_gen_start", request_id=req_id, model=model_choice)
"""

import logging
import json
import time
import uuid
from contextlib import contextmanager
from typing import Optional


def new_request_id() -> str:
    """Short unique ID for tracing a single user request across log lines."""
    return uuid.uuid4().hex[:10]


class _StructuredFormatter(logging.Formatter):
    """Emit each log record as a single JSON line for easy grep / ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # attach any extra fields passed via log.info("msg", extra={...})
        for key in ("request_id", "model", "quality", "duration_s",
                     "prompt_len", "code_len", "error", "endpoint"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info and record.exc_info[0]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Replace the root handler with a structured JSON formatter."""
    root = logging.getLogger()
    root.setLevel(level)

    # remove existing handlers to avoid duplication
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(_StructuredFormatter())
    root.addHandler(handler)


@contextmanager
def log_timing(logger: logging.Logger, label: str, request_id: Optional[str] = None):
    """Context manager that logs elapsed wall-clock time on exit."""
    start = time.perf_counter()
    yield
    elapsed = round(time.perf_counter() - start, 3)
    extra = {"duration_s": elapsed}
    if request_id:
        extra["request_id"] = request_id
    logger.info(f"{label} completed", extra=extra)
