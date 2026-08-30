"""Structured (JSON-lines) logging.

Cloud platforms (Vercel, ECS, etc.) capture stdout and index it as logs
-- emitting one JSON object per line means those platforms' log search
works on fields (provider, tier, cost_usd, ...) instead of grepping
free-text messages. Pass structured fields via `extra={"fields": {...}}`
on the usual logging calls.
"""

from __future__ import annotations

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("llm_gateway")
    if logger.handlers:
        return  # idempotent -- avoid duplicate handlers across re-init/reload
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
