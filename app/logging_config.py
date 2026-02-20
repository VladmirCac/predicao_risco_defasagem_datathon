"""Configuração de logging estruturado para a API.

Usa apenas a biblioteca padrão `logging`, emitindo JSON em stdout. Todos
os campos extras passados em `logger.*(..., extra={...})` são mesclados
ao payload de log.
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from typing import Any, Dict

# Atributos padrão de LogRecord que não queremos repetir no JSON final.
RESERVED_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "stacklevel",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Formatter simples que serializa o LogRecord em JSON."""

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inclui campos extras adicionados via `extra={}`.
        for key, value in record.__dict__.items():
            if key in RESERVED_ATTRS or key.startswith("_"):
                continue
            base[key] = value

        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=True)


LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "app.logging_config.JsonFormatter",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
    "loggers": {
        # Mantém logs do uvicorn no mesmo formato.
        "uvicorn.error": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.access": {"level": "INFO", "handlers": ["default"], "propagate": False},
    },
}


def configure_logging() -> None:
    """Aplica a configuração de logging estruturado."""

    logging.config.dictConfig(LOGGING_CONFIG)

