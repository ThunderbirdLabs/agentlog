"""Logging.

Everything goes to stderr. stdout is reserved: injection hooks write their
context block there, and CLI output is piped by users. A log line on stdout
would corrupt both.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.environ.get("AGENTLOG_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("agentlog %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("agentlog")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"agentlog.{name}")
