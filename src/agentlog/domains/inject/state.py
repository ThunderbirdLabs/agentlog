"""Which records a session has already been shown.

Hooks are separate processes, so "did I already inject this" cannot live in
memory. Without it the same record reappears on every turn that mentions the
same file, which is the fastest way to get an injection hook disabled.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentlog.core.logging import get_logger

log = get_logger("inject.state")

STATE_DIRNAME = "sessions"
# Enough to keep a long session from repeating itself, small enough that the
# file stays trivial to read and write on every turn.
MAX_REMEMBERED = 200


def _path(data_dir: Path, session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "unknown"
    return data_dir / STATE_DIRNAME / f"{safe}.json"


def seen(data_dir: Path, session_id: str) -> set[str]:
    path = _path(data_dir, session_id)
    if not path.is_file():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Losing this costs a repeated record, never correctness.
        return set()
    return set(raw) if isinstance(raw, list) else set()


def remember(data_dir: Path, session_id: str, ids: list[str]) -> None:
    if not ids:
        return
    path = _path(data_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = list(seen(data_dir, session_id))
    for record_id in ids:
        if record_id not in current:
            current.append(record_id)
    try:
        path.write_text(json.dumps(current[-MAX_REMEMBERED:]), encoding="utf-8")
    except OSError as exc:
        log.debug("could not write session state: %s", exc)
