"""Locating and streaming session files.

Sessions live at `~/.claude/projects/<cwd-with-separators-as-hyphens>/<session-id>.jsonl`.
The project directory is derived from the hook's `cwd`, never guessed.

A missing transcript is a normal condition, not an error: sessions get cleaned
up, hooks fire for directories that were never captured, retention expires.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

from agentlog.core.errors import TranscriptError
from agentlog.core.logging import get_logger

log = get_logger("transcript.reader")

PROJECTS_ROOT = Path.home() / ".claude" / "projects"


def project_dir_name(cwd: str | Path) -> str:
    """Encode a working directory the way Claude Code names its project dir.

    Every path separator becomes a hyphen, including the leading one, so
    `/Users/x/proj` becomes `-Users-x-proj`.
    """
    resolved = str(Path(cwd))
    return resolved.replace(os.sep, "-")


def project_dir(cwd: str | Path, root: Path | None = None) -> Path:
    return (root or PROJECTS_ROOT) / project_dir_name(cwd)


def find_session(session_id: str, cwd: str | Path, root: Path | None = None) -> Path | None:
    """Return the transcript for a session, or None if it is not on disk."""
    candidate = project_dir(cwd, root) / f"{session_id}.jsonl"
    return candidate if candidate.is_file() else None


def list_sessions(cwd: str | Path, root: Path | None = None) -> list[Path]:
    """All transcripts for a working directory, newest first."""
    directory = project_dir(cwd, root)
    if not directory.is_dir():
        return []
    files = [p for p in directory.glob("*.jsonl") if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def list_all_sessions(root: Path | None = None) -> list[Path]:
    directory = root or PROJECTS_ROOT
    if not directory.is_dir():
        return []
    files = [p for p in directory.glob("*/*.jsonl") if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def recent_sessions(days: int, root: Path | None = None) -> list[Path]:
    cutoff = time.time() - days * 86400
    return [p for p in list_all_sessions(root) if p.stat().st_mtime >= cutoff]


def stream_lines(path: Path) -> Iterator[str]:
    """Yield raw lines from a transcript.

    Opened with `errors="replace"`: a decoding error on one line must not cost
    the rest of the file. Anything unreadable at the file level is a
    `TranscriptError` — the caller decides whether that is fatal.
    """
    if not path.is_file():
        raise TranscriptError(f"not a file: {path}")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line
    except OSError as exc:
        raise TranscriptError(f"could not read {path}: {exc}") from exc


def prune(days: int, root: Path | None = None) -> int:
    """Delete transcripts older than `days`, keyed off mtime.

    Returns the number removed. Failures to unlink individual files are logged
    and skipped: cleanup is best-effort and must never abort a capture run.
    """
    cutoff = time.time() - days * 86400
    removed = 0
    for path in list_all_sessions(root):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            log.warning("could not prune %s: %s", path, exc)
    return removed
