"""Turns -> work units.

A session covers several pieces of work. One record per session is too coarse;
one per turn is noise. Group consecutive turns whose touched-file sets overlap,
and end a segment when the file set shifts substantially.

Deterministic, tunable, no model call. A wrong segmentation costs a slightly
noisy record, not a wrong one.

Two details do most of the work:

* Overlap is measured against a sliding window of recent file-bearing turns,
  not the segment's whole union. Otherwise one file touched early keeps
  everything after it "overlapping" and the session never splits.
* A shift has to persist for `boundary_misses` turns before it counts. A single
  divergent turn — running the test file, reading a doc — is part of the same
  work, and splitting on it would cut every edit/test cycle in half.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import timedelta

from agentlog.core.config import SegmentationConfig
from agentlog.domains.transcript.schemas import Segment, Transcript, Turn

# Whether a check failed or passed has to be read out of the text, not just the
# tool's error flag. A pipeline like `pytest | tail` exits 0 and reports no
# error while printing five failures, and that is a normal way to run a test
# suite — so the flag alone misses exactly the runs that matter.
_FAILURE_TEXT = re.compile(
    r"(?m)^\s*FAILED\b|\b\d+ failed\b|Traceback \(most recent call last\)|^\s*E\s+\w*Error\b"
)
_SUCCESS_TEXT = re.compile(
    r"(?m)\b\d+ passed\b|All checks passed|\bBuild succeeded\b|\b0 errors?\b"
)


def _union(entries) -> set[str]:
    files: set[str] = set()
    for entry in entries:
        files |= entry
    return files


def _shows_failure(turn: Turn) -> bool:
    if turn.had_error:
        return True
    return any(_FAILURE_TEXT.search(result.text) for result in turn.tool_results)


def _shows_success(turn: Turn) -> bool:
    """A clean verification. Checked only after `_shows_failure` says no.

    `5 failed, 135 passed` matches both patterns, and it is a failure.
    """
    return any(_SUCCESS_TEXT.search(result.text) for result in turn.tool_results)


def _build(
    session_id: str,
    turns: list[Turn],
    cfg: SegmentationConfig,
) -> Segment | None:
    if len(turns) < cfg.min_turns:
        return None

    files: dict[str, None] = {}
    for turn in turns:
        for path in turn.paths:
            files[path] = None

    if cfg.require_files and not files:
        # No anchors means nothing could ever retrieve this record. An
        # unreachable record is worse than no record: it costs a model call and
        # a line in the log and can never be returned.
        return None

    stamps = [turn.timestamp for turn in turns if turn.timestamp]
    errors = sum(1 for turn in turns if turn.had_error)

    return Segment(
        session_id=session_id,
        start_turn=turns[0].index,
        end_turn=turns[-1].index,
        turns=tuple(turns),
        files=tuple(files),
        started_at=min(stamps) if stamps else None,
        ended_at=max(stamps) if stamps else None,
        tool_errors=errors,
    )


def segment(transcript: Transcript, cfg: SegmentationConfig | None = None) -> list[Segment]:
    cfg = cfg or SegmentationConfig()
    segments: list[Segment] = []

    # Turns confirmed to belong to the segment being built.
    current: list[Turn] = []
    # Turns in an unresolved run of non-overlapping file sets. They join
    # `current` if the run turns out to be a detour, and become the seed of the
    # next segment if it turns out to be a shift.
    pending: list[Turn] = []
    recent: deque[frozenset[str]] = deque(maxlen=cfg.window)
    pending_files: list[frozenset[str]] = []

    def flush() -> None:
        nonlocal current
        built = _build(transcript.session_id, current, cfg)
        if built is not None:
            segments.append(built)
        current = []

    def accept(turn: Turn, files: frozenset[str]) -> None:
        current.extend(pending)
        pending.clear()
        for entry in pending_files:
            recent.append(entry)
        pending_files.clear()
        if files:
            recent.append(files)
        current.append(turn)

    def last_turn() -> Turn | None:
        if pending:
            return pending[-1]
        return current[-1] if current else None

    def hard_boundary(turn: Turn) -> bool:
        previous = last_turn()
        if previous is None:
            return False
        if len(current) + len(pending) >= cfg.max_turns:
            return True
        if turn.timestamp and previous.timestamp:
            return turn.timestamp - previous.timestamp > timedelta(minutes=cfg.max_gap_minutes)
        return False

    # True once a check has failed and nothing has verified clean since. While
    # it holds, the file-set rule cannot end the segment — the fix and its
    # confirming run belong with the failure that caused them.
    unresolved = False

    for turn in transcript.turns:
        if hard_boundary(turn):
            current.extend(pending)
            pending.clear()
            pending_files.clear()
            flush()
            recent.clear()
            unresolved = False

        if _shows_failure(turn):
            unresolved = True
        elif unresolved and _shows_success(turn):
            unresolved = False

        files = frozenset(turn.paths)
        if not files:
            # Turns that touch nothing (plain text, a question, a search)
            # belong to whatever work is already in progress.
            (pending if pending else current).append(turn)
            continue

        window = _union(recent)
        if not window or (files & window):
            accept(turn, files)
            continue

        if cfg.hold_open_on_failure and unresolved:
            # A shift away from the failing file is usually the fix, not new
            # work. Absorb it; `max_turns` and the time gap still bound this.
            accept(turn, files)
            continue

        pending.append(turn)
        pending_files.append(files)
        if len(pending_files) < cfg.boundary_misses:
            continue

        # The shift persisted. The run itself starts the next segment, so the
        # first divergent turn is not stranded at the end of the old one.
        flush()
        current = list(pending)
        pending.clear()
        recent.clear()
        for entry in pending_files:
            recent.append(entry)
        pending_files.clear()

    current.extend(pending)
    flush()
    return segments
