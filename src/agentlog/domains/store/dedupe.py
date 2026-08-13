"""Never pay for the same segment twice.

Hooks fire repeatedly. `PreCompact` fires many times in a long session and its
window overlaps the eventual `SessionEnd`. Re-running the pipeline over a
processed session must produce zero new records and cost zero model calls.

Two mechanisms, because one is not enough:

* a **cursor** per session, recording the last turn processed — handles the
  normal incremental path
* a **segment hash** over `(session_id, start_turn, end_turn)` — catches
  replays, manual re-runs, and backfill overlapping live capture

The hash set cannot be derived from the log, because a segment that yielded
zero records leaves no trace there. Without it, every uninteresting segment
would be re-extracted on every run — the most expensive possible way to learn
nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agentlog.core.logging import get_logger
from agentlog.domains.transcript.schemas import Segment

log = get_logger("store.dedupe")

CURSORS_NAME = "cursors.json"


def cursors_path(data_dir: Path) -> Path:
    return data_dir / CURSORS_NAME


def segment_hash(segment: Segment) -> str:
    raw = f"{segment.session_id}:{segment.start_turn}:{segment.end_turn}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class Cursors:
    """Per-session progress. Small enough to rewrite whole on every save."""

    def __init__(self, data: dict[str, dict]) -> None:
        self._data = data

    @classmethod
    def load(cls, data_dir: Path) -> Cursors:
        path = cursors_path(data_dir)
        if not path.is_file():
            return cls({})
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Losing the cursor costs a re-extraction, not correctness. Better
            # to start clean than to abort a capture run.
            log.warning("could not read %s (%s); starting from scratch", path, exc)
            return cls({})
        if not isinstance(raw, dict):
            return cls({})
        return cls(raw)

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = cursors_path(data_dir)
        # Write-then-rename: a crash mid-write leaves the old cursor intact
        # rather than a truncated file that reads as "nothing processed".
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _session(self, session_id: str) -> dict:
        entry = self._data.setdefault(session_id, {})
        entry.setdefault("last_turn", -1)
        entry.setdefault("segments", [])
        return entry

    def last_turn(self, session_id: str) -> int:
        return int(self._session(session_id)["last_turn"])

    def seen(self, segment: Segment) -> bool:
        entry = self._session(segment.session_id)
        return segment_hash(segment) in set(entry["segments"])

    def mark(self, segment: Segment) -> None:
        entry = self._session(segment.session_id)
        digest = segment_hash(segment)
        if digest not in entry["segments"]:
            entry["segments"].append(digest)
        entry["last_turn"] = max(int(entry["last_turn"]), segment.end_turn)

    def unprocessed(self, segments: list[Segment]) -> list[Segment]:
        """Segments that still need extracting.

        The cursor is the fast path; the hash is the correctness check. A
        segment is only skipped when it ends at or before the cursor *and* its
        hash has been seen — a segment that grew because more turns arrived
        after a compaction has a new end turn, so it is correctly treated as
        new work.
        """
        pending = []
        for segment in segments:
            if self.seen(segment):
                continue
            if segment.end_turn <= self.last_turn(segment.session_id):
                continue
            pending.append(segment)
        return pending

    @property
    def sessions(self) -> dict[str, dict]:
        return self._data
