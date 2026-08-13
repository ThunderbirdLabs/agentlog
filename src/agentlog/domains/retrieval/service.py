"""Reading records back out.

Two queries cover most use:

    agentlog file src/features/inspections/service.py   timeline for a path
    agentlog search "export timeout"                    FTS across records

Ranking, in order:

1. anchor overlap — exact file, then route, then symbol, then setting, then issue
2. failed attempts with real evidence, boosted hardest — these are the records
   that stop the loop
3. recency
4. `source == inferred` excluded unless asked for
5. superseded records excluded
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agentlog.domains.store import index as index_module
from agentlog.domains.store import log as log_module
from agentlog.domains.store.schemas import Record

# A failed attempt backed by observable evidence is the whole product, so it
# outranks everything except an exact anchor match.
_DEAD_END_BOOST = 40.0
_ANCHOR_WEIGHT = {
    "file": 50.0,
    "route": 45.0,
    "symbol": 40.0,
    "setting": 45.0,
    "issue": 30.0,
    "branch": 10.0,
    "commit": 10.0,
}


@dataclass(frozen=True)
class Hit:
    record: Record
    score: float
    matched: tuple[str, ...]


def _load(data_dir: Path) -> tuple[dict[str, Record], set[str]]:
    records = list(log_module.read_all(data_dir))
    by_id = {record.id: record for record in records}
    return by_id, log_module.superseded_ids(records)


def _recency_bonus(record: Record, newest: float, oldest: float) -> float:
    if newest <= oldest:
        return 0.0
    age = (record.occurred_at.timestamp() - oldest) / (newest - oldest)
    return age * 10.0


def _rank(
    candidates: dict[str, list[str]],
    by_id: dict[str, Record],
    superseded: set[str],
    include_inferred: bool,
) -> list[Hit]:
    live = [
        (rid, matched)
        for rid, matched in candidates.items()
        if rid in by_id and rid not in superseded
    ]
    if not include_inferred:
        live = [(rid, m) for rid, m in live if by_id[rid].source == "stated"]
    if not live:
        return []

    stamps = [by_id[rid].created_at.timestamp() for rid, _ in live]
    newest, oldest = max(stamps), min(stamps)

    hits = []
    for rid, matched in live:
        record = by_id[rid]
        score = sum(_ANCHOR_WEIGHT.get(kind, 5.0) for kind in matched)
        if record.is_dead_end:
            score += _DEAD_END_BOOST
        score += _recency_bonus(record, newest, oldest)
        hits.append(Hit(record=record, score=score, matched=tuple(sorted(set(matched)))))
    hits.sort(key=lambda hit: (-hit.score, -hit.record.occurred_at.timestamp()))
    return hits


def by_anchor(
    data_dir: Path,
    kind: str,
    value: str,
    include_inferred: bool = False,
) -> list[Hit]:
    conn = index_module.ensure_current(data_dir)
    try:
        rows = conn.execute(
            "SELECT record_id FROM anchors WHERE kind = ? AND value = ?", (kind, value)
        ).fetchall()
    finally:
        conn.close()
    by_id, superseded = _load(data_dir)
    candidates: dict[str, list[str]] = {}
    for row in rows:
        candidates.setdefault(row["record_id"], []).append(kind)
    return _rank(candidates, by_id, superseded, include_inferred)


def timeline(data_dir: Path, path: str, include_inferred: bool = False) -> list[Hit]:
    """Every record touching a file, oldest first.

    Deliberately chronological rather than ranked. The question this answers is
    "what happened to this file, and in what order" — a ranked list would hide
    the very thing you came for, which is that it worked in June and does not
    work now.
    """
    hits = by_anchor(data_dir, "file", path, include_inferred)
    return sorted(hits, key=lambda hit: hit.record.occurred_at)


def search(
    data_dir: Path, query: str, include_inferred: bool = False, limit: int = 50
) -> list[Hit]:
    conn = index_module.ensure_current(data_dir)
    try:
        try:
            rows = conn.execute(
                """
                SELECT m.record_id AS record_id
                FROM records_fts f
                JOIN fts_map m ON m.rowid = f.rowid
                WHERE records_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # A bare FTS5 syntax error (an unbalanced quote, a stray operator)
            # should read as "no results", not as a crash in a hook.
            return []
    finally:
        conn.close()

    by_id, superseded = _load(data_dir)
    candidates = {row["record_id"]: ["text"] for row in rows}
    return _rank(candidates, by_id, superseded, include_inferred)


def get(data_dir: Path, record_id: str) -> Record | None:
    for record in log_module.read_all(data_dir):
        if record.id == record_id or record.id.endswith(record_id):
            return record
    return None
