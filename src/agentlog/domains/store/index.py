"""The derived index. Disposable by design.

SQLite with FTS5, rebuilt from `records.jsonl` at any time by `agentlog
reindex`. The log gives durability and review; the index gives keyword search
and anchor lookup, which a flat file cannot do past a few hundred records.

Because the index is derived, it can never disagree with history for long. If
it is wrong, delete it and rebuild.

No embeddings, no vector store. Missing a record leaves you where you already
are; returning a near-match tells an agent "this deadlocked here" about
different code, and it avoids a good approach for a fake reason. Poor recall
with perfect precision is the correct trade.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from agentlog.domains.store import log as log_module
from agentlog.domains.store.schemas import Record

INDEX_NAME = "index.db"

ANCHOR_KINDS = ("file", "route", "symbol", "setting", "commit", "issue", "branch")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id            TEXT PRIMARY KEY,
    occurred_at   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    start_turn    INTEGER NOT NULL,
    end_turn      INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    outcome       TEXT,
    evidence      TEXT NOT NULL,
    source        TEXT NOT NULL,
    confidence    TEXT NOT NULL,
    extractor     TEXT NOT NULL,
    supersedes    TEXT,
    summary       TEXT NOT NULL,
    detail        TEXT NOT NULL,
    branch        TEXT,
    issue         TEXT,
    head_sha      TEXT
);

CREATE TABLE IF NOT EXISTS anchors (
    record_id TEXT NOT NULL,
    kind      TEXT NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (record_id, kind, value)
);

CREATE INDEX IF NOT EXISTS anchors_lookup ON anchors (kind, value);
CREATE INDEX IF NOT EXISTS records_occurred ON records (occurred_at);

CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5 (
    summary,
    detail,
    anchor_text,
    content=''
);

CREATE TABLE IF NOT EXISTS fts_map (
    rowid     INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def index_path(data_dir: Path) -> Path:
    return data_dir / INDEX_NAME


def connect(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(index_path(data_dir))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _anchor_rows(record: Record) -> list[tuple[str, str, str]]:
    a = record.anchors
    rows: list[tuple[str, str, str]] = []
    for value in a.files:
        rows.append((record.id, "file", value))
    for value in a.routes:
        rows.append((record.id, "route", value))
    for value in a.symbols:
        rows.append((record.id, "symbol", value))
    for value in a.settings:
        rows.append((record.id, "setting", value))
    for value in a.commits:
        rows.append((record.id, "commit", value))
    if a.issue:
        rows.append((record.id, "issue", a.issue))
    if a.branch:
        rows.append((record.id, "branch", a.branch))
    return rows


def add(conn: sqlite3.Connection, records: Iterable[Record]) -> int:
    written = 0
    for record in records:
        conn.execute(
            """
            INSERT OR REPLACE INTO records
              (id, occurred_at, created_at, session_id, start_turn, end_turn,
               kind, outcome, evidence, source, confidence, extractor,
               supersedes, summary, detail, branch, issue, head_sha)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.id,
                record.occurred_at.isoformat(),
                record.created_at.isoformat(),
                record.session_id,
                record.segment.start_turn,
                record.segment.end_turn,
                record.kind,
                record.outcome,
                record.evidence,
                record.source,
                record.confidence,
                record.extractor,
                record.supersedes,
                record.summary,
                record.detail,
                record.anchors.branch,
                record.anchors.issue,
                record.anchors.head_sha,
            ),
        )
        rows = _anchor_rows(record)
        conn.executemany("INSERT OR IGNORE INTO anchors VALUES (?,?,?)", rows)

        # Anchors go into the FTS row too. Searching a setting name has to work
        # whether the extractor wrote it into the summary or not — the anchor
        # is computed, the summary is not, so the anchor is the reliable half.
        anchor_text = " ".join(value for _id, _kind, value in rows)
        cursor = conn.execute(
            "INSERT INTO fts_map (record_id) VALUES (?) ON CONFLICT(record_id) DO NOTHING",
            (record.id,),
        )
        if cursor.rowcount:
            rowid = cursor.lastrowid
        else:
            existing = conn.execute(
                "SELECT rowid FROM fts_map WHERE record_id = ?", (record.id,)
            ).fetchone()
            rowid = existing["rowid"]
            conn.execute("DELETE FROM records_fts WHERE rowid = ?", (rowid,))
        conn.execute(
            "INSERT INTO records_fts (rowid, summary, detail, anchor_text) VALUES (?,?,?,?)",
            (rowid, record.summary, record.detail, anchor_text),
        )
        written += 1
    conn.commit()
    return written


def _log_fingerprint(data_dir: Path) -> str:
    """A cheap stand-in for "has the log changed".

    Size and mtime, not a line count. The log is append-only, so size alone is
    already a strong signal, and reading the whole file to count lines is
    exactly the O(history) cost this is here to avoid.
    """
    path = log_module.log_path(data_dir)
    try:
        stat = path.stat()
    except OSError:
        return "0:0"
    return f"{stat.st_size}:{int(stat.st_mtime_ns)}"


def _stamp(conn: sqlite3.Connection, data_dir: Path) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('log_fingerprint', ?)",
        (_log_fingerprint(data_dir),),
    )
    conn.commit()


def rebuild(data_dir: Path) -> int:
    """Drop the index and rebuild it from the log.

    The whole point of the log being the source of truth: this is always safe.
    """
    path = index_path(data_dir)
    if path.exists():
        path.unlink()
    conn = connect(data_dir)
    try:
        written = add(conn, log_module.read_all(data_dir))
        _stamp(conn, data_dir)
        return written
    finally:
        conn.close()


def ensure_current(data_dir: Path) -> sqlite3.Connection:
    """Open the index, rebuilding it if it has fallen behind the log.

    The freshness check is a stat call, not a line count. This runs on every
    query, including a per-turn hook, so it has to be O(1) no matter how long
    the history gets.
    """
    conn = connect(data_dir)
    row = conn.execute("SELECT value FROM meta WHERE key = 'log_fingerprint'").fetchone()
    if row is not None and row["value"] == _log_fingerprint(data_dir):
        return conn
    conn.close()
    rebuild(data_dir)
    return connect(data_dir)
