"""The append-only log. The source of truth.

`.agentlog/records.jsonl` — one JSON object per line, plain text. Git-diffable,
greppable, reviewable in a pull request, and still readable after this tool is
uninstalled. It is never rewritten.

Everything else in the store is derived from this file and can be deleted at
any time. If the index disagrees with the log, the log is right.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from agentlog.core.logging import get_logger
from agentlog.domains.store.schemas import Record

log = get_logger("store.log")

LOG_NAME = "records.jsonl"
GITIGNORE_NAME = ".gitignore"

# The log is committed; everything derived from it is not. It is plain text,
# one record per line, and diffs cleanly — so a teammate cloning the repo
# starts with the history instead of an empty file, and a record that looks
# wrong can be argued with in a pull request.
#
# Everything below is rebuildable from the log or specific to one machine.
_DEFAULT_GITIGNORE = """\
index.db
pending/
sessions/
cursors.json
"""


def log_path(data_dir: Path) -> Path:
    return data_dir / LOG_NAME


# What the first release wrote: ignore everything. Upgraded in place, because
# an installation that keeps ignoring the log silently never shares it.
_LEGACY_GITIGNORE = "*\n"


def ensure_dir(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    gitignore = data_dir / GITIGNORE_NAME
    if not gitignore.exists():
        gitignore.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
        return data_dir
    # Only replace the exact text we wrote ourselves. Anything else is the
    # user's, and they may have ignored the log deliberately.
    if gitignore.read_text(encoding="utf-8") == _LEGACY_GITIGNORE:
        gitignore.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
    return data_dir


def append(data_dir: Path, records: Iterable[Record]) -> int:
    """Append records. Returns how many were written.

    Opened in append mode and flushed per call: a crash mid-run costs the
    records still in memory, never the ones already on disk.
    """
    records = list(records)
    if not records:
        return 0
    ensure_dir(data_dir)
    path = log_path(data_dir)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=False))
            handle.write("\n")
    return len(records)


def read_all(data_dir: Path) -> Iterator[Record]:
    """Stream every record.

    A corrupt line is skipped with a warning rather than aborting the read. A
    half-written line from a killed process must not make the whole history
    unreadable.
    """
    path = log_path(data_dir)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 - one bad line, not a bad file
                log.warning("skipping unreadable record at %s:%d: %s", path, number, exc)


def superseded_ids(records: Iterable[Record]) -> set[str]:
    """Ids that some later record replaces.

    Computed at read time. Storing the back-reference would mean editing a
    record that has already been written.
    """
    return {record.supersedes for record in records if record.supersedes}


def count(data_dir: Path) -> int:
    return sum(1 for _ in read_all(data_dir))
