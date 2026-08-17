"""Mechanical checks on the records themselves.

Extraction quality is the part of this tool most likely to be quietly bad, and
the part hardest to judge by reading. A log can look reasonable line by line
and still be mostly restated events, or full of identifiers the anchors already
carry.

These checks are the tuning instrument for the prompt. They do not judge whether
a record is *true* — nothing here can — only whether it is the shape the design
asked for. When the numbers move the wrong way after a prompt change, that is
the signal to stop.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from agentlog.domains.store.schemas import Record

# A path or filename. The anchors already carry these, so a record naming one
# is spending words on something retrieval knows better.
_PATH = re.compile(
    r"(?:[\w.\-]+/)+[\w.\-]+|\b[\w.\-]+\.(?:py|ts|tsx|js|jsx|go|rs|rb|java|sql|ya?ml|toml|json|md|sh)\b"
)
_COMMIT = re.compile(r"\b(?:commit\s+)?[0-9a-f]{7,40}\b|\bPR\s*#?\d+\b|\b[A-Z]{2,10}-\d+\b")
_SLICE_REF = re.compile(r"\b(?:line|turn)s?\s+\d+", re.I)
# Work whose record git already keeps. Duplicating it is noise by design.
_GIT_VISIBLE = re.compile(
    r"\b(?:created?|opened?|merged?|pushed?|committed|squash(?:ed)?)\b.{0,30}"
    r"\b(?:PR|pull request|branch|commit)\b",
    re.I,
)


@dataclass
class Finding:
    record_id: str
    check: str
    detail: str


@dataclass
class Report:
    total: int = 0
    kinds: Counter = field(default_factory=Counter)
    findings: list[Finding] = field(default_factory=list)

    @property
    def dead_ends(self) -> int:
        return self.kinds["attempt/failed"]

    @property
    def dead_end_share(self) -> float:
        return self.dead_ends / self.total if self.total else 0.0

    @property
    def by_check(self) -> Counter:
        return Counter(f.check for f in self.findings)

    @property
    def clean(self) -> int:
        flagged = {f.record_id for f in self.findings}
        return self.total - len(flagged)


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def check(records: list[Record]) -> Report:
    report = Report(total=len(records))
    seen_summaries: dict[str, str] = {}

    for record in records:
        label = record.kind + (f"/{record.outcome}" if record.outcome else "")
        report.kinds[label] += 1
        body = f"{record.summary} {record.detail}"

        hit = _first(_PATH, body)
        if hit:
            report.findings.append(Finding(record.id, "names-a-path", hit))

        hit = _first(_COMMIT, body)
        if hit:
            report.findings.append(Finding(record.id, "names-a-commit-or-ticket", hit))

        hit = _first(_SLICE_REF, body)
        if hit:
            # These reference the numbering of the prompt the record came from,
            # which means nothing once the record is stored.
            report.findings.append(Finding(record.id, "references-slice-numbering", hit))

        hit = _first(_GIT_VISIBLE, record.summary)
        if hit:
            report.findings.append(Finding(record.id, "git-already-records-this", hit))

        if record.kind == "attempt" and record.evidence == "none":
            report.findings.append(
                Finding(record.id, "attempt-without-evidence", f"outcome={record.outcome}")
            )

        key = re.sub(r"[^a-z ]", "", record.summary.lower())[:70]
        if key and key in seen_summaries:
            report.findings.append(
                Finding(record.id, "near-duplicate", f"of {seen_summaries[key][:12]}")
            )
        else:
            seen_summaries[key] = record.id

    return report
