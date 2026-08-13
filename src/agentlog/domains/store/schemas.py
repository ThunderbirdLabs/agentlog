"""The record — the one thing that is actually persisted.

A record is written once and never edited. Corrections append a new record
carrying `supersedes`; the back-reference is computed at read time so the log
stays append-only and a `git diff` of it only ever grows.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from agentlog.domains.anchors.schemas import Anchors
from agentlog.domains.extraction.schemas import Candidate

SCHEMA_VERSION = 1


def new_id() -> str:
    return f"rec_{secrets.token_hex(10)}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SegmentRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_turn: int
    end_turn: int


class Record(BaseModel):
    """One extracted record, keyed to computed anchors.

    Field order here is the field order on disk. It is stable on purpose: the
    log is meant to be read in a pull request, and a reviewer should see the
    same shape on every line.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=new_id)
    # When the work happened. Timelines sort on this. Backfilling a month of
    # history has to place records where they belong in time, not stamp them
    # all with the moment the backfill ran.
    occurred_at: datetime
    # When the record was written. Only useful for auditing the tool itself.
    created_at: datetime = Field(default_factory=_now)
    session_id: str
    segment: SegmentRef

    anchors: Anchors

    kind: str
    outcome: str | None = None
    summary: str
    detail: str = ""

    evidence: str = "none"
    source: str = "stated"
    confidence: str = "medium"

    extractor: str
    supersedes: str | None = None

    @classmethod
    def build(
        cls,
        candidate: Candidate,
        anchors: Anchors,
        session_id: str,
        start_turn: int,
        end_turn: int,
        extractor: str,
        occurred_at: datetime | None = None,
    ) -> Record:
        return cls(
            occurred_at=occurred_at or _now(),
            session_id=session_id,
            segment=SegmentRef(start_turn=start_turn, end_turn=end_turn),
            anchors=anchors,
            kind=candidate.kind,
            outcome=candidate.outcome,
            summary=candidate.summary,
            detail=candidate.detail,
            evidence=candidate.evidence,
            source=candidate.source,
            confidence=candidate.confidence,
            extractor=extractor,
        )

    @property
    def is_dead_end(self) -> bool:
        """The records that stop a loop: a real attempt that really failed."""
        return self.kind == "attempt" and self.outcome == "failed" and self.evidence != "none"
