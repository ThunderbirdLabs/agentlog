"""What the model is allowed to return.

Validated before anything is written. A malformed response retries once, then
the segment is skipped with a warning. An unvalidated record is never written.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

Kind = Literal["attempt", "decision", "note"]
Outcome = Literal["worked", "failed", "partial"]
Evidence = Literal["test_failure", "error_output", "user_rejected", "agent_abandoned", "none"]
Source = Literal["stated", "inferred"]
Confidence = Literal["high", "medium", "low"]


class Candidate(BaseModel):
    """One extracted record, before anchors are attached.

    `kind` is deliberately three values. Issues and fixes have clear git
    signals already, and duplicating them is noise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Kind
    # Present only on attempts. A decision has no outcome, and inventing one
    # would let a preference read as a verified result.
    outcome: Outcome | None = None
    summary: str
    detail: str = ""
    # "The tests failed" and "the agent wandered off" are different signals.
    # Flattening them makes retrieval lie.
    evidence: Evidence = "none"
    # Did the transcript say it, or did the extractor infer it? Only `stated`
    # is injected by default; without this the tool becomes a hallucination
    # laundry.
    source: Source = "stated"
    confidence: Confidence = "medium"

    @model_validator(mode="after")
    def _check_outcome(self) -> Candidate:
        if self.kind == "attempt" and self.outcome is None:
            raise ValueError("an attempt must carry an outcome")
        if self.kind != "attempt" and self.outcome is not None:
            raise ValueError(f"{self.kind} records must not carry an outcome")
        if not self.summary.strip():
            raise ValueError("summary must not be empty")
        return self


class ExtractionResponse(BaseModel):
    """The model's whole reply.

    An empty list is correct and common — most segments contain nothing worth
    logging, and a model that always finds something is a model inventing
    things.
    """

    model_config = ConfigDict(extra="forbid")

    records: list[Candidate] = []
