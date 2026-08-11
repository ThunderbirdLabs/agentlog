"""Segment -> candidate records.

The only place in agentlog that calls a model. One call per segment.

Redaction runs here, before the client is touched, and the client's signature
refuses anything that has not been through it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentlog.core.errors import RedactionError
from agentlog.core.logging import get_logger
from agentlog.core.redaction import scrub, scrub_record_field
from agentlog.domains.anchors.schemas import Anchors
from agentlog.domains.extraction import prompts
from agentlog.domains.extraction.schemas import Candidate, ExtractionResponse
from agentlog.domains.transcript.schemas import Segment, Turn
from agentlog.external.anthropic import ModelClient

log = get_logger("extraction.service")

MAX_SEGMENT_CHARS = 12000


@dataclass(frozen=True)
class SegmentExtraction:
    segment: Segment
    anchors: Anchors
    candidates: tuple[Candidate, ...]
    extractor: str
    # True when the segment was dropped rather than extracted. Dropped
    # segments are reported, never silently swallowed.
    dropped: bool = False
    drop_reason: str | None = None


def _render_turn(turn: Turn) -> str:
    lines = [f"[{turn.index}] {'human' if turn.is_human else turn.role}:"]
    if turn.text:
        lines.append(f"  {turn.text}")
    for call in turn.tool_calls:
        lines.append(f"  -> {call.text}")
    for result in turn.tool_results:
        marker = "ERROR" if result.is_error else "ok"
        if result.text:
            lines.append(f"  <- [{marker}] {result.text}")
        else:
            lines.append(f"  <- [{marker}]")
    return "\n".join(lines)


def render(segment: Segment) -> str:
    """Flatten a segment into the text the extractor reads."""
    body = "\n".join(_render_turn(turn) for turn in segment.turns)
    if len(body) <= MAX_SEGMENT_CHARS:
        return body
    # Keep both ends: the start says what was attempted, the end says how it
    # landed. Eliding the middle loses the least.
    half = MAX_SEGMENT_CHARS // 2
    return f"{body[:half]}\n\n…[middle of segment elided]…\n\n{body[-half:]}"


def _clean(candidate: Candidate) -> Candidate:
    """Pass 2 redaction: scrub the model's own words on the way to the log."""
    return candidate.model_copy(
        update={
            "summary": scrub_record_field(candidate.summary),
            "detail": scrub_record_field(candidate.detail),
        }
    )


def extract(
    segment: Segment,
    anchors: Anchors,
    client: ModelClient,
    model: str,
) -> SegmentExtraction:
    """Extract candidate records from one segment.

    Returns a `SegmentExtraction` in every case. A drop is data — the caller
    reports it — not an exception to swallow.
    """
    extractor = prompts.extractor_id(model)

    def dropped(reason: str) -> SegmentExtraction:
        return SegmentExtraction(
            segment=segment,
            anchors=anchors,
            candidates=(),
            extractor=extractor,
            dropped=True,
            drop_reason=reason,
        )

    try:
        scrubbed = scrub(prompts.user_prompt(render(segment)))
    except RedactionError as exc:
        # Fail closed. Nothing leaves the machine and nothing is written.
        log.error(
            "redaction failed for session %s turns %d-%d; segment dropped: %s",
            segment.session_id,
            segment.start_turn,
            segment.end_turn,
            exc,
        )
        return dropped(f"redaction failed: {exc}")

    response: ExtractionResponse | None = None
    for attempt in (1, 2):
        try:
            response = client.classify(prompts.SYSTEM, scrubbed, ExtractionResponse)
        except Exception as exc:  # noqa: BLE001 - one bad segment must not end the run
            log.warning("extraction call failed (attempt %d): %s", attempt, exc)
            response = None
        if response is not None:
            break

    if response is None:
        log.warning(
            "no valid extraction for session %s turns %d-%d after retry; segment skipped",
            segment.session_id,
            segment.start_turn,
            segment.end_turn,
        )
        return dropped("model returned no valid response after one retry")

    try:
        candidates = tuple(_clean(candidate) for candidate in response.records)
    except RedactionError as exc:
        log.error("redaction failed on extracted records; segment dropped: %s", exc)
        return dropped(f"redaction failed on records: {exc}")

    return SegmentExtraction(
        segment=segment,
        anchors=anchors,
        candidates=candidates,
        extractor=extractor,
    )
