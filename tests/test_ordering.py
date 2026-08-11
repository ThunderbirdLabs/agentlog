"""Redaction must precede extraction — and the code must make it impossible to swap.

If someone reorders the pipeline so the model call happens first, one of these
fails. That is the whole point: the ordering is enforced by the type system,
not by a comment.
"""

from __future__ import annotations

import pytest

from agentlog.core.errors import RedactionError
from agentlog.core.redaction import Scrubbed, scrub
from agentlog.domains.anchors.schemas import Anchors
from agentlog.domains.extraction.schemas import ExtractionResponse
from agentlog.domains.extraction.service import extract
from agentlog.external.anthropic import _require_scrubbed

from .conftest import make_segment

_EMPTY_ANCHORS = Anchors()


def test_scrubbed_cannot_be_constructed_directly() -> None:
    with pytest.raises(RedactionError):
        Scrubbed("ANTHROPIC_API_KEY=sk-ant-api03-leak")


def test_scrubbed_cannot_be_constructed_with_a_forged_token() -> None:
    with pytest.raises(RedactionError):
        Scrubbed("secret", object())


def test_client_refuses_raw_text() -> None:
    with pytest.raises(TypeError):
        _require_scrubbed("postgres://u:p@h/db")


def test_client_accepts_scrubbed() -> None:
    value = scrub("hello")
    assert _require_scrubbed(value) is value


class _SpyClient:
    """Records exactly what the network layer was handed."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    def classify(self, system, user, schema):  # noqa: ANN001, ANN201
        self.seen.append(user)
        return ExtractionResponse(records=[])


def test_extraction_hands_the_client_only_scrubbed_text() -> None:
    client = _SpyClient()
    segment = make_segment(
        texts=[
            "the database is at postgres://svc:s3cr3tpw@db:5432/app",
            "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuv",
        ]
    )
    result = extract(segment, _EMPTY_ANCHORS, client, "claude-haiku-4-5")

    assert not result.dropped
    assert len(client.seen) == 1
    payload = client.seen[0]
    assert isinstance(payload, Scrubbed)
    assert "s3cr3tpw" not in payload.text
    assert "sk-ant-api03" not in payload.text


def test_a_redaction_failure_drops_the_segment_without_calling_the_model(monkeypatch) -> None:
    """Fail closed: a scrubber that raises must stop the segment, not pass it through."""
    import agentlog.domains.extraction.service as service

    def boom(_text: str) -> Scrubbed:
        raise RedactionError("simulated scrubber failure")

    monkeypatch.setattr(service, "scrub", boom)

    client = _SpyClient()
    result = service.extract(make_segment(texts=["anything"]), _EMPTY_ANCHORS, client, "m")

    assert result.dropped
    assert result.candidates == ()
    assert client.seen == [], "the model was called despite redaction failing"
