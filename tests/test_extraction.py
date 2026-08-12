"""Extraction, with a mocked model client. No test here hits the network."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentlog.domains.anchors.schemas import Anchors
from agentlog.domains.extraction import prompts
from agentlog.domains.extraction.schemas import Candidate, ExtractionResponse
from agentlog.domains.extraction.service import extract, render

from .conftest import make_segment

ANCHORS = Anchors(files=("src/service.py",))


class FakeClient:
    """Returns a scripted sequence of responses and records the calls."""

    def __init__(self, *responses: ExtractionResponse | None) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.systems: list[str] = []

    def classify(self, system, user, schema):  # noqa: ANN001, ANN201
        self.calls += 1
        self.systems.append(system)
        if not self._responses:
            return None
        return self._responses.pop(0)


class ExplodingClient:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, system, user, schema):  # noqa: ANN001, ANN201
        self.calls += 1
        raise RuntimeError("network is down")


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_attempt_requires_an_outcome() -> None:
    with pytest.raises(ValidationError):
        Candidate(kind="attempt", summary="tried something")


def test_decision_must_not_carry_an_outcome() -> None:
    """A preference must never be able to read as a verified result."""
    with pytest.raises(ValidationError):
        Candidate(kind="decision", outcome="worked", summary="chose X")


def test_summary_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        Candidate(kind="note", summary="   ")


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Candidate(kind="issue", summary="x")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Candidate(kind="note", summary="x", severity="high")


def test_empty_records_is_valid() -> None:
    assert ExtractionResponse(records=[]).records == []


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def test_prompt_forbids_literals_and_keys() -> None:
    system = prompts.SYSTEM
    assert "Never reproduce code" in system
    assert "Never name a file, route, branch, commit, or test" in system
    assert "one sentence" in system
    assert "empty array" in system


def test_prompt_biases_toward_failed_attempts() -> None:
    assert "dead end" in prompts.SYSTEM
    assert "failed" in prompts.SYSTEM


def test_prompt_fences_the_slice_as_untrusted() -> None:
    assert "untrusted data" in prompts.SYSTEM
    assert "<session_slice>" in prompts.user_prompt("x")


def test_extractor_id_is_versioned() -> None:
    assert prompts.extractor_id("claude-haiku-4-5") == f"haiku-4.5/{prompts.PROMPT_VERSION}"
    assert (
        prompts.extractor_id("claude-haiku-4-5-20251001") == f"haiku-4.5/{prompts.PROMPT_VERSION}"
    )


# --------------------------------------------------------------------------
# service
# --------------------------------------------------------------------------


def test_render_includes_tool_calls_and_error_markers() -> None:
    segment = make_segment(texts=["first", "second"])
    text = render(segment)
    assert "-> Edit(src/app.py)" in text
    assert "[0]" in text


def test_render_elides_the_middle_of_an_oversized_segment() -> None:
    segment = make_segment(texts=["x" * 4000 for _ in range(10)])
    text = render(segment)
    assert "elided" in text
    assert len(text) < 14000


def test_records_are_returned_and_scrubbed() -> None:
    response = ExtractionResponse(
        records=[
            Candidate(
                kind="attempt",
                outcome="failed",
                summary="Streaming the response did not clear the proxy timeout.",
                detail="Set DB_PASSWORD=hunter2hunter2 while debugging.",
                evidence="test_failure",
                source="stated",
                confidence="high",
            )
        ]
    )
    client = FakeClient(response)
    result = extract(make_segment(), ANCHORS, client, "claude-haiku-4-5")

    assert not result.dropped
    assert len(result.candidates) == 1
    # Pass 2: whatever the model echoed back is scrubbed on the way out.
    assert "hunter2" not in result.candidates[0].detail
    assert result.extractor == f"haiku-4.5/{prompts.PROMPT_VERSION}"


def test_one_model_call_per_segment() -> None:
    client = FakeClient(ExtractionResponse(records=[]))
    extract(make_segment(), ANCHORS, client, "m")
    assert client.calls == 1


def test_a_null_response_retries_once_then_skips() -> None:
    client = FakeClient(None, None)
    result = extract(make_segment(), ANCHORS, client, "m")
    assert client.calls == 2
    assert result.dropped
    assert result.candidates == ()


def test_a_null_response_followed_by_a_good_one_succeeds() -> None:
    good = ExtractionResponse(records=[Candidate(kind="note", summary="a note")])
    client = FakeClient(None, good)
    result = extract(make_segment(), ANCHORS, client, "m")
    assert client.calls == 2
    assert not result.dropped
    assert len(result.candidates) == 1


def test_a_raising_client_does_not_end_the_run() -> None:
    client = ExplodingClient()
    result = extract(make_segment(), ANCHORS, client, "m")
    assert client.calls == 2
    assert result.dropped
    assert "no valid response" in (result.drop_reason or "")


def test_an_empty_response_is_a_normal_outcome() -> None:
    client = FakeClient(ExtractionResponse(records=[]))
    result = extract(make_segment(), ANCHORS, client, "m")
    assert not result.dropped
    assert result.candidates == ()
    assert result.drop_reason is None


def test_prompt_permits_technique_names_while_banning_identifiers() -> None:
    """The v1 rule was both violated and too blunt.

    Records must be able to say "streaming the response" — that is the whole
    content of a dead end — while still not naming files or tests.
    """
    system = prompts.SYSTEM
    assert "the ban is on identifiers, not on being specific" in system
    assert "Never name a file, route, branch, commit, or test" in system


def test_prompt_keeps_constraints_discovered_through_failure() -> None:
    """A failure fixed in the same slice still leaves a durable constraint."""
    assert "still a record" in prompts.SYSTEM
    assert "Record the constraint, not the repair" in prompts.SYSTEM


def test_prompt_forbids_referencing_the_slice_numbering() -> None:
    """v1 produced a record citing a line number of the prompt itself."""
    assert "numbering of this slice" in prompts.SYSTEM
