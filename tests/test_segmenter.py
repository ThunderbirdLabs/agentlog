"""Segmentation is deterministic and file-set driven."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from agentlog.core.config import SegmentationConfig
from agentlog.domains.transcript import parser, segmenter
from agentlog.domains.transcript.schemas import Transcript

from .conftest import BASE_TIME as BASE
from .conftest import make_turn


def _transcript(turns) -> Transcript:
    return Transcript(session_id="s", path="x", turns=tuple(turns))


def test_a_file_set_shift_ends_a_segment() -> None:
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        make_turn(1, paths=("src/a.py",), minutes=1),
        make_turn(2, paths=("src/b.py",), minutes=2),
        make_turn(3, paths=("src/b.py",), minutes=3),
    ]
    segments = segmenter.segment(_transcript(turns))
    assert len(segments) == 2
    assert segments[0].files == ("src/a.py",)
    assert segments[1].files == ("src/b.py",)


def test_overlapping_files_stay_in_one_segment() -> None:
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        make_turn(1, paths=("src/a.py", "src/b.py"), minutes=1),
        make_turn(2, paths=("src/b.py",), minutes=2),
    ]
    segments = segmenter.segment(_transcript(turns))
    assert len(segments) == 1
    assert set(segments[0].files) == {"src/a.py", "src/b.py"}


def test_turns_touching_nothing_join_the_current_segment() -> None:
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        make_turn(1, text="thinking out loud", minutes=1),
        make_turn(2, paths=("src/a.py",), minutes=2),
    ]
    segments = segmenter.segment(_transcript(turns))
    assert len(segments) == 1
    assert segments[0].turn_count == 3


def test_a_long_gap_is_a_boundary_even_with_the_same_files() -> None:
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        make_turn(1, paths=("src/a.py",), minutes=1),
        make_turn(2, paths=("src/a.py",), minutes=400),
        make_turn(3, paths=("src/a.py",), minutes=401),
    ]
    segments = segmenter.segment(_transcript(turns))
    assert len(segments) == 2


def test_max_turns_caps_a_uniform_session() -> None:
    turns = [make_turn(i, paths=("src/a.py",), minutes=i) for i in range(25)]
    cfg = SegmentationConfig(max_turns=10)
    segments = segmenter.segment(_transcript(turns), cfg)
    assert len(segments) == 3
    assert all(seg.turn_count <= 10 for seg in segments)


def test_the_window_stops_one_common_file_gluing_the_session() -> None:
    """A file touched early must not keep everything after it "overlapping".

    Without the sliding window, `conftest.py` touched once at the start would
    make every later turn overlap and the whole session becomes one segment.
    """
    turns = [make_turn(0, paths=("tests/conftest.py",), minutes=0)]
    turns += [make_turn(i, paths=("src/a.py",), minutes=i) for i in range(1, 5)]
    turns += [make_turn(i, paths=("src/b.py",), minutes=i) for i in range(5, 9)]
    segments = segmenter.segment(_transcript(turns), SegmentationConfig(window=2))
    assert len(segments) >= 2
    assert segments[-1].files == ("src/b.py",)


def test_segments_with_no_files_are_dropped() -> None:
    """No anchors means nothing could ever retrieve the record."""
    turns = [make_turn(i, text="chatting", minutes=i) for i in range(4)]
    assert segmenter.segment(_transcript(turns)) == []


def test_min_turns_filters_noise() -> None:
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        make_turn(1, paths=("src/b.py",), minutes=1),
        make_turn(2, paths=("src/b.py",), minutes=2),
    ]
    segments = segmenter.segment(_transcript(turns), SegmentationConfig(min_turns=2))
    assert len(segments) == 1
    assert segments[0].files == ("src/b.py",)


def test_segmentation_is_deterministic(fixture_transcript: Path, git_repo: Path) -> None:
    transcript = parser.parse_file(fixture_transcript)
    first = segmenter.segment(transcript)
    second = segmenter.segment(transcript)
    assert [(s.start_turn, s.end_turn, s.files) for s in first] == [
        (s.start_turn, s.end_turn, s.files) for s in second
    ]


def test_fixture_splits_export_work_from_auth_work(
    fixture_transcript: Path, git_repo: Path
) -> None:
    transcript = parser.parse_file(fixture_transcript)
    known = {"src/service.py", "src/auth.py", "tests/test_export.py"}
    normalized = parser.normalize_paths(transcript, git_repo, known)
    segments = segmenter.segment(normalized)
    assert len(segments) == 2
    assert "src/service.py" in segments[0].files
    assert "src/auth.py" in segments[1].files
    assert segments[0].tool_errors == 2, "both timeout failures land in the export segment"


# --------------------------------------------------------------------------
# a failure and its resolution must stay in one segment
# --------------------------------------------------------------------------


def _result_turn(index: int, text: str, minutes: int, paths=()) -> object:
    from agentlog.domains.transcript.schemas import ToolCall, ToolResult, Turn

    return Turn(
        index=index,
        role="user",
        timestamp=BASE + timedelta(minutes=minutes),
        tool_calls=(
            (ToolCall(id=f"t{index}", name="Bash", paths=paths, text="Bash(pytest)"),)
            if paths
            else ()
        ),
        tool_results=(ToolResult(tool_use_id=f"t{index}", text=text),),
    )


def test_failure_and_its_fix_stay_together() -> None:
    """The case that broke a real timeline.

    A check fails against one file, the fix lands in a different file, and the
    confirming run comes later. Splitting on the file shift makes the outcome
    unknowable — the extractor sees a failure with no resolution.
    """
    turns = [
        make_turn(0, paths=("tests/test_x.py",), minutes=0),
        _result_turn(
            1, "FAILED tests/test_x.py::test_y\n5 failed, 135 passed", 1, ("tests/test_x.py",)
        ),
        make_turn(2, paths=("src/redaction.py",), minutes=2),
        make_turn(3, paths=("src/redaction.py",), minutes=3),
        _result_turn(4, "140 passed in 1.20s", 4, ("tests/test_x.py",)),
    ]
    segments = segmenter.segment(_transcript(turns))
    assert len(segments) == 1, "failure, fix and verification must land in one segment"
    body = " ".join(r.text for t in segments[0].turns for r in t.tool_results)
    assert "5 failed" in body and "140 passed" in body


def test_the_hold_releases_once_something_passes() -> None:
    """After a clean run, an ordinary file shift ends the segment again."""
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        _result_turn(1, "FAILED tests/test_a.py::test_one", 1, ("src/a.py",)),
        _result_turn(2, "12 passed in 0.4s", 2, ("src/a.py",)),
        make_turn(3, paths=("src/b.py",), minutes=3),
        make_turn(4, paths=("src/b.py",), minutes=4),
    ]
    segments = segmenter.segment(_transcript(turns))
    assert len(segments) == 2
    assert segments[1].files == ("src/b.py",)


def test_a_masked_exit_code_still_counts_as_failure() -> None:
    """`pytest | tail` exits 0 and sets no error flag while printing failures."""
    turns = [
        make_turn(0, paths=("src/a.py",), minutes=0),
        _result_turn(1, "5 failed, 135 passed in 1.12s", 1, ("src/a.py",)),
        make_turn(2, paths=("src/b.py",), minutes=2),
        make_turn(3, paths=("src/b.py",), minutes=3),
    ]
    assert segments_are_one(turns)


def segments_are_one(turns) -> bool:
    return len(segmenter.segment(_transcript(turns))) == 1


def test_a_never_resolved_failure_is_still_bounded_by_max_turns() -> None:
    """The hold must not be able to swallow a whole session."""
    turns = [_result_turn(0, "FAILED tests/test_a.py::test_one", 0, ("src/a.py",))]
    turns += [make_turn(i, paths=(f"src/f{i}.py",), minutes=i) for i in range(1, 20)]
    segments = segmenter.segment(_transcript(turns), SegmentationConfig(max_turns=8))
    assert len(segments) >= 2
    assert all(s.turn_count <= 8 for s in segments)


def test_hold_can_be_turned_off() -> None:
    turns = [
        make_turn(0, paths=("tests/test_x.py",), minutes=0),
        _result_turn(1, "5 failed, 135 passed", 1, ("tests/test_x.py",)),
        make_turn(2, paths=("src/redaction.py",), minutes=2),
        make_turn(3, paths=("src/redaction.py",), minutes=3),
    ]
    cfg = SegmentationConfig(hold_open_on_failure=False)
    assert len(segmenter.segment(_transcript(turns), cfg)) == 2
