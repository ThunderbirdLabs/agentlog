"""Parser behaviour, including the defensive paths."""

from __future__ import annotations

import json
from pathlib import Path

from agentlog.domains.transcript import parser

from .conftest import write_transcript


def test_parses_the_fixture(fixture_transcript: Path) -> None:
    transcript = parser.parse_file(fixture_transcript)
    assert transcript.session_id == "sess-export-timeout"
    assert transcript.cwd == "/tmp/agentlog-fixture-repo"
    assert len(transcript.turns) >= 18


def test_malformed_line_skips_that_line_and_continues(fixture_transcript: Path) -> None:
    """The fixture contains a deliberately broken JSON line partway through."""
    transcript = parser.parse_file(fixture_transcript)
    assert transcript.skipped_lines >= 1
    # Turns after the broken line are still present.
    assert any("background job" in turn.text for turn in transcript.turns)


def test_unknown_line_types_are_recorded_not_fatal(fixture_transcript: Path) -> None:
    transcript = parser.parse_file(fixture_transcript)
    assert "file-history-snapshot" in transcript.unknown_types


def test_human_turns_are_marked(fixture_transcript: Path) -> None:
    transcript = parser.parse_file(fixture_transcript)
    humans = [turn for turn in transcript.turns if turn.is_human]
    assert len(humans) == 2
    assert "export endpoint times out" in humans[0].text


def test_tool_errors_are_carried_through(fixture_transcript: Path) -> None:
    transcript = parser.parse_file(fixture_transcript)
    errored = [turn for turn in transcript.turns if turn.had_error]
    assert len(errored) == 2, "both timeout failures should survive parsing"


def test_thinking_is_captured_but_truncated(fixture_transcript: Path) -> None:
    transcript = parser.parse_file(fixture_transcript)
    assert any("[thinking]" in turn.text for turn in transcript.turns)


def test_empty_file_yields_no_turns(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    transcript = parser.parse_file(path)
    assert transcript.turns == ()


def test_unrecognised_shapes_do_not_raise(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "weird.jsonl",
        [
            {"type": "assistant"},  # no message
            {"type": "assistant", "message": {"content": "a string, not a list"}},
            {"type": "user", "message": {"content": []}},  # empty content
            {"type": "user", "message": {"content": "hello"}, "isMeta": True},  # meta
            {"nope": 1},  # no type at all
        ],
    )
    transcript = parser.parse_file(path)
    assert transcript.turns == ()
    assert transcript.skipped_lines == 5


def test_meta_turns_are_dropped(tmp_path: Path) -> None:
    """System-injected context would read to the extractor as instructions."""
    path = write_transcript(
        tmp_path / "meta.jsonl",
        [
            {
                "type": "user",
                "message": {"content": "<system-reminder>do X</system-reminder>"},
                "isMeta": True,
            },
            {
                "type": "user",
                "message": {"content": "actual human message"},
                "origin": {"kind": "human"},
            },
        ],
    )
    transcript = parser.parse_file(path)
    assert len(transcript.turns) == 1
    assert transcript.turns[0].text == "actual human message"


def test_normalize_paths_keeps_only_real_repo_files(tmp_path: Path, git_repo: Path) -> None:
    path = write_transcript(
        tmp_path / "paths.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {
                                "command": (
                                    "pytest src/service.py && cat /etc/hosts.allow "
                                    "&& tail /var/log/app.log && ls other/repo/thing.py"
                                )
                            },
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)
    known = {"src/service.py"}
    normalized = parser.normalize_paths(transcript, git_repo, known)
    assert normalized.turns[0].paths == ("src/service.py",)


def test_absolute_paths_become_repo_relative(tmp_path: Path, git_repo: Path) -> None:
    path = write_transcript(
        tmp_path / "abs.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Edit",
                            "input": {"file_path": str(git_repo / "src" / "service.py")},
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)
    normalized = parser.normalize_paths(transcript, git_repo, {"src/service.py"})
    assert normalized.turns[0].paths == ("src/service.py",)


def test_paths_outside_the_repo_are_dropped(tmp_path: Path, git_repo: Path) -> None:
    path = write_transcript(
        tmp_path / "outside.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "/etc/passwd.bak"},
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)
    normalized = parser.normalize_paths(transcript, git_repo, {"src/service.py"})
    assert normalized.turns[0].paths == ()


def test_directory_valued_path_param_is_not_a_file(tmp_path: Path, git_repo: Path) -> None:
    """`Glob(path="src")` names a directory, not a file anchor."""
    path = write_transcript(
        tmp_path / "glob.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Glob",
                            "input": {"path": "src", "pattern": "**/*"},
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)
    normalized = parser.normalize_paths(transcript, git_repo, {"src/service.py"})
    assert normalized.turns[0].paths == ()


def test_fixture_is_valid_jsonl_except_the_deliberate_break(fixture_transcript: Path) -> None:
    bad = 0
    for line in fixture_transcript.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad += 1
    assert bad == 1, "the fixture should contain exactly one intentionally broken line"


def test_strict_normalization_rejects_untracked_disk_files(tmp_path: Path, git_repo: Path) -> None:
    """Strict mode is how gitignored junk is kept out of the anchor set."""
    junk = git_repo / ".venv" / "lib"
    junk.mkdir(parents=True)
    (junk / "thing.py").write_text("x = 1", encoding="utf-8")

    path = write_transcript(
        tmp_path / "junk.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": ".venv/lib/thing.py"},
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)

    permissive = parser.normalize_paths(transcript, git_repo)
    assert permissive.turns[0].paths == (".venv/lib/thing.py",)

    strict = parser.normalize_paths(transcript, git_repo, {"src/service.py"}, strict=True)
    assert strict.turns[0].paths == ()


def test_absolute_repo_paths_are_stripped_from_rendered_text(
    tmp_path: Path, git_repo: Path
) -> None:
    """The rendered summary must not carry the user's home directory.

    It is PII, it is noise, and it costs tokens on every segment sent.
    """
    path = write_transcript(
        tmp_path / "abs_text.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Write",
                            "input": {"file_path": str(git_repo / "src" / "service.py")},
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)
    normalized = parser.normalize_paths(transcript, git_repo, {"src/service.py"})
    assert normalized.turns[0].tool_calls[0].text == "Write(src/service.py)"
    assert str(git_repo) not in normalized.turns[0].tool_calls[0].text


def test_tool_output_keeps_the_verdict_at_the_end(tmp_path: Path, git_repo: Path) -> None:
    """Head-only truncation discards the one line that says whether it worked.

    Found by evaluating extraction on a real session: `5 failed, 135 passed`
    sat at character 2572 of a pytest result and never reached the model, so
    no test failure in the whole session was detectable.
    """
    body = "detail line\n" * 400 + "FAILED tests/test_x.py::test_y\n5 failed, 135 passed in 1.12s"
    path = write_transcript(
        tmp_path / "long_result.jsonl",
        [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": body,
                            "is_error": True,
                        }
                    ],
                },
            }
        ],
    )
    transcript = parser.parse_file(path)
    kept = transcript.turns[0].tool_results[0].text

    assert "5 failed, 135 passed" in kept, "the verdict must survive truncation"
    assert "FAILED tests/test_x.py::test_y" in kept
    assert "detail line" in kept, "the head must survive too"
    assert "elided" in kept
    assert len(kept) < len(body)
