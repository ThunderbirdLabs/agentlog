"""The whole thing, on a simulated history.

Simulates the failure the tool exists for: something worked in June, a config
key was renamed in July, it broke in August, and three sessions were burned
re-testing knobs that had already been ruled out.

The question this has to answer is the one that cost three days — *what
changed between when it worked and now* — and it has to answer it from the
transcripts alone, without cloning anything or diffing by hand.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentlog import pipeline
from agentlog.core.config import Config
from agentlog.domains.extraction.schemas import Candidate, ExtractionResponse
from agentlog.domains.retrieval import service as retrieval
from agentlog.domains.store import dedupe
from agentlog.domains.store import index as index_module
from agentlog.domains.store import log as log_module

JUNE = datetime(2026, 6, 14, 10, 0, tzinfo=timezone.utc)
JULY = datetime(2026, 7, 22, 15, 0, tzinfo=timezone.utc)
AUGUST = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)

WORKER = "worker/pipeline.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _session(path: Path, session_id: str, cwd: Path, start: datetime, turns: list[dict]) -> Path:
    """Write a transcript in the shape Claude Code actually produces."""
    lines = []
    for i, turn in enumerate(turns):
        stamp = (start + timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
        base = {
            "uuid": f"{session_id}-{i}",
            "timestamp": stamp,
            "isSidechain": False,
            "cwd": str(cwd),
            "sessionId": session_id,
            "gitBranch": turn.get("branch", "main"),
            "userType": "external",
        }
        if turn["role"] == "human":
            lines.append(
                {
                    **base,
                    "type": "user",
                    "message": {"role": "user", "content": turn["text"]},
                    "origin": {"kind": "human"},
                    "promptSource": "typed",
                }
            )
        elif turn["role"] == "tool":
            lines.append(
                {
                    **base,
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": f"t{i}",
                                "name": turn.get("tool", "Edit"),
                                "input": turn["input"],
                            }
                        ],
                    },
                }
            )
        elif turn["role"] == "result":
            lines.append(
                {
                    **base,
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"t{i - 1}",
                                "content": turn["text"],
                                **({"is_error": True} if turn.get("error") else {}),
                            }
                        ],
                    },
                }
            )
        else:
            lines.append(
                {
                    **base,
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": turn["text"]}],
                    },
                }
            )
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def history(tmp_path: Path) -> tuple[Path, list[Path]]:
    """A repo whose splitter worked in June and broke after a July rename."""
    repo = tmp_path / "survey-worker"
    (repo / "worker").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "agentlog tests")
    _git(repo, "config", "commit.gpgsign", "false")

    worker = repo / WORKER
    worker.write_text(
        "def run(job):\n"
        "    return process(\n"
        "        splitter_chunk_size=512,\n"
        "        fill_occlusion_holes=True,\n"
        "    )\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "worker: known good")

    # July: the rename that breaks it two months later.
    worker.write_text(
        "def run(job):\n"
        "    return process(\n"
        "        chunk_size_for_splitter=512,\n"
        "        fill_occlusion_holes=True,\n"
        "    )\n",
        encoding="utf-8",
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    paths = [
        _session(
            sessions / "june.jsonl",
            "sess-june",
            repo,
            JUNE,
            [
                {"role": "human", "text": "get the splitter running on the big dataset"},
                {
                    "role": "tool",
                    "input": {"file_path": WORKER, "old_string": "a", "new_string": "b"},
                },
                {"role": "result", "text": "The file has been updated."},
                {"role": "tool", "tool": "Bash", "input": {"command": "python -m worker.run"}},
                {"role": "result", "text": "processed 1240 tiles\n0 errors\nOK"},
                {"role": "text", "text": "Splitter runs clean at that chunk size."},
            ],
        ),
        _session(
            sessions / "july.jsonl",
            "sess-july",
            repo,
            JULY,
            [
                {"role": "human", "text": "tidy up the worker config naming"},
                {
                    "role": "tool",
                    "input": {"file_path": WORKER, "old_string": "x", "new_string": "y"},
                },
                {"role": "result", "text": "The file has been updated."},
                {"role": "tool", "tool": "Bash", "input": {"command": "python -m worker.run"}},
                {"role": "result", "text": "processed 1240 tiles\n0 errors\nOK"},
            ],
        ),
        _session(
            sessions / "august.jsonl",
            "sess-august",
            repo,
            AUGUST,
            [
                {"role": "human", "text": "the splitter stopped working, it ran fine before"},
                {"role": "tool", "tool": "Bash", "input": {"command": "python -m worker.run"}},
                {
                    "role": "result",
                    "text": "Traceback (most recent call last)\nMemoryError",
                    "error": True,
                },
                {"role": "text", "text": "Looks like memory. Lowering the worker count."},
                {
                    "role": "tool",
                    "input": {"file_path": WORKER, "old_string": "p", "new_string": "q"},
                },
                {"role": "result", "text": "The file has been updated."},
                {"role": "tool", "tool": "Bash", "input": {"command": "python -m worker.run"}},
                {
                    "role": "result",
                    "text": "Traceback (most recent call last)\nMemoryError",
                    "error": True,
                },
                {
                    "role": "text",
                    "text": "Not memory then. Trying a different densification preset.",
                },
                {
                    "role": "tool",
                    "input": {"file_path": WORKER, "old_string": "r", "new_string": "s"},
                },
                {"role": "result", "text": "The file has been updated."},
                {"role": "tool", "tool": "Bash", "input": {"command": "python -m worker.run"}},
                {
                    "role": "result",
                    "text": "Traceback (most recent call last)\nMemoryError",
                    "error": True,
                },
            ],
        ),
    ]
    return repo, paths


class ScriptedClient:
    """Stands in for the extractor so the test is deterministic.

    Extraction quality is evaluated separately; what this exercises is
    everything around it — anchors, dedupe, the log, the index, retrieval.
    """

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, system, user, schema):  # noqa: ANN001, ANN201
        self.calls += 1
        text = user.text
        if "stopped working" in text:
            return ExtractionResponse(
                records=[
                    Candidate(
                        kind="attempt",
                        outcome="failed",
                        summary="Lowering the worker count did not stop the crash.",
                        detail="The run failed the same way after reducing concurrency.",
                        evidence="error_output",
                        source="stated",
                        confidence="high",
                    ),
                    Candidate(
                        kind="attempt",
                        outcome="failed",
                        summary="A different densification preset made no difference either.",
                        detail="Same crash, so the preset is not what changed.",
                        evidence="error_output",
                        source="stated",
                        confidence="high",
                    ),
                ]
            )
        if "naming" in text:
            return ExtractionResponse(
                records=[
                    Candidate(
                        kind="decision",
                        summary="Renamed the splitter chunk size setting for consistency.",
                        detail="A naming tidy-up with no behaviour change intended.",
                        evidence="none",
                        source="stated",
                        confidence="high",
                    )
                ]
            )
        return ExtractionResponse(
            records=[
                Candidate(
                    kind="attempt",
                    outcome="worked",
                    summary="The splitter completed the full dataset at the configured chunk size.",
                    detail="Ran to completion with no errors.",
                    evidence="none",
                    source="stated",
                    confidence="high",
                )
            ]
        )


def _run(repo: Path, paths: list[Path], client, write: bool = True):  # noqa: ANN001
    return pipeline.process(
        paths, Config(repo_root=repo), client=client, repo_override=repo, write=write
    )


def test_backfill_writes_a_timeline(history) -> None:
    repo, paths = history
    cfg = Config(repo_root=repo)
    result = _run(repo, paths, ScriptedClient())

    assert result.records, "backfill produced no records"
    assert log_module.log_path(cfg.data_dir).is_file()

    hits = retrieval.timeline(cfg.data_dir, WORKER)
    assert len(hits) >= 3
    stamps = [hit.record.occurred_at for hit in hits]
    assert stamps == sorted(stamps), "a timeline must be chronological"
    assert stamps[0].month == 6 and stamps[-1].month == 8


def test_the_working_baseline_is_findable(history) -> None:
    """ "It worked in June" has to be retrievable, not just the failures."""
    repo, paths = history
    cfg = Config(repo_root=repo)
    _run(repo, paths, ScriptedClient())

    worked = [
        h.record
        for h in retrieval.timeline(cfg.data_dir, WORKER)
        if h.record.kind == "attempt" and h.record.outcome == "worked"
    ]
    assert worked, "the last-known-good is what you diff against"
    assert worked[0].occurred_at.month == 6
    assert worked[0].anchors.head_sha


def test_searching_the_old_setting_name_finds_the_rename(history) -> None:
    """The three-day question, answered in one query.

    You search the name that was in use the last time it worked. It only
    exists on the removed side of a diff, so this is the case that fails
    entirely if only added lines are anchored.
    """
    repo, paths = history
    cfg = Config(repo_root=repo)
    _run(repo, paths, ScriptedClient())

    hits = retrieval.by_anchor(cfg.data_dir, "setting", "splitter_chunk_size")
    assert hits, "the old setting name must still be reachable"

    by_search = retrieval.search(cfg.data_dir, "splitter_chunk_size")
    assert by_search, "FTS must match on the anchor even when no summary mentions it"


def test_dead_ends_outrank_everything_else(history) -> None:
    repo, paths = history
    cfg = Config(repo_root=repo)
    _run(repo, paths, ScriptedClient())

    hits = retrieval.by_anchor(cfg.data_dir, "file", WORKER)
    assert hits[0].record.is_dead_end, "a failed attempt with evidence should rank first"


def test_processing_twice_writes_nothing_new(history) -> None:
    """Hooks fire repeatedly. A replay must cost nothing and change nothing."""
    repo, paths = history
    cfg = Config(repo_root=repo)

    first_client = ScriptedClient()
    first = _run(repo, paths, first_client)
    before = log_module.count(cfg.data_dir)

    second_client = ScriptedClient()
    second = _run(repo, paths, second_client)

    assert second.records == []
    assert second_client.calls == 0, "a replay must not spend a single model call"
    assert log_module.count(cfg.data_dir) == before
    assert first.records


def test_reindex_reproduces_the_index_exactly(history) -> None:
    repo, paths = history
    cfg = Config(repo_root=repo)
    _run(repo, paths, ScriptedClient())

    before = index_module.rebuild(cfg.data_dir)
    conn = index_module.connect(cfg.data_dir)
    first = conn.execute("SELECT id, summary FROM records ORDER BY id").fetchall()
    anchors_first = conn.execute("SELECT * FROM anchors ORDER BY record_id, kind, value").fetchall()
    conn.close()

    after = index_module.rebuild(cfg.data_dir)
    conn = index_module.connect(cfg.data_dir)
    second = conn.execute("SELECT id, summary FROM records ORDER BY id").fetchall()
    anchors_second = conn.execute(
        "SELECT * FROM anchors ORDER BY record_id, kind, value"
    ).fetchall()
    conn.close()

    assert before == after == log_module.count(cfg.data_dir)
    assert [tuple(r) for r in first] == [tuple(r) for r in second]
    assert [tuple(r) for r in anchors_first] == [tuple(r) for r in anchors_second]


def test_the_log_survives_losing_the_index(history) -> None:
    """The index is disposable. Deleting it must cost nothing but time."""
    repo, paths = history
    cfg = Config(repo_root=repo)
    _run(repo, paths, ScriptedClient())
    expected = len(retrieval.timeline(cfg.data_dir, WORKER))

    index_module.index_path(cfg.data_dir).unlink()
    assert len(retrieval.timeline(cfg.data_dir, WORKER)) == expected


def test_a_dry_run_spends_nothing_and_writes_nothing(history) -> None:
    repo, paths = history
    cfg = Config(repo_root=repo)
    result = pipeline.process(paths, cfg, client=None, repo_override=repo)

    assert result.segments_new > 0
    assert result.records == []
    assert result.estimated_cost_usd > 0
    assert not log_module.log_path(cfg.data_dir).exists()
    assert not dedupe.cursors_path(cfg.data_dir).exists()


def test_transcripts_from_another_repo_are_not_folded_in(history, tmp_path: Path) -> None:
    """Anchors are repo-relative; a foreign session's paths do not exist here."""
    repo, paths = history
    other = tmp_path / "somewhere-else"
    other.mkdir()
    foreign = _session(
        tmp_path / "foreign.jsonl",
        "sess-foreign",
        other,
        AUGUST,
        [{"role": "human", "text": "unrelated work"}],
    )
    assert pipeline.belongs_to(paths[0], repo)
    assert not pipeline.belongs_to(foreign, repo)


def test_staging_writes_the_instructions_beside_the_queue(history) -> None:
    """A queued unit is only the slice; the rules live in the system prompt.

    Without this file whoever drains the queue — a skill, a person — has a
    transcript excerpt and no idea what to produce from it.
    """
    from agentlog.domains.extraction import prompts

    repo, paths = history
    cfg = Config(repo_root=repo)
    queued = pipeline.stage(paths, cfg, repo_override=repo)

    assert queued > 0
    instructions = pipeline.instructions_path(cfg)
    assert instructions.is_file()
    assert instructions.read_text(encoding="utf-8") == prompts.SYSTEM
    assert instructions not in pipeline.pending(cfg), "it must not look like a work unit"
