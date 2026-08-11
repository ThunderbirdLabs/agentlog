"""Shared fixtures.

Two rules the whole suite obeys:

* no test touches the network — extraction is always given a fake client
* fixture transcripts are hand-written and contain no real secrets
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentlog.domains.transcript.schemas import Segment, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"
BASE_TIME = datetime(2026, 8, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_turn(
    index: int,
    role: str = "assistant",
    text: str = "",
    paths: tuple[str, ...] = (),
    minutes: int = 0,
    is_error: bool = False,
    tool_name: str = "Edit",
) -> Turn:
    calls = (
        (ToolCall(id=f"t{index}", name=tool_name, paths=paths, text=f"{tool_name}({paths[0]})"),)
        if paths
        else ()
    )
    results = (
        (ToolResult(tool_use_id=f"t{index}", is_error=is_error, text="boom" if is_error else "ok"),)
        if paths
        else ()
    )
    return Turn(
        index=index,
        role=role,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        text=text,
        tool_calls=calls,
        tool_results=results,
    )


def make_segment(
    texts: list[str] | None = None,
    files: tuple[str, ...] = ("src/app.py",),
    session_id: str = "sess-test",
) -> Segment:
    texts = texts or ["did a thing"]
    turns = tuple(
        make_turn(index=i, text=text, paths=files, minutes=i) for i, text in enumerate(texts)
    )
    return Segment(
        session_id=session_id,
        start_turn=turns[0].index,
        end_turn=turns[-1].index,
        turns=turns,
        files=files,
        started_at=turns[0].timestamp,
        ended_at=turns[-1].timestamp,
    )


def write_transcript(path: Path, lines: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def fixture_transcript() -> Path:
    return FIXTURES / "export_timeout.jsonl"


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A real git repo with one commit. Anchors are tested offline against this."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)

    def run(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
        )

    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "agentlog tests")
    run("config", "commit.gpgsign", "false")

    (repo / "src" / "service.py").write_text(
        "from fastapi import APIRouter\n"
        "\n"
        'router = APIRouter(prefix="/api/v1/inspections")\n'
        "\n"
        "\n"
        "class InspectionExportService:\n"
        "    def render(self):\n"
        "        return None\n"
        "\n"
        "    def other(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        '@router.post("/{id}/export")\n'
        "def export(id: str):\n"
        "    return InspectionExportService().render()\n",
        encoding="utf-8",
    )
    run("add", "-A")
    run("commit", "-q", "-m", "initial")
    run("checkout", "-q", "-b", "feat/THU-142-inspection-export")
    return repo
