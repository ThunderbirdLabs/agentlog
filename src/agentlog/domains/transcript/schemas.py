"""Normalised transcript types.

Nothing downstream of the parser touches raw session JSONL. The session file
format is not a published contract and will change; when it does, only
`parser.py` changes. The same boundary is what lets a second reader (Cursor,
Codex) be added later without touching segmentation, anchors, or extraction.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    # Repo-relative once `parser.normalize_paths` has run; raw strings before.
    paths: tuple[str, ...] = ()
    # A compact rendering of the tool input for the extractor to read. Never
    # the raw input dict: file bodies and diffs would dominate the prompt.
    text: str = ""


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_use_id: str
    is_error: bool = False
    text: str = ""


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    role: str  # "user" | "assistant"
    timestamp: datetime | None = None
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    uuid: str | None = None
    is_sidechain: bool = False
    is_human: bool = False
    cwd: str | None = None
    git_branch: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for call in self.tool_calls:
            for path in call.paths:
                seen[path] = None
        return tuple(seen)

    @property
    def had_error(self) -> bool:
        return any(result.is_error for result in self.tool_results)


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    path: str
    cwd: str | None = None
    turns: tuple[Turn, ...] = ()
    # Lines that failed to parse or were of an unrecognised type. Surfaced by
    # the CLI: a file that yields zero turns but many skips is a format change,
    # and silent zero-yield looks like success.
    skipped_lines: int = 0
    unknown_types: tuple[str, ...] = ()


class Segment(BaseModel):
    """A unit of work: consecutive turns whose touched files overlap."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    start_turn: int
    end_turn: int
    turns: tuple[Turn, ...]
    files: tuple[str, ...] = ()
    started_at: datetime | None = None
    ended_at: datetime | None = None
    tool_errors: int = Field(default=0)

    @property
    def turn_count(self) -> int:
        return len(self.turns)
