"""JSONL -> normalised `Turn`.

Parse defensively. The session format is not a published contract:

* unknown message types are skipped, never fatal
* a malformed line skips that line and continues, never aborts the file
* nothing downstream touches raw JSONL keys

If a non-empty file yields zero turns, the caller warns by name. Silent
zero-yield looks like success and is the worst outcome available.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from agentlog.core.logging import get_logger
from agentlog.domains.transcript import reader
from agentlog.domains.transcript.schemas import ToolCall, ToolResult, Transcript, Turn

log = get_logger("transcript.parser")

# Line types that carry conversation. Everything else in a session file is
# bookkeeping (hook summaries, file snapshots, queue operations, titles).
_CONVERSATION_TYPES = frozenset({"user", "assistant"})

# Tool input keys whose value is always a single file path.
_FILE_PARAMS = ("file_path", "notebook_path", "filePath")
# Keys that are *sometimes* a file and sometimes a directory.
_AMBIGUOUS_PARAMS = ("path",)

_PATH_TOKEN = re.compile(r"[A-Za-z0-9_./\-]*[A-Za-z0-9_\-]\.[A-Za-z0-9]{1,10}")

_MAX_TEXT = 2000
_MAX_THINKING = 400
_MAX_TOOL_TEXT = 300
# Tool output is kept from both ends. Test runners, linters, compilers and
# build tools all print detail first and the verdict last, so head-only
# truncation throws away the one line that says whether it worked.
_RESULT_HEAD = 300
_RESULT_TAIL = 600


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …[truncated]"


def _truncate_ends(text: str, head: int, tail: int) -> str:
    """Keep both ends of a long output, eliding the middle.

    The tail is the larger half deliberately: `5 failed, 135 passed` is the
    whole signal, and it is the last line of a 2,600-character result.
    """
    text = text.strip()
    if len(text) <= head + tail:
        return text
    elided = len(text) - head - tail
    return f"{text[:head].rstrip()}\n…[{elided} chars elided]…\n{text[-tail:].lstrip()}"


def _parse_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _candidate_paths(value: object, depth: int = 0) -> list[str]:
    """Pull path-shaped tokens out of an arbitrary tool input value.

    Precision comes later, in `normalize_paths`, where candidates are checked
    against the repo's actual files. Being liberal here and strict there beats
    hand-maintaining a per-tool schema that breaks whenever a tool changes.
    """
    if depth > 4:
        return []
    found: list[str] = []
    if isinstance(value, str):
        found.extend(_PATH_TOKEN.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_candidate_paths(item, depth + 1))
    elif isinstance(value, list):
        for item in value:
            found.extend(_candidate_paths(item, depth + 1))
    return found


def _tool_paths(tool_input: object) -> tuple[str, ...]:
    if not isinstance(tool_input, dict):
        return ()
    ordered: dict[str, None] = {}
    for key in _FILE_PARAMS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            ordered[value] = None
    for key in _AMBIGUOUS_PARAMS:
        value = tool_input.get(key)
        # A bare `path` is a directory as often as a file (Glob, Grep). Only
        # take it when it carries a suffix.
        if isinstance(value, str) and value and Path(value).suffix:
            ordered[value] = None
    for candidate in _candidate_paths(tool_input):
        ordered[candidate] = None
    return tuple(ordered)


def _tool_summary(name: str, tool_input: object) -> str:
    """A one-line rendering of a tool call for the extractor to read.

    Never the raw input: a single Edit carries the whole old and new file body,
    which would crowd out the reasoning the extractor actually needs.
    """
    if not isinstance(tool_input, dict):
        return name
    for key in (*_FILE_PARAMS, *_AMBIGUOUS_PARAMS):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return f"{name}({value})"
    for key in ("command", "pattern", "query", "prompt", "description"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return f"{name}({_truncate(value, _MAX_TOOL_TEXT)})"
    return name


def _result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _parse_assistant(obj: dict, index: int) -> Turn | None:
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None

    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        elif kind == "thinking":
            thought = block.get("thinking")
            if isinstance(thought, str) and thought.strip():
                # Thinking is where abandoned approaches get stated outright,
                # so it is worth carrying — heavily truncated.
                #
                # Usually there is nothing to carry. On current models the
                # transcript stores `thinking` blocks whose text is empty,
                # because `display` defaults to "omitted". Extraction therefore
                # cannot rely on reasoning being present; the load-bearing
                # signals are assistant text, tool calls, and tool errors.
                texts.append(f"[thinking] {_truncate(thought, _MAX_THINKING)}")
        elif kind == "tool_use":
            tool_input = block.get("input")
            calls.append(
                ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or "unknown"),
                    paths=_tool_paths(tool_input),
                    text=_tool_summary(str(block.get("name") or "unknown"), tool_input),
                )
            )

    if not texts and not calls:
        return None

    return Turn(
        index=index,
        role="assistant",
        timestamp=_parse_timestamp(obj.get("timestamp")),
        text=_truncate("\n".join(texts), _MAX_TEXT),
        tool_calls=tuple(calls),
        uuid=obj.get("uuid") if isinstance(obj.get("uuid"), str) else None,
        is_sidechain=bool(obj.get("isSidechain")),
        cwd=obj.get("cwd") if isinstance(obj.get("cwd"), str) else None,
        git_branch=obj.get("gitBranch") if isinstance(obj.get("gitBranch"), str) else None,
    )


def _parse_user(obj: dict, index: int) -> Turn | None:
    # Meta turns are system-injected context, not anything a human or the
    # agent said. They would read to the extractor as instructions.
    if obj.get("isMeta"):
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")

    origin = obj.get("origin")
    is_human = isinstance(origin, dict) and origin.get("kind") == "human"

    if isinstance(content, str):
        if not content.strip():
            return None
        return Turn(
            index=index,
            role="user",
            timestamp=_parse_timestamp(obj.get("timestamp")),
            text=_truncate(content, _MAX_TEXT),
            is_human=is_human or obj.get("promptSource") == "typed",
            uuid=obj.get("uuid") if isinstance(obj.get("uuid"), str) else None,
            is_sidechain=bool(obj.get("isSidechain")),
            cwd=obj.get("cwd") if isinstance(obj.get("cwd"), str) else None,
            git_branch=obj.get("gitBranch") if isinstance(obj.get("gitBranch"), str) else None,
        )

    if not isinstance(content, list):
        return None

    results: list[ToolResult] = []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            text = _result_text(block.get("content"))
            results.append(
                ToolResult(
                    tool_use_id=str(block.get("tool_use_id") or ""),
                    is_error=bool(block.get("is_error")),
                    text=_truncate_ends(text, _RESULT_HEAD, _RESULT_TAIL),
                )
            )
        elif block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    if not results and not texts:
        return None

    return Turn(
        index=index,
        role="user",
        timestamp=_parse_timestamp(obj.get("timestamp")),
        text=_truncate("\n".join(texts), _MAX_TEXT),
        tool_results=tuple(results),
        is_human=bool(texts) and is_human,
        uuid=obj.get("uuid") if isinstance(obj.get("uuid"), str) else None,
        is_sidechain=bool(obj.get("isSidechain")),
        cwd=obj.get("cwd") if isinstance(obj.get("cwd"), str) else None,
        git_branch=obj.get("gitBranch") if isinstance(obj.get("gitBranch"), str) else None,
    )


def parse_file(path: Path) -> Transcript:
    """Parse a session file into a `Transcript`.

    Never raises for content reasons. Only `reader.stream_lines` can raise, and
    only when the file itself is unusable.
    """
    turns: list[Turn] = []
    skipped = 0
    unknown: Counter[str] = Counter()
    session_id = path.stem
    cwd: str | None = None

    for raw in reader.stream_lines(path):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(obj, dict):
            skipped += 1
            continue

        line_type = obj.get("type")
        if line_type not in _CONVERSATION_TYPES:
            if isinstance(line_type, str):
                unknown[line_type] += 1
            else:
                skipped += 1
            continue

        if cwd is None and isinstance(obj.get("cwd"), str):
            cwd = obj["cwd"]
        raw_session = obj.get("sessionId") or obj.get("session_id")
        if isinstance(raw_session, str) and raw_session:
            session_id = raw_session

        index = len(turns)
        turn = _parse_assistant(obj, index) if line_type == "assistant" else _parse_user(obj, index)
        if turn is None:
            skipped += 1
            continue
        turns.append(turn)

    if not turns:
        log.warning(
            "parsed zero turns from %s (%d lines skipped, other types: %s) — "
            "the transcript format may have changed",
            path,
            skipped,
            dict(unknown) or "none",
        )

    return Transcript(
        session_id=session_id,
        path=str(path),
        cwd=cwd,
        turns=tuple(turns),
        skipped_lines=skipped,
        unknown_types=tuple(sorted(unknown)),
    )


def normalize_paths(
    transcript: Transcript,
    repo_root: Path,
    known_files: set[str] | None = None,
    strict: bool = False,
) -> Transcript:
    """Rewrite tool-call paths to repo-relative form and drop the rest.

    This is where liberal extraction becomes precise. A candidate survives only
    if it resolves to a real file in the repo. Everything else — log lines that
    look like paths, files in another repo, `/tmp` scratch — is discarded.

    `strict` restricts acceptance to `known_files` alone, disabling the
    on-disk fallback. The composition root runs this twice: once permissively
    to gather candidates, then strictly against the set git says is neither
    ignored nor untracked junk.
    """
    repo_root = repo_root.resolve()

    def resolve(candidate: str) -> str | None:
        cleaned = candidate.strip().strip("`'\"(),;:")
        if not cleaned:
            return None
        try:
            path = Path(cleaned)
            absolute = path if path.is_absolute() else repo_root / path
            resolved = absolute.resolve()
            relative = resolved.relative_to(repo_root)
        except (ValueError, OSError):
            return None
        rel = relative.as_posix()
        if not rel or rel == ".":
            return None
        if known_files is not None and rel in known_files:
            return rel
        if strict:
            return None
        if resolved.is_file():
            return rel
        return None

    # Absolute paths in the rendered tool summary carry the user's home
    # directory into the extraction prompt: it is PII, it is noise, and it
    # costs tokens on every segment. Rewrite them to repo-relative.
    root_prefix = f"{repo_root}/"

    new_turns = []
    for turn in transcript.turns:
        calls = []
        for call in turn.tool_calls:
            resolved: dict[str, None] = {}
            for candidate in call.paths:
                rel = resolve(candidate)
                if rel is not None:
                    resolved[rel] = None
            calls.append(
                call.model_copy(
                    update={
                        "paths": tuple(resolved),
                        "text": call.text.replace(root_prefix, ""),
                    }
                )
            )
        results = tuple(
            result.model_copy(update={"text": result.text.replace(root_prefix, "")})
            for result in turn.tool_results
        )
        new_turns.append(
            turn.model_copy(
                update={
                    "tool_calls": tuple(calls),
                    "tool_results": results,
                    "text": turn.text.replace(root_prefix, ""),
                }
            )
        )

    return transcript.model_copy(update={"turns": tuple(new_turns)})
