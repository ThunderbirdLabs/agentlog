"""Command line interface.

v0.1 writes nothing, anywhere. `capture` reads a transcript and prints what it
found at the stage you ask for. There is no `--write` flag yet, because there
is no log to write to — a flag that errors would be worse than its absence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from agentlog.core import config as config_module
from agentlog.core.errors import AgentlogError
from agentlog.core.logging import configure, get_logger
from agentlog.domains.anchors import git
from agentlog.domains.anchors import service as anchors_service
from agentlog.domains.extraction import service as extraction_service
from agentlog.domains.transcript import parser, segmenter
from agentlog.domains.transcript.schemas import Segment, Transcript

log = get_logger("cli")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Key what an AI coding agent tried and failed to the code it touched.",
)

STAGES = ("turns", "segments", "anchors", "payload", "extract")


@app.callback()
def _root() -> None:
    """Keep subcommand form even while `capture` is the only command."""


def _echo(text: str = "") -> None:
    typer.echo(text)


def _resolve_repo(transcript: Transcript, override: Optional[Path]) -> Path:  # noqa: UP045
    if override is not None:
        candidate = override.resolve()
    elif transcript.cwd:
        candidate = Path(transcript.cwd)
    else:
        candidate = Path.cwd()
    root = git.repo_root(candidate)
    if root is not None:
        return root
    log.warning(
        "%s is not inside a git repository; git-derived anchors will be empty",
        candidate,
    )
    return candidate


def _load(path: Path, repo_override: Path | None) -> tuple[Transcript, Path]:
    """Parse a transcript and resolve its paths against the repo.

    Two normalisation passes. The first is permissive and only asks whether a
    candidate is a real file; the second keeps what git either tracks or does
    not ignore. Without the second pass, anything the tooling writes inside the
    repo — `.venv/`, `node_modules/`, build output — becomes a file anchor the
    moment an agent reads it.
    """
    transcript = parser.parse_file(path)
    repo = _resolve_repo(transcript, repo_override)

    tracked = git.tracked_files(repo)
    permissive = parser.normalize_paths(transcript, repo)
    candidates = {path for turn in permissive.turns for path in turn.paths}
    allowed = tracked | (candidates - git.ignored(repo, candidates))
    return parser.normalize_paths(transcript, repo, allowed, strict=True), repo


def _segment_header(segment: Segment) -> str:
    span = f"turns {segment.start_turn}-{segment.end_turn}"
    when = segment.started_at.isoformat() if segment.started_at else "unknown time"
    return f"segment {span} ({segment.turn_count} turns, {when}, {segment.tool_errors} tool errors)"


@app.command()
def capture(
    path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Session transcript (.jsonl)"
    ),
    stage: str = typer.Option(
        "extract",
        "--stage",
        help=f"How far to run the pipeline: {', '.join(STAGES)}",
    ),
    repo: Optional[Path] = typer.Option(  # noqa: UP045 - typer needs Optional here
        None, "--repo", help="Repo root. Defaults to the cwd recorded in the transcript."
    ),
    limit: int = typer.Option(0, "--limit", help="Only process the first N segments (0 = all)."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of prose."),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING, ERROR."),
) -> None:
    """Process one transcript and print the result. Reads only; writes nothing."""
    configure(log_level)
    if stage not in STAGES:
        raise typer.BadParameter(f"stage must be one of: {', '.join(STAGES)}")

    transcript, repo_root = _load(path, repo)

    if stage == "turns":
        _emit_turns(transcript, repo_root, as_json)
        return

    cfg = config_module.load(repo_root)
    segments = segmenter.segment(transcript, cfg.segmentation)
    if limit > 0:
        segments = segments[:limit]

    if stage == "segments":
        _emit_segments(transcript, segments, as_json)
        return

    resolved = [(segment, anchors_service.resolve(segment, cfg, repo_root)) for segment in segments]

    if stage == "anchors":
        _emit_anchors(transcript, resolved, as_json)
        return

    if stage == "payload":
        _emit_payload(resolved)
        return

    _emit_extraction(transcript, resolved, cfg, as_json)


def _emit_payload(resolved: list[tuple]) -> None:
    """Print the exact scrubbed text that would be sent, and nothing more.

    This is the last checkpoint before anything leaves the machine. It runs the
    real `scrub()`, so what you read here is byte-for-byte what the model would
    see — and it costs nothing, which makes it the right place to judge whether
    the prompt is worth paying for.
    """
    from agentlog.core.redaction import scrub
    from agentlog.domains.extraction import prompts

    _echo("=" * 78)
    _echo("SYSTEM PROMPT")
    _echo("=" * 78)
    _echo(prompts.SYSTEM)
    for segment, _anchors in resolved:
        _echo("=" * 78)
        _echo(f"USER MESSAGE — {_segment_header(segment)}")
        _echo("=" * 78)
        scrubbed = scrub(prompts.user_prompt(extraction_service.render(segment)))
        _echo(scrubbed.text)
        _echo()


def _emit_turns(transcript: Transcript, repo: Path, as_json: bool) -> None:
    if as_json:
        _echo(json.dumps(json.loads(transcript.model_dump_json()), indent=2))
        return
    _echo(f"session {transcript.session_id}")
    _echo(f"repo    {repo}")
    _echo(
        f"turns   {len(transcript.turns)}  "
        f"(skipped {transcript.skipped_lines} lines; other types: "
        f"{', '.join(transcript.unknown_types) or 'none'})"
    )
    _echo()
    for turn in transcript.turns:
        who = "human" if turn.is_human else turn.role
        when = turn.timestamp.strftime("%H:%M:%S") if turn.timestamp else "--:--:--"
        _echo(f"[{turn.index:>4}] {when} {who}")
        if turn.text:
            first = turn.text.splitlines()[0]
            _echo(f"         {first[:140]}")
        for call in turn.tool_calls:
            paths = f"  files={list(call.paths)}" if call.paths else ""
            _echo(f"         -> {call.text[:120]}{paths}")
        for result in turn.tool_results:
            marker = "ERROR" if result.is_error else "ok"
            _echo(
                f"         <- [{marker}] {result.text.splitlines()[0][:110] if result.text else ''}"
            )


def _emit_segments(transcript: Transcript, segments: list[Segment], as_json: bool) -> None:
    if as_json:
        payload = [json.loads(segment.model_dump_json()) for segment in segments]
        _echo(json.dumps(payload, indent=2))
        return
    _echo(
        f"session {transcript.session_id}: {len(segments)} segments from {len(transcript.turns)} turns"
    )
    _echo()
    for segment in segments:
        _echo(_segment_header(segment))
        for file in segment.files:
            _echo(f"  file  {file}")
        _echo()


def _emit_anchors(transcript: Transcript, resolved: list[tuple], as_json: bool) -> None:
    if as_json:
        payload = [
            {
                "segment": json.loads(segment.model_dump_json(exclude={"turns"})),
                "anchors": json.loads(anchors.model_dump_json()),
            }
            for segment, anchors in resolved
        ]
        _echo(json.dumps(payload, indent=2))
        return
    _echo(f"session {transcript.session_id}: {len(resolved)} segments")
    _echo()
    for segment, anchors in resolved:
        _echo(_segment_header(segment))
        _echo(f"  branch   {anchors.branch or '-'}")
        _echo(f"  issue    {anchors.issue or '-'}")
        _echo(f"  head     {anchors.head_sha or '-'}")
        _echo(f"  commits  {', '.join(anchors.commits) or '-'}")
        for file in anchors.files:
            _echo(f"  file     {file}")
        for route in anchors.routes:
            _echo(f"  route    {route}")
        for symbol in anchors.symbols:
            _echo(f"  symbol   {symbol}")
        _echo()


def _emit_extraction(transcript: Transcript, resolved: list[tuple], cfg, as_json: bool) -> None:
    from agentlog.external.anthropic import AnthropicClient

    client = AnthropicClient(model=cfg.model)
    results = [
        extraction_service.extract(segment, anchors, client, cfg.model)
        for segment, anchors in resolved
    ]

    if as_json:
        payload = [
            {
                "segment": {
                    "session_id": result.segment.session_id,
                    "start_turn": result.segment.start_turn,
                    "end_turn": result.segment.end_turn,
                },
                "anchors": json.loads(result.anchors.model_dump_json()),
                "extractor": result.extractor,
                "dropped": result.dropped,
                "drop_reason": result.drop_reason,
                "records": [json.loads(c.model_dump_json()) for c in result.candidates],
            }
            for result in results
        ]
        _echo(json.dumps(payload, indent=2))
        return

    total = sum(len(result.candidates) for result in results)
    dropped = sum(1 for result in results if result.dropped)
    _echo(
        f"session {transcript.session_id}: {len(results)} segments, "
        f"{total} candidate records, {dropped} segments dropped"
    )
    _echo()
    for result in results:
        _echo(_segment_header(result.segment))
        if result.dropped:
            _echo(f"  DROPPED: {result.drop_reason}")
            _echo()
            continue
        if not result.candidates:
            _echo("  (nothing worth recording)")
            _echo()
            continue
        for candidate in result.candidates:
            label = candidate.kind
            if candidate.outcome:
                label = f"{label}/{candidate.outcome}"
            _echo(
                f"  {label}  [{candidate.evidence}] "
                f"source={candidate.source} confidence={candidate.confidence}"
            )
            _echo(f"    {candidate.summary}")
            if candidate.detail:
                _echo(f"    {candidate.detail}")
        anchors = result.anchors
        keys = [f"file:{f}" for f in anchors.files[:4]]
        keys += [f"route:{r}" for r in anchors.routes[:2]]
        keys += [f"symbol:{s}" for s in anchors.symbols[:3]]
        if anchors.issue:
            keys.append(f"issue:{anchors.issue}")
        _echo(f"    keyed to: {', '.join(keys) or '-'}")
        _echo()


def main() -> None:  # pragma: no cover - console entry point
    try:
        app()
    except AgentlogError as exc:
        typer.echo(f"agentlog: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
