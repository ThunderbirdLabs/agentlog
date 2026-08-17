"""Command line interface.

`capture` inspects one transcript and prints what it found at whichever stage
you ask for; it never writes. `backfill` is the one-time step after install —
it reads what is already on disk so the log is useful immediately, and it is a
dry run unless you pass `--write`. Everything else reads the store back out.

The entry-point block lives at the very bottom of this file on purpose: under
`python -m agentlog.cli` it executes the moment the interpreter reaches it, so
any command defined below it would never register.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from agentlog import install, pipeline, workspace
from agentlog.core import config as config_module
from agentlog.core.errors import AgentlogError
from agentlog.core.logging import configure, get_logger
from agentlog.domains.anchors import git
from agentlog.domains.anchors import service as anchors_service
from agentlog.domains.extraction import service as extraction_service
from agentlog.domains.inject import service as inject_service
from agentlog.domains.retrieval import quality
from agentlog.domains.retrieval import service as retrieval
from agentlog.domains.store import dedupe
from agentlog.domains.store import index as index_module
from agentlog.domains.store import log as log_module
from agentlog.domains.transcript import parser, reader, segmenter
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


# --------------------------------------------------------------------------
# v0.2 — the store, and reading it back
# --------------------------------------------------------------------------


def _data_dir(repo: Optional[Path]) -> tuple[Path, "config_module.Config"]:  # noqa: UP037,UP045
    root = git.repo_root(repo or Path.cwd()) or (repo or Path.cwd())
    cfg = config_module.load(root)
    return cfg.data_dir, cfg


def _sources(cfg) -> list[tuple[str, Path]]:  # noqa: ANN001
    return [(m.name, m.data_dir) for m in workspace.members(cfg)]


def _format_hit(hit, show_id: bool = True) -> list[str]:  # noqa: ANN001
    record = hit.record
    label = record.kind + (f"/{record.outcome}" if record.outcome else "")
    when = record.occurred_at.strftime("%Y-%m-%d %H:%M")
    head = f"{when}  {label}"
    if getattr(hit, "repo", ""):
        head = f"{when}  [{hit.repo}]  {label}"
    if record.evidence != "none":
        head += f"  [{record.evidence}]"
    if record.source == "inferred":
        head += "  (inferred)"
    if show_id:
        head += f"  {record.id}"
    lines = [head, f"    {record.summary}"]
    if record.detail:
        lines.append(f"    {record.detail}")
    a = record.anchors
    keys = []
    if a.settings:
        keys += [f"setting:{s}" for s in a.settings[:4]]
    if a.routes:
        keys += [f"route:{r}" for r in a.routes[:2]]
    if a.symbols:
        keys += [f"symbol:{s}" for s in a.symbols[:3]]
    if a.issue:
        keys.append(f"issue:{a.issue}")
    if a.head_sha:
        keys.append(f"at:{a.head_sha}")
    if keys:
        lines.append(f"    {'  '.join(keys)}")
    return lines


@app.command()
def backfill(
    days: int = typer.Option(30, "--days", help="How far back to read transcripts."),
    repo: Optional[Path] = typer.Option(None, "--repo", help="Repo root."),  # noqa: UP045
    write: bool = typer.Option(False, "--write", help="Actually extract and write records."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation."),
    max_cost: float = typer.Option(2.0, "--max-cost", help="Confirm above this estimate (USD)."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Process existing transcripts. Dry-run by default; --write to persist.

    This is the one-time step after install. It reads what is already on disk,
    so the log is useful immediately rather than only for work done from now on.
    """
    configure(log_level)
    data_dir, cfg = _data_dir(repo)
    root = cfg.repo_root
    found = reader.recent_sessions(days)
    paths = [p for p in found if pipeline.belongs_to(p, root)]
    if not found:
        _echo(f"No transcripts in the last {days} days.")
        return
    if not paths:
        _echo(f"None of the {len(found)} transcripts in the last {days} days ran in {root}.")
        return

    dry = pipeline.process(paths, cfg, client=None, repo_override=root)
    _echo(f"repo              {root}")
    _echo(f"transcripts       {dry.transcripts} of {len(found)} in the window ran here")
    _echo(f"segments          {dry.segments_total} ({dry.segments_new} not yet processed)")
    _echo(f"estimated cost    ${dry.estimated_cost_usd:.2f}")
    if dry.unreadable:
        _echo(f"unreadable        {len(dry.unreadable)}")

    if not write:
        _echo()
        _echo("Dry run. Re-run with --write to extract and store.")
        return
    if dry.segments_new == 0:
        _echo()
        _echo("Nothing new to process.")
        return

    if not yes and dry.estimated_cost_usd > max_cost:
        confirm = typer.confirm(f"This will spend roughly ${dry.estimated_cost_usd:.2f}. Continue?")
        if not confirm:
            _echo("Aborted.")
            raise typer.Exit(1)

    from agentlog.external.anthropic import AnthropicClient

    client = AnthropicClient(model=cfg.model)
    result = pipeline.process(paths, cfg, client=client, repo_override=root, write=True)
    _echo()
    _echo(f"records written   {len(result.records)}  ({result.dead_ends} dead ends)")
    _echo(f"segments dropped  {result.dropped}")
    _echo(f"log               {data_dir / 'records.jsonl'}")


@app.command(name="file")
def file_timeline(
    path: str = typer.Argument(..., help="Repo-relative path."),
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    include_inferred: bool = typer.Option(False, "--inferred", help="Include inferred records."),
) -> None:
    """Timeline for a file, oldest first."""
    configure("WARNING")
    _data, cfg = _data_dir(repo)
    hits = retrieval.timeline_across(_sources(cfg), path, include_inferred)
    if not hits:
        _echo(f"No records for {path}.")
        return
    _echo(f"{path} — {len(hits)} records")
    _echo()
    for hit in hits:
        for line in _format_hit(hit):
            _echo(line)
        _echo()


@app.command()
def search(
    query: str = typer.Argument(..., help="Keyword query (FTS5 syntax)."),
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    include_inferred: bool = typer.Option(False, "--inferred"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Keyword search across records, including their anchors."""
    configure("WARNING")
    _data, cfg = _data_dir(repo)
    hits = retrieval.search_across(_sources(cfg), query, include_inferred, limit)
    if not hits:
        _echo(f"No records matching {query!r}.")
        return
    for hit in hits[:limit]:
        for line in _format_hit(hit):
            _echo(line)
        _echo()


@app.command()
def setting(
    key: str = typer.Argument(..., help="Configuration key, e.g. fill_occlusion_holes."),
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    include_inferred: bool = typer.Option(False, "--inferred"),
) -> None:
    """Everything that ever touched a configuration key, oldest first.

    Including the sessions where it was renamed away — the old name lives on
    the removed side of the diff, which is exactly what you reach for when
    something worked two months ago and does not work now.
    """
    configure("WARNING")
    data_dir, _ = _data_dir(repo)
    hits = sorted(
        retrieval.by_anchor(data_dir, "setting", key, include_inferred),
        key=lambda h: h.record.created_at,
    )
    if not hits:
        _echo(f"No records for setting {key!r}.")
        return
    _echo(f"setting:{key} — {len(hits)} records")
    _echo()
    for hit in hits:
        for line in _format_hit(hit):
            _echo(line)
        _echo()


@app.command()
def show(
    record_id: str = typer.Argument(..., help="Record id, or a unique suffix of one."),
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
) -> None:
    """One record in full, as stored."""
    configure("WARNING")
    data_dir, _ = _data_dir(repo)
    record = retrieval.get(data_dir, record_id)
    if record is None:
        _echo(f"No record {record_id!r}.")
        raise typer.Exit(1)
    _echo(json.dumps(json.loads(record.model_dump_json()), indent=2))


@app.command()
def reindex(repo: Optional[Path] = typer.Option(None, "--repo")) -> None:  # noqa: UP045
    """Rebuild the index from the log. Always safe — the log is the truth."""
    configure("INFO")
    data_dir, _ = _data_dir(repo)
    written = index_module.rebuild(data_dir)
    _echo(f"reindexed {written} records from {data_dir / 'records.jsonl'}")


@app.command()
def status(repo: Optional[Path] = typer.Option(None, "--repo")) -> None:  # noqa: UP045
    """Counts, cursors, and where everything lives."""
    configure("WARNING")
    data_dir, cfg = _data_dir(repo)
    records = list(log_module.read_all(data_dir))
    cursors = dedupe.Cursors.load(data_dir)
    dead_ends = sum(1 for r in records if r.is_dead_end)
    inferred = sum(1 for r in records if r.source == "inferred")

    _echo(f"repo            {cfg.repo_root}")
    _echo(f"data            {data_dir}")
    _echo(f"records         {len(records)}  ({dead_ends} dead ends, {inferred} inferred)")
    _echo(f"sessions seen   {len(cursors.sessions)}")
    _echo(f"model           {cfg.model}")
    if records:
        first = min(r.occurred_at for r in records)
        last = max(r.occurred_at for r in records)
        _echo(f"span            {first:%Y-%m-%d} to {last:%Y-%m-%d}")
        kinds: dict[str, int] = {}
        for r in records:
            label = r.kind + (f"/{r.outcome}" if r.outcome else "")
            kinds[label] = kinds.get(label, 0) + 1
        _echo("kinds           " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))


@app.command()
def init(
    repo: Optional[Path] = typer.Option(None, "--repo", help="Repo root. Defaults to cwd."),  # noqa: UP045
    with_repo: list[Path] = typer.Option(  # noqa: B006,UP006
        None, "--with", help="A sibling repo to read alongside this one. Repeatable."
    ),
) -> None:
    """Install hooks and create .agentlog/. The one command you run to start."""
    configure("INFO")
    root = git.repo_root(repo or Path.cwd()) or (repo or Path.cwd()).resolve()
    result = install.install(root)

    cfg = config_module.load(root)
    linked = []
    for other in with_repo or []:
        cfg, changed = workspace.link(cfg, other)
        if changed:
            linked.append(other)
    if linked:
        workspace.save(cfg)

    _echo(f"repo       {root}")
    _echo(f"python     {result['python']}")
    _echo(f"settings   {result['settings']}")
    if result["hooks_changed"]:
        _echo(f"hooks      {', '.join(result['hooks_changed'])}")
    else:
        _echo("hooks      already installed")
    _echo(f"data       {result['data_dir']}  (gitignored)")
    _echo(f"skill      {result['skill']}")
    _echo(
        f"CLAUDE.md  {'added' if result['claude_md_added'] else 'updated'}"
        " — how the agent learns it exists"
    )
    members = workspace.members(config_module.load(root))
    if len(members) > 1:
        _echo(f"reads with {', '.join(m.name for m in members[1:])}")
    _echo()
    _echo("Capture runs on PreCompact and SessionEnd; injection on SessionStart.")
    _echo("Neither needs an API key — staging and injection are local.")
    _echo()
    _echo(f"Next:  agentlog backfill --days 30 --repo {root}")


@app.command()
def stage(
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    days: int = typer.Option(2, "--days", help="How far back to look for new work."),
) -> None:
    """Queue new work units for extraction. Local only — no model, no key.

    This is what the capture hooks run. It parses, segments, resolves anchors
    and scrubs, then writes the scrubbed payload to a pending queue. It cannot
    fail on a missing key or a network problem, which is the point: a hook that
    can fail is a hook that eventually breaks someone's session.
    """
    configure("WARNING")
    data_dir, cfg = _data_dir(repo)
    root = cfg.repo_root
    paths = [p for p in reader.recent_sessions(days) if pipeline.belongs_to(p, root)]
    if not paths:
        return
    queued = pipeline.stage(paths, cfg, repo_override=root)
    if queued:
        _echo(f"queued {queued} work units for extraction")


@app.command()
def inject(
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    budget: int = typer.Option(0, "--budget", help="Token budget (0 = config default)."),
) -> None:
    """Print the context block for a starting session. Local only.

    Fenced and labelled as a record of prior work, never as instructions. A
    malicious `.agentlog/` in a cloned repo is a prompt-injection vector, so
    the block has to read as data being reported.
    """
    configure("WARNING")
    data_dir, cfg = _data_dir(repo)
    block = inject_service.build(data_dir, cfg, budget or cfg.token_budget)
    if block:
        _echo(block)


@app.command()
def drain(
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    results: Optional[Path] = typer.Option(  # noqa: UP045
        None, "--results", help="Directory of <hash>.json files produced elsewhere."
    ),
    limit: int = typer.Option(0, "--limit", help="Drain at most N queued units."),
) -> None:
    """Turn the queue into records.

    With --results, ingests JSON extracted by a session running on your plan.
    Without it, calls the model directly and needs ANTHROPIC_API_KEY.
    """
    configure("INFO")
    data_dir, cfg = _data_dir(repo)
    queued = pipeline.pending(cfg)
    if not queued:
        _echo("Nothing queued.")
        return

    if results is not None:
        result = pipeline.drain(cfg, results_dir=results, limit=limit)
    else:
        from agentlog.external.anthropic import AnthropicClient

        result = pipeline.drain(cfg, client=AnthropicClient(model=cfg.model), limit=limit)

    _echo(f"drained  {result.segments_new} of {len(queued)} queued")
    _echo(f"records  {len(result.records)}  ({result.dead_ends} dead ends)")
    if result.dropped:
        _echo(f"dropped  {result.dropped}")
    _echo(f"log      {data_dir / 'records.jsonl'}")


@app.command()
def lint(
    repo: Optional[Path] = typer.Option(None, "--repo"),  # noqa: UP045
    show: int = typer.Option(8, "--show", help="Example findings to print per check."),
) -> None:
    """Score the log against the shape the design asked for.

    Not a truth check — nothing here can tell whether a record is correct. It
    measures whether records are dead ends or restated events, and whether they
    name identifiers the anchors already carry. Use it to tune the prompt.
    """
    configure("WARNING")
    data_dir, _ = _data_dir(repo)
    records = list(log_module.read_all(data_dir))
    if not records:
        _echo("No records yet.")
        return

    report = quality.check(records)
    _echo(f"records          {report.total}")
    _echo(f"clean            {report.clean}  ({report.clean / report.total * 100:.0f}%)")
    _echo(f"dead ends        {report.dead_ends}  ({report.dead_end_share * 100:.0f}% of records)")
    _echo()
    _echo("kinds")
    for label, n in report.kinds.most_common():
        _echo(f"  {label:<22} {n:>4}  {n / report.total * 100:>3.0f}%")

    if not report.findings:
        _echo()
        _echo("No findings.")
        return

    _echo()
    _echo("findings")
    for name, n in report.by_check.most_common():
        _echo(f"  {name:<28} {n:>4}")
    _echo()
    for name, _n in report.by_check.most_common():
        examples = [f for f in report.findings if f.check == name][:show]
        _echo(f"{name}:")
        for finding in examples:
            _echo(f"  {finding.record_id[:14]}  {finding.detail[:70]}")
        _echo()


def main() -> None:  # pragma: no cover - console entry point
    try:
        app()
    except AgentlogError as exc:
        typer.echo(f"agentlog: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
