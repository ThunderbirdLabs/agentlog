"""Composition: transcript in, records out.

The one place that knows about every domain. Domains do not import each other
across boundaries; they meet here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agentlog.core.config import Config
from agentlog.core.logging import get_logger
from agentlog.domains.anchors import git
from agentlog.domains.anchors import service as anchors_service
from agentlog.domains.extraction import service as extraction_service
from agentlog.domains.store import dedupe
from agentlog.domains.store import log as log_module
from agentlog.domains.store.schemas import Record
from agentlog.domains.transcript import parser, segmenter
from agentlog.domains.transcript.schemas import Segment, Transcript

log = get_logger("pipeline")

# Haiku tier, per million tokens. Used only to show an estimate before
# spending anything — never to decide whether to spend it.
_INPUT_PER_MTOK = 1.0
_OUTPUT_PER_MTOK = 5.0
_CHARS_PER_TOKEN = 4
_EST_OUTPUT_TOKENS = 400


@dataclass
class RunResult:
    transcripts: int = 0
    segments_total: int = 0
    segments_new: int = 0
    records: list[Record] = field(default_factory=list)
    dropped: int = 0
    input_chars: int = 0
    unreadable: list[str] = field(default_factory=list)

    @property
    def estimated_cost_usd(self) -> float:
        tokens_in = self.input_chars / _CHARS_PER_TOKEN
        tokens_out = self.segments_new * _EST_OUTPUT_TOKENS
        return tokens_in / 1e6 * _INPUT_PER_MTOK + tokens_out / 1e6 * _OUTPUT_PER_MTOK

    @property
    def dead_ends(self) -> int:
        return sum(1 for record in self.records if record.is_dead_end)


def load(path: Path, repo_override: Path | None = None) -> tuple[Transcript, Path]:
    """Parse a transcript and resolve its paths against the repo.

    Two normalisation passes: permissive to gather candidates, then strict
    against what git tracks or does not ignore. Without the second, anything
    the tooling writes inside the repo becomes a file anchor.
    """
    transcript = parser.parse_file(path)
    if repo_override is not None:
        candidate = repo_override.resolve()
    elif transcript.cwd:
        candidate = Path(transcript.cwd)
    else:
        candidate = Path.cwd()
    repo = git.repo_root(candidate) or candidate

    tracked = git.tracked_files(repo)
    permissive = parser.normalize_paths(transcript, repo)
    candidates = {p for turn in permissive.turns for p in turn.paths}
    allowed = tracked | (candidates - git.ignored(repo, candidates))
    return parser.normalize_paths(transcript, repo, allowed, strict=True), repo


def segments_for(transcript: Transcript, cfg: Config) -> list[Segment]:
    return segmenter.segment(transcript, cfg.segmentation)


def belongs_to(path: Path, repo: Path) -> bool:
    """Whether a transcript's session actually ran inside `repo`.

    A log is repo-scoped: its anchors are repo-relative paths and its commits
    are that repo's commits. Folding a session from another checkout into it
    produces records keyed to paths that do not exist here — which is worse
    than no record, because it retrieves and then misleads.

    Read from the `cwd` the session recorded, not from the transcript's
    location on disk.
    """
    try:
        cwd = None
        for line in path.open("r", encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("cwd"), str):
                cwd = obj["cwd"]
                break
    except OSError:
        return False
    if not cwd:
        return False
    session_repo = git.repo_root(Path(cwd))
    if session_repo is not None:
        return session_repo.resolve() == repo.resolve()
    try:
        Path(cwd).resolve().relative_to(repo.resolve())
    except ValueError:
        return False
    return True


def process(
    paths: list[Path],
    cfg: Config,
    client,  # noqa: ANN001 - ModelClient, or None for a dry run
    repo_override: Path | None = None,
    write: bool = False,
    limit_segments: int = 0,
) -> RunResult:
    """Run the pipeline over transcripts.

    With `client=None` nothing is extracted and nothing is written — the run
    reports what it *would* do, which is how `backfill` can quote a cost before
    spending anything.
    """
    result = RunResult()
    cursors = dedupe.Cursors.load(cfg.data_dir)

    for path in paths:
        try:
            transcript, repo = load(path, repo_override)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end a backfill
            log.warning("could not read %s: %s", path, exc)
            result.unreadable.append(str(path))
            continue

        result.transcripts += 1
        segments = segments_for(transcript, cfg)
        result.segments_total += len(segments)

        pending = cursors.unprocessed(segments)
        if limit_segments > 0:
            pending = pending[:limit_segments]
        result.segments_new += len(pending)

        for segment in pending:
            result.input_chars += len(extraction_service.render(segment))
            if client is None:
                continue

            anchors = anchors_service.resolve(segment, cfg, repo)
            extracted = extraction_service.extract(segment, anchors, client, cfg.model)
            if extracted.dropped:
                result.dropped += 1
                # A dropped segment is still marked: retrying it would fail the
                # same way and cost the same money.
                cursors.mark(segment)
                continue

            for candidate in extracted.candidates:
                result.records.append(
                    Record.build(
                        candidate,
                        anchors,
                        segment.session_id,
                        segment.start_turn,
                        segment.end_turn,
                        extracted.extractor,
                        occurred_at=segment.started_at,
                    )
                )
            cursors.mark(segment)

    if write and client is not None:
        log_module.append(cfg.data_dir, result.records)
        cursors.save(cfg.data_dir)

    return result
