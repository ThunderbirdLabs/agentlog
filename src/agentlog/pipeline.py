"""Composition: transcript in, records out.

The one place that knows about every domain. Domains do not import each other
across boundaries; they meet here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agentlog.core.config import Config
from agentlog.core.errors import RedactionError
from agentlog.core.logging import get_logger
from agentlog.core.redaction import scrub
from agentlog.domains.anchors import git
from agentlog.domains.anchors import service as anchors_service
from agentlog.domains.extraction import prompts
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
    """Whether a session did work in `repo`.

    The test is which files it *touched*, not where it was launched. People
    run an agent from a home directory or a workspace root and edit files in
    checkouts underneath — measured on one real machine, every session that
    did a month of work on a given repo was launched from somewhere else, so a
    cwd test would have excluded all of it.

    A cheap substring scan over the raw file, because tool inputs record
    absolute paths. Streamed rather than read whole: session files reach
    hundreds of megabytes.
    """
    root = str(repo.resolve())
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if root in line:
                    return True
    except OSError:
        return False
    return False


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


PENDING_DIRNAME = "pending"


def pending_dir(cfg: Config) -> Path:
    return cfg.data_dir / PENDING_DIRNAME


def stage(paths: list[Path], cfg: Config, repo_override: Path | None = None) -> int:
    """Parse, segment, anchor, scrub — and queue. No model, no network, no key.

    This is what a capture hook runs. Splitting staging from extraction is what
    lets the hook be unconditionally safe: there is nothing here that can fail
    on a missing credential, a rate limit, or an outage, so the hook cannot
    break the session that triggered it.

    Redaction happens here, before anything is written, so a queued payload is
    already safe to send.
    """
    queue = pending_dir(cfg)
    queue.mkdir(parents=True, exist_ok=True)
    cursors = dedupe.Cursors.load(cfg.data_dir)
    queued = 0

    for path in paths:
        try:
            transcript, repo = load(path, repo_override)
        except Exception as exc:  # noqa: BLE001 - a hook never aborts
            log.warning("could not read %s: %s", path, exc)
            continue

        for segment in cursors.unprocessed(segments_for(transcript, cfg)):
            digest = dedupe.segment_hash(segment)
            target = queue / f"{digest}.json"
            if target.exists():
                continue
            try:
                scrubbed = scrub(prompts.user_prompt(extraction_service.render(segment)))
            except RedactionError as exc:
                # Fail closed. Nothing unscrubbed is ever written to disk.
                log.error("redaction failed; segment dropped: %s", exc)
                cursors.mark(segment)
                continue

            anchors = anchors_service.resolve(segment, cfg, repo)
            payload = {
                "hash": digest,
                "session_id": segment.session_id,
                "start_turn": segment.start_turn,
                "end_turn": segment.end_turn,
                "occurred_at": segment.started_at.isoformat() if segment.started_at else None,
                "anchors": json.loads(anchors.model_dump_json()),
                "prompt": scrubbed.text,
            }
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            queued += 1

    cursors.save(cfg.data_dir)
    return queued


def pending(cfg: Config) -> list[Path]:
    queue = pending_dir(cfg)
    if not queue.is_dir():
        return []
    return sorted(queue.glob("*.json"))


def _record_from_payload(payload: dict, candidate, extractor: str) -> Record:  # noqa: ANN001
    from datetime import datetime

    from agentlog.domains.anchors.schemas import Anchors

    occurred = payload.get("occurred_at")
    return Record.build(
        candidate,
        Anchors.model_validate(payload["anchors"]),
        payload["session_id"],
        payload["start_turn"],
        payload["end_turn"],
        extractor,
        occurred_at=datetime.fromisoformat(occurred) if occurred else None,
    )


def drain(
    cfg: Config,
    client=None,  # noqa: ANN001 - ModelClient, or None when results are supplied
    results_dir: Path | None = None,
    limit: int = 0,
) -> RunResult:
    """Turn queued payloads into records.

    Two drivers, one queue. With a `client` this calls the model directly; with
    `results_dir` it ingests JSON produced elsewhere — which is how the same
    queue can be drained by a session running on a subscription instead of an
    API key. Either way the records are identical and the `extractor` field
    records which produced them.

    A drained payload is deleted. A payload that yields nothing is deleted too:
    re-running it would cost the same and learn the same nothing.
    """
    from agentlog.domains.extraction import prompts
    from agentlog.domains.extraction.schemas import ExtractionResponse

    result = RunResult()
    queued = pending(cfg)
    if limit > 0:
        queued = queued[:limit]

    for path in queued:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("unreadable queued payload %s: %s", path, exc)
            continue

        response = None
        extractor = prompts.extractor_id(cfg.model)
        if results_dir is not None:
            candidate_file = results_dir / f"{payload['hash']}.json"
            if not candidate_file.is_file():
                # No result supplied for this unit. Leave it queued — a partial
                # drain must not silently discard the rest of the backlog.
                continue
            try:
                response = ExtractionResponse.model_validate_json(
                    candidate_file.read_text(encoding="utf-8")
                )
            except Exception as exc:  # noqa: BLE001 - a bad result is a skip, not a crash
                log.warning("invalid result for %s: %s", payload["hash"], exc)
                result.dropped += 1
                continue
        elif client is not None:
            from agentlog.core.redaction import scrub

            scrubbed = scrub(payload["prompt"])
            for _attempt in (1, 2):
                try:
                    response = client.classify(prompts.SYSTEM, scrubbed, ExtractionResponse)
                except Exception as exc:  # noqa: BLE001
                    log.warning("extraction call failed: %s", exc)
                    response = None
                if response is not None:
                    break
            if response is None:
                result.dropped += 1
                result.segments_new += 1
                path.unlink(missing_ok=True)
                continue
        else:
            raise ValueError("drain needs either a client or a results directory")

        result.segments_new += 1
        result.input_chars += len(payload.get("prompt", ""))

        from agentlog.core.redaction import scrub_record_field

        for candidate in response.records:
            cleaned = candidate.model_copy(
                update={
                    "summary": scrub_record_field(candidate.summary),
                    "detail": scrub_record_field(candidate.detail),
                }
            )
            result.records.append(_record_from_payload(payload, cleaned, extractor))
        path.unlink(missing_ok=True)

    log_module.append(cfg.data_dir, result.records)
    return result
