"""Compose anchors for a segment.

Offline and deterministic end to end: git plus source parsing, no network and
no model. If this file ever needs either, the design has gone wrong.
"""

from __future__ import annotations

from pathlib import Path

from agentlog.core.config import Config
from agentlog.domains.anchors import git, routes, settings, symbols
from agentlog.domains.anchors.schemas import Anchors
from agentlog.domains.transcript.schemas import Segment

_SOURCE_SUFFIXES = (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs")


def _merge_ranges(
    per_file: dict[str, list[tuple[int, int]]],
    extra: dict[str, list[tuple[int, int]]],
) -> None:
    for path, ranges in extra.items():
        per_file.setdefault(path, []).extend(ranges)


def resolve(segment: Segment, cfg: Config, repo: Path | None = None) -> Anchors:
    repo = repo or cfg.repo_root

    branch_name = git.branch(repo)
    if branch_name is None:
        # Fall back to what the session itself recorded — useful when HEAD has
        # since moved on or detached.
        for turn in segment.turns:
            if turn.git_branch and turn.git_branch != "HEAD":
                branch_name = turn.git_branch
                break

    head = git.head_sha(repo)
    commits = git.commits_in_window(repo, segment.started_at, segment.ended_at)

    files: dict[str, None] = {path: None for path in segment.files}
    for path in git.files_in_commits(repo, commits):
        files[path] = None

    line_ranges: dict[str, list[tuple[int, int]]] = {}
    diff_lines: dict[str, list[tuple[str, str]]] = {}
    for sha in commits:
        _merge_ranges(line_ranges, git.changed_line_ranges(repo, sha))
        for path, lines in git.changed_lines(repo, sha).items():
            diff_lines.setdefault(path, []).extend(lines)
    _merge_ranges(line_ranges, git.working_tree_line_ranges(repo))
    for path, lines in git.changed_lines(repo).items():
        diff_lines.setdefault(path, []).extend(lines)

    found_routes: dict[str, None] = {}
    found_symbols: dict[str, None] = {}
    found_settings: dict[str, None] = {}
    for path in files:
        for key in settings.extract(path, diff_lines.get(path, [])):
            found_settings[key] = None
        if not path.endswith(_SOURCE_SUFFIXES):
            continue
        source = git.read_file(repo, path, head)
        if source is None:
            continue
        for route in routes.extract(path, source):
            found_routes[route] = None
        ranges = line_ranges.get(path)
        if ranges:
            for symbol in symbols.extract(path, source, ranges):
                found_symbols[symbol] = None

    return Anchors(
        files=tuple(files),
        routes=tuple(found_routes),
        symbols=tuple(found_symbols),
        settings=tuple(found_settings),
        branch=branch_name,
        issue=git.issue_from_branch(branch_name, cfg.issue_pattern),
        commits=commits,
        head_sha=head,
    )
