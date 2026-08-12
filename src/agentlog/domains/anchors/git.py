"""Git-derived anchors.

Zero network, zero model calls. Every function here is offline and gets its own
test suite against fixture repos.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

from agentlog.core.logging import get_logger

log = get_logger("anchors.git")

_TIMEOUT = 20


def _run(repo: Path, *args: str) -> str | None:
    """Run a git command, returning None on any failure.

    Anchors degrade rather than abort: a repo with no commits, a detached HEAD,
    or a missing git binary should still produce file anchors from the
    transcript itself.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if result.returncode != 0:
        log.debug("git %s exited %d: %s", " ".join(args), result.returncode, result.stderr.strip())
        return None
    return result.stdout


def is_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=_TIMEOUT, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def repo_root(path: Path) -> Path | None:
    out = _run(path, "rev-parse", "--show-toplevel")
    if not out:
        return None
    return Path(out.strip())


def branch(repo: Path) -> str | None:
    # `--show-current` answers on an unborn branch too, which `rev-parse` does
    # not. That case is common in exactly the sessions worth capturing: a
    # feature branch cut and worked on before its first commit.
    out = _run(repo, "branch", "--show-current")
    if out is not None and out.strip():
        return out.strip()
    out = _run(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if not out:
        return None
    name = out.strip()
    # A detached HEAD reports "HEAD", which is not a branch name and carries no
    # feature identity.
    return None if name in ("", "HEAD") else name


def ignored(repo: Path, paths: set[str]) -> set[str]:
    """The subset of `paths` that git ignores.

    Batched into one call. Without this, anything the tooling happens to write
    inside the repo — `.venv/`, `node_modules/`, build output — becomes a file
    anchor as soon as an agent reads it.
    """
    if not paths:
        return set()
    ordered = sorted(paths)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "--stdin"],
            input="\n".join(ordered),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("git check-ignore failed: %s", exc)
        return set()
    # 0 = some paths matched, 1 = none matched. Anything else is a real error.
    if result.returncode not in (0, 1):
        log.debug("git check-ignore exited %d: %s", result.returncode, result.stderr.strip())
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def head_sha(repo: Path, short: bool = True) -> str | None:
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    out = _run(repo, *args)
    if not out:
        return None
    return out.strip() or None


def issue_from_branch(branch_name: str | None, pattern: str) -> str | None:
    if not branch_name:
        return None
    try:
        match = re.search(pattern, branch_name)
    except re.error as exc:
        log.warning("invalid issue_pattern %r: %s", pattern, exc)
        return None
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def commits_in_window(
    repo: Path,
    start: datetime | None,
    end: datetime | None,
) -> tuple[str, ...]:
    """Commits authored inside the segment's time window."""
    if start is None or end is None:
        return ()
    out = _run(
        repo,
        "log",
        "--all",
        f"--since={start.isoformat()}",
        f"--until={end.isoformat()}",
        "--format=%h",
    )
    if not out:
        return ()
    return tuple(line.strip() for line in out.splitlines() if line.strip())


def tracked_files(repo: Path) -> set[str]:
    out = _run(repo, "ls-files")
    if not out:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def files_in_commits(repo: Path, shas: tuple[str, ...]) -> tuple[str, ...]:
    files: dict[str, None] = {}
    for sha in shas:
        out = _run(repo, "show", "--name-only", "--format=", sha)
        if not out:
            continue
        for line in out.splitlines():
            name = line.strip()
            if name:
                files[name] = None
    return tuple(files)


def _parse_hunks(out: str) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = (
                None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            )
        elif line.startswith("@@") and current:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2) or "1")
                if count > 0:
                    ranges.setdefault(current, []).append((start, start + count - 1))
    return ranges


def changed_line_ranges(repo: Path, sha: str) -> dict[str, list[tuple[int, int]]]:
    """Post-image line ranges touched by a commit, per file.

    Parsed from unified-diff hunk headers (`@@ -a,b +c,d @@`), used to map an
    edit back to the function or class that encloses it.
    """
    out = _run(repo, "show", "--unified=0", "--format=", sha)
    return _parse_hunks(out) if out else {}


def working_tree_line_ranges(repo: Path) -> dict[str, list[tuple[int, int]]]:
    """Post-image line ranges of uncommitted changes.

    This is the case that matters most. Broken versions do not get committed,
    so the work a dead end belongs to is usually still uncommitted when the
    session ends — without this, the exact records the tool exists to capture
    would carry no symbol anchors.
    """
    out = _run(repo, "diff", "HEAD", "--unified=0")
    return _parse_hunks(out) if out else {}


def _parse_diff_lines(out: str) -> dict[str, list[tuple[str, str]]]:
    changed: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = (
                None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            )
        elif line.startswith("--- ") or line.startswith("@@"):
            continue
        elif current and line[:1] in ("+", "-") and not line.startswith(("+++", "---")):
            changed.setdefault(current, []).append((line[0], line[1:]))
    return changed


def changed_lines(repo: Path, sha: str | None = None) -> dict[str, list[tuple[str, str]]]:
    """Added and removed lines per file, as `(sign, text)` pairs.

    Both sides are kept deliberately. When a config key is *renamed*, the old
    name only exists on the removed side — and the old name is what a search
    six weeks later will be reaching for, because it is the name that was in
    use when the thing last worked.
    """
    args = ["show", "--unified=0", "--format=", sha] if sha else ["diff", "HEAD", "--unified=0"]
    out = _run(repo, *args)
    return _parse_diff_lines(out) if out else {}


def file_at(repo: Path, sha: str, path: str) -> str | None:
    return _run(repo, "show", f"{sha}:{path}")


def read_file(repo: Path, path: str, sha: str | None = None) -> str | None:
    """Read a repo file, preferring the working tree and falling back to git.

    The working tree is what the agent was actually editing. Falling back to
    the commit matters for files that were later deleted or renamed.
    """
    candidate = repo / path
    try:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    if sha:
        return file_at(repo, sha, path)
    return None
