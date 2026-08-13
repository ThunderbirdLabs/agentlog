"""Several repos, one answer.

A log lives in the repo it describes, because anchors are repo-relative and a
plain-text file that travels with its repo can be reviewed in a pull request
like anything else in it. But one agent often works across a backend and a
frontend, and asking it to know which log to consult defeats the point.

So: keep the logs separate, read them together. Each record is tagged with the
repo it came from at read time, which costs nothing and keeps the anchors
unambiguous — `worker/src/pipeline.py` means something different in each repo,
and merging them into one namespace would quietly conflate the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentlog.core.config import Config
from agentlog.core.config import load as load_config
from agentlog.core.logging import get_logger
from agentlog.domains.anchors import git

log = get_logger("workspace")


@dataclass(frozen=True)
class Member:
    name: str
    root: Path
    data_dir: Path


def members(cfg: Config) -> list[Member]:
    """This repo, plus any sibling it was told about.

    Missing siblings are warned about and skipped: a checkout someone deleted
    should degrade the answer, not break the command.
    """
    found = [Member(name=cfg.repo_root.name, root=cfg.repo_root, data_dir=cfg.data_dir)]
    seen = {cfg.repo_root.resolve()}

    for entry in cfg.repos:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (cfg.repo_root / candidate).resolve()
        root = git.repo_root(candidate) or candidate
        if not root.is_dir():
            log.warning("linked repo %s does not exist; skipping", root)
            continue
        if root.resolve() in seen:
            continue
        seen.add(root.resolve())
        found.append(Member(name=root.name, root=root, data_dir=root / ".agentlog"))
    return found


def link(cfg: Config, other: Path) -> tuple[Config, bool]:
    """Add a sibling to this repo's config. Returns whether anything changed."""
    root = git.repo_root(other) or other.resolve()
    existing = {Path(r).resolve() for r in cfg.repos}
    if root.resolve() in existing or root.resolve() == cfg.repo_root.resolve():
        return cfg, False
    return (
        Config(
            repo_root=cfg.repo_root,
            model=cfg.model,
            issue_pattern=cfg.issue_pattern,
            retention_days=cfg.retention_days,
            token_budget=cfg.token_budget,
            repos=(*cfg.repos, str(root)),
            segmentation=cfg.segmentation,
        ),
        True,
    )


def save(cfg: Config) -> Path:
    """Persist the parts of config a user is expected to edit."""
    import json

    from agentlog.core.config import CONFIG_FILENAME

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.data_dir / CONFIG_FILENAME
    existing = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    existing["repos"] = list(cfg.repos)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return path


def reload(cfg: Config) -> Config:
    return load_config(cfg.repo_root)
