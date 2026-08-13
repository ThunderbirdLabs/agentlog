"""Configuration.

Defaults live here. A repo may override them in `.agentlog/config.json`.
Secrets are never part of config — the Anthropic key is read from the
environment at the point of use and never persisted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from agentlog.core.errors import ConfigError

CONFIG_DIRNAME = ".agentlog"
CONFIG_FILENAME = "config.json"

# Haiku tier: this is classification, not reasoning.
DEFAULT_MODEL = "claude-haiku-4-5"


@dataclass(frozen=True)
class SegmentationConfig:
    """Thresholds for grouping turns into work units.

    A wrong segmentation costs a slightly noisy record, not a wrong one, so
    these are deliberately blunt and deterministic.
    """

    # A turn's files are compared against the union of files from the last
    # `window` file-bearing turns, not the whole segment. Without the window a
    # single ubiquitous file (a config, a test helper) glues an entire session
    # into one segment.
    window: int = 6
    # How many consecutive non-overlapping turns end a segment. One is too
    # eager: real work on `service.py` runs `test_service.py` between edits,
    # and a one-turn rule splits every edit/test cycle into its own segment.
    boundary_misses: int = 2
    # Hard ceiling so a long uniform session still splits.
    max_turns: int = 60
    # A long silence is a work boundary regardless of file overlap.
    max_gap_minutes: int = 120
    # Segments shorter than this are noise.
    min_turns: int = 2
    # Hold a segment open while a failure is unresolved, even if the file set
    # shifts. A fix usually touches a different file than the failing check
    # did, so without this "tried X, it failed, then Y worked" splits across
    # segments and the outcome becomes unknowable — which is the one thing a
    # timeline has to get right.
    hold_open_on_failure: bool = True
    # A segment with no files has no anchors, so nothing could ever retrieve
    # it. Dropping it is better than logging an unreachable record.
    require_files: bool = True


@dataclass(frozen=True)
class Config:
    repo_root: Path
    model: str = DEFAULT_MODEL
    # Branch names are the cheapest stable feature key available. Default
    # matches the common `LINEAR-123` shape; override per repo.
    issue_pattern: str = r"\b([A-Z][A-Z0-9]{1,9}-\d+)\b"
    retention_days: int = 30
    token_budget: int = 1500
    # Sibling repos whose logs this one reads alongside its own. One agent
    # working across a backend and a frontend should get one answer, not two —
    # but each repo keeps its own log, because anchors are repo-relative and a
    # log that travels with its repo can be reviewed and shared like any other
    # file in it.
    repos: tuple[str, ...] = ()
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    @property
    def data_dir(self) -> Path:
        return self.repo_root / CONFIG_DIRNAME


def _coerce(raw: dict, repo_root: Path) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")

    seg_raw = raw.get("segmentation") or {}
    if not isinstance(seg_raw, dict):
        raise ConfigError("segmentation must be an object")
    seg_known = {f.name for f in fields(SegmentationConfig)}
    seg_unknown = set(seg_raw) - seg_known
    if seg_unknown:
        raise ConfigError(f"unknown segmentation keys: {sorted(seg_unknown)}")

    values = {k: v for k, v in raw.items() if k != "segmentation" and k != "repo_root"}
    return Config(repo_root=repo_root, segmentation=SegmentationConfig(**seg_raw), **values)


def load(repo_root: Path) -> Config:
    """Load config for a repo, falling back to defaults.

    A malformed config file is an error, not a silent fallback — a user who
    edited it wants their values used.
    """
    path = repo_root / CONFIG_DIRNAME / CONFIG_FILENAME
    if not path.exists():
        cfg = Config(repo_root=repo_root)
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"could not read {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a JSON object")
        cfg = _coerce(raw, repo_root)

    env_model = os.environ.get("AGENTLOG_MODEL")
    if env_model:
        cfg = Config(
            repo_root=cfg.repo_root,
            model=env_model,
            issue_pattern=cfg.issue_pattern,
            retention_days=cfg.retention_days,
            token_budget=cfg.token_budget,
            repos=cfg.repos,
            segmentation=cfg.segmentation,
        )
    return cfg
