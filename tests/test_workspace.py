"""Reading several repos as one."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentlog import workspace
from agentlog.core.config import Config


def _repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    return path


def test_a_lone_repo_is_its_own_workspace(git_repo: Path) -> None:
    found = workspace.members(Config(repo_root=git_repo))
    assert [m.root for m in found] == [git_repo]


def test_linking_a_sibling_reads_both(tmp_path: Path, git_repo: Path) -> None:
    frontend = _repo(tmp_path / "frontend")
    cfg, changed = workspace.link(Config(repo_root=git_repo), frontend)
    assert changed
    found = workspace.members(cfg)
    assert [m.name for m in found] == [git_repo.name, "frontend"]
    assert found[1].data_dir == frontend / ".agentlog"


def test_linking_is_idempotent_and_self_safe(tmp_path: Path, git_repo: Path) -> None:
    frontend = _repo(tmp_path / "frontend")
    cfg, _ = workspace.link(Config(repo_root=git_repo), frontend)
    cfg, changed = workspace.link(cfg, frontend)
    assert not changed
    _, self_changed = workspace.link(cfg, git_repo)
    assert not self_changed, "a repo must not link to itself"


def test_a_missing_sibling_is_skipped_not_fatal(tmp_path: Path, git_repo: Path) -> None:
    """A deleted checkout should degrade the answer, not break the command."""
    cfg = Config(repo_root=git_repo, repos=(str(tmp_path / "gone"),))
    assert [m.root for m in workspace.members(cfg)] == [git_repo]


def test_links_round_trip_through_config(tmp_path: Path, git_repo: Path) -> None:
    frontend = _repo(tmp_path / "frontend")
    cfg, _ = workspace.link(Config(repo_root=git_repo), frontend)
    workspace.save(cfg)
    reloaded = workspace.reload(cfg)
    assert [m.name for m in workspace.members(reloaded)] == [git_repo.name, "frontend"]
