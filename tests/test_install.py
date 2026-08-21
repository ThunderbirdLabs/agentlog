"""Installing into a repo, and the hook scripts it writes."""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

from agentlog import install


def test_init_creates_hooks_and_data_dir(git_repo: Path) -> None:
    result = install.install(git_repo)
    settings = json.loads((git_repo / install.SETTINGS_PATH).read_text(encoding="utf-8"))

    assert set(settings["hooks"]) == {"PreCompact", "SessionEnd", "SessionStart"}
    assert (git_repo / ".agentlog" / ".gitignore").is_file()
    for script in result["scripts"]:
        path = Path(script)
        assert path.is_file()
        assert path.stat().st_mode & stat.S_IXUSR, "hooks must be executable"


def test_hooks_bake_in_an_absolute_interpreter(git_repo: Path) -> None:
    """Hooks run without a profile, so `command -v python` finds nothing."""
    install.install(git_repo, python="/opt/weird/bin/python3.12")
    body = (git_repo / ".claude" / "hooks" / "agentlog-precompact.sh").read_text(encoding="utf-8")
    assert "/opt/weird/bin/python3.12" in body
    assert "command -v python3" in body, "and a fallback if that path goes away"


def test_every_hook_exits_zero_even_when_agentlog_is_gone(git_repo: Path) -> None:
    """A capture failure must never break the user's session."""
    install.install(git_repo, python="/nonexistent/python")
    hooks = sorted((git_repo / ".claude" / "hooks").glob("agentlog-*.sh"))
    assert hooks
    for hook in hooks:
        done = subprocess.run(
            ["/bin/sh", str(hook)], input="{}", capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 0, f"{hook.name} exited {done.returncode}"


def test_init_is_idempotent(git_repo: Path) -> None:
    install.install(git_repo)
    first = json.loads((git_repo / install.SETTINGS_PATH).read_text(encoding="utf-8"))
    second_result = install.install(git_repo)
    second = json.loads((git_repo / install.SETTINGS_PATH).read_text(encoding="utf-8"))

    assert first == second
    assert second_result["hooks_changed"] == [], "a re-run should report no changes"


def test_existing_hooks_from_other_tools_survive(git_repo: Path) -> None:
    """Merging into someone's settings must not disturb what is already there."""
    settings_path = git_repo / install.SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {
                    "SessionStart": [{"hooks": [{"type": "command", "command": "other-tool.sh"}]}],
                    "Stop": [{"hooks": [{"type": "command", "command": "notify.sh"}]}],
                },
            }
        ),
        encoding="utf-8",
    )
    install.install(git_repo)
    after = json.loads(settings_path.read_text(encoding="utf-8"))

    assert after["model"] == "opus"
    assert after["hooks"]["Stop"][0]["hooks"][0]["command"] == "notify.sh"
    commands = [h["command"] for m in after["hooks"]["SessionStart"] for h in m["hooks"]]
    assert "other-tool.sh" in commands
    assert any("agentlog" in c for c in commands)


def test_a_reinstall_after_moving_updates_the_interpreter(git_repo: Path) -> None:
    """Upgrading or moving the venv is fixed by re-running init."""
    install.install(git_repo, python="/old/python")
    old = (git_repo / ".claude" / "hooks" / "agentlog-sessionstart.sh").read_text(encoding="utf-8")
    assert "/old/python" in old

    install.install(git_repo, python="/new/python")
    new = (git_repo / ".claude" / "hooks" / "agentlog-sessionstart.sh").read_text(encoding="utf-8")
    assert "/new/python" in new
    assert "/old/python" not in new


def test_claude_md_block_is_added_and_replaced(git_repo: Path) -> None:
    """Without this the CLI is installed and the agent never reaches for it."""
    (git_repo / "CLAUDE.md").write_text("# My project\n\nExisting notes.\n", encoding="utf-8")

    install.install(git_repo)
    text = (git_repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Existing notes." in text
    assert "agentlog file <path>" in text
    assert text.count(install._CLAUDE_MD_BEGIN) == 1

    install.install(git_repo)
    again = (git_repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert again.count(install._CLAUDE_MD_BEGIN) == 1, "a re-run must not duplicate the block"
    assert "Existing notes." in again


def test_the_drain_skill_is_installed(git_repo: Path) -> None:
    """Extraction needs a model; the skill is the path that needs no API key."""
    result = install.install(git_repo)
    path = Path(result["skill"])
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), "a skill needs frontmatter to be discoverable"
    assert "name: agentlog-drain" in text
    assert "agentlog drain --repo" in text, "it must say how to ingest the results"
    assert "ten" in text, "draining hundreds in one turn exhausts context"
    # The rules and schema are not in the payload; the skill has to point at
    # the file that holds them or whoever drains has only the slice.
    assert "_INSTRUCTIONS.md" in text


def test_hook_commands_survive_a_repo_move(git_repo: Path) -> None:
    """Absolute paths in settings.json die silently when a repo is moved.

    Found the hard way: a directory reorganisation orphaned every hook, nothing
    errored, and capture stopped for a week while looking like the tool simply
    did not work.
    """
    install.install(git_repo)
    settings = json.loads((git_repo / install.SETTINGS_PATH).read_text(encoding="utf-8"))
    for event in ("PreCompact", "SessionEnd", "SessionStart"):
        command = settings["hooks"][event][0]["hooks"][0]["command"]
        assert "$CLAUDE_PROJECT_DIR" in command
        assert str(git_repo) not in command, "no absolute path may be baked in"


def test_doctor_catches_an_orphaned_hook(git_repo: Path) -> None:
    install.install(git_repo)
    assert all(ok for _n, ok, _d in install.diagnose(git_repo))

    settings_path = git_repo / install.SETTINGS_PATH
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["hooks"]["SessionStart"][0]["hooks"][0]["command"] = "/gone/agentlog-sessionstart.sh"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    failures = [name for name, ok, _d in install.diagnose(git_repo) if not ok]
    assert "SessionStart hook registered" in failures


def test_the_log_is_committed_but_derived_files_are_not(git_repo: Path) -> None:
    """Graphify's split: ship the artifact, ignore the cache."""
    from agentlog.domains.store import log as log_module

    install.install(git_repo)
    ignored = (git_repo / ".agentlog" / ".gitignore").read_text(encoding="utf-8")
    assert "index.db" in ignored and "pending/" in ignored
    assert "records.jsonl" not in ignored, "the log is the thing worth sharing"

    subprocess.run(["git", "-C", str(git_repo), "add", "-A"], check=True, capture_output=True)
    tracked = subprocess.run(
        ["git", "-C", str(git_repo), "ls-files", ".agentlog"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert ".agentlog/.gitignore" in tracked
    assert "index.db" not in tracked

    log_module.append(git_repo / ".agentlog", [])
    assert (git_repo / ".agentlog").is_dir()


def test_a_users_own_gitignore_is_left_alone(git_repo: Path) -> None:
    from agentlog.domains.store import log as log_module

    data = git_repo / ".agentlog"
    data.mkdir(parents=True, exist_ok=True)
    (data / ".gitignore").write_text("# mine\n*\n", encoding="utf-8")
    log_module.ensure_dir(data)
    assert (data / ".gitignore").read_text(encoding="utf-8") == "# mine\n*\n"
