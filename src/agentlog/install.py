"""Installing into a repo: hooks, directories, and the settings merge.

Two rules shape everything here.

**Never write to a user's repo without explicit opt-in.** `init` is the opt-in.
It touches exactly two things — `.claude/settings.json` and `.agentlog/` — and
it says what it changed.

**Resolve the interpreter at install time.** Hooks run under a non-interactive
shell that does not source `.bashrc` or `.zshrc`, so a bare `command -v python`
fails for anyone on pyenv, conda, or a venv. The absolute path is baked into
the script, with a runtime fallback in case the venv later moves.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from agentlog.core.logging import get_logger

log = get_logger("install")

HOOKS_DIRNAME = "hooks"
SETTINGS_PATH = Path(".claude") / "settings.json"

# The two capture hooks. PreCompact is the important one: it fires at the exact
# moment a long session is about to lose its own history.
_CAPTURE_EVENTS = ("PreCompact", "SessionEnd")
# The injection hook. Runs on every new session, costs nothing, needs no key.
_INJECT_EVENTS = ("SessionStart",)

_MARKER = "agentlog"

_SCRIPT_TEMPLATE = """#!/bin/sh
# Installed by `agentlog init`. Safe to delete; re-run init to restore.
#
# Always exits 0. A capture failure must never break the user's session, so
# every path here ends in `exit 0` — including the one where agentlog is gone.
#
# The interpreter path is resolved at install time: hooks run under a
# non-interactive shell that does not source a profile, so `command -v python`
# finds nothing for anyone on pyenv, conda, or a venv.

AGENTLOG_PYTHON="{python}"
if [ ! -x "$AGENTLOG_PYTHON" ]; then
  AGENTLOG_PYTHON="$(command -v python3 2>/dev/null)" || exit 0
  [ -n "$AGENTLOG_PYTHON" ] || exit 0
fi

{body}

exit 0
"""

# Capture is fire-and-forget: detached, output discarded, never blocking the
# session it was triggered from.
_CAPTURE_BODY = """cat > /dev/null 2>&1   # drain the hook payload on stdin

(
  "$AGENTLOG_PYTHON" -m agentlog.cli stage --repo "{repo}" >/dev/null 2>&1
) &

"""

# Injection is the opposite: it must run in the foreground, because its stdout
# is what lands in the agent's context.
_INJECT_BODY = """cat > /dev/null 2>&1

"$AGENTLOG_PYTHON" -m agentlog.cli inject --repo "{repo}" 2>/dev/null

"""


def resolve_python() -> str:
    """The interpreter to bake into the hooks.

    Prefers the one running this, which is the one that has agentlog importable.
    """
    return sys.executable or shutil.which("python3") or "python3"


def hook_scripts(repo: Path, python: str) -> dict[str, str]:
    scripts = {}
    for event in _CAPTURE_EVENTS:
        scripts[event] = _SCRIPT_TEMPLATE.format(
            python=python, body=_CAPTURE_BODY.format(repo=repo)
        )
    for event in _INJECT_EVENTS:
        scripts[event] = _SCRIPT_TEMPLATE.format(python=python, body=_INJECT_BODY.format(repo=repo))
    return scripts


def write_hooks(repo: Path, python: str) -> list[Path]:
    directory = repo / ".claude" / HOOKS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for event, body in hook_scripts(repo, python).items():
        path = directory / f"agentlog-{event.lower()}.sh"
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        written.append(path)
    return written


def _entry(command: str) -> dict:
    return {"type": "command", "command": command}


def merge_settings(existing: dict, repo: Path) -> tuple[dict, list[str]]:
    """Add agentlog's hooks to a settings object without disturbing others.

    Merged rather than replaced, and idempotent: re-running `init` after an
    upgrade rewrites agentlog's own entries and leaves every other hook alone.
    """
    settings = json.loads(json.dumps(existing))  # deep copy
    hooks = settings.setdefault("hooks", {})
    changed = []

    for event in (*_CAPTURE_EVENTS, *_INJECT_EVENTS):
        script = repo / ".claude" / HOOKS_DIRNAME / f"agentlog-{event.lower()}.sh"
        command = str(script)
        matchers = hooks.setdefault(event, [])
        if not isinstance(matchers, list):
            log.warning("settings.hooks.%s is not a list; leaving it alone", event)
            continue

        ours = None
        for matcher in matchers:
            if not isinstance(matcher, dict):
                continue
            for hook in matcher.get("hooks", []):
                if isinstance(hook, dict) and _MARKER in str(hook.get("command", "")):
                    ours = hook
                    break
            if ours:
                break

        if ours is not None:
            if ours.get("command") != command:
                ours["command"] = command
                changed.append(f"{event} (updated)")
            continue

        matchers.append({"hooks": [_entry(command)]})
        changed.append(event)

    return settings, changed


def install(repo: Path, python: str | None = None) -> dict:
    """Install hooks and create `.agentlog/`. Returns what changed."""
    from agentlog.domains.store import log as log_module

    python = python or resolve_python()
    scripts = write_hooks(repo, python)

    settings_path = repo / SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Never clobber a settings file we cannot parse — it is the user's,
            # and a bad merge would break every other hook they rely on.
            raise
    else:
        existing = {}

    merged, changed = merge_settings(existing, repo)
    settings_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    data_dir = log_module.ensure_dir(repo / ".agentlog")

    return {
        "python": python,
        "scripts": [str(p) for p in scripts],
        "settings": str(settings_path),
        "hooks_changed": changed,
        "data_dir": str(data_dir),
    }
