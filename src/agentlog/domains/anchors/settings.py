"""Settings anchors: the configuration keys a change turned.

Files, routes and symbols are the right keys for application code. They are the
wrong keys for work against a library or an SDK, where the unit of work is a
knob — a densification preset, a timeout, a feature flag, an occlusion-fill
option. Nobody searches for the file that configured the thing; they search for
the thing.

Keys are read out of changed diff lines, so they are computed like every other
anchor. The model never invents one.

Both sides of the diff are kept. A renamed key only exists on the removed side,
and the old name is what someone reaches for later, because it is the name that
was in use the last time it worked.
"""

from __future__ import annotations

import keyword
import re

# key = value / key: value — Python kwargs and assignments, YAML, TOML, .env.
_ASSIGNMENT = re.compile(r"(?<![\w.\-])(?P<key>[A-Za-z_][A-Za-z0-9_]{2,63})\s*(?:=|:)(?!=)")
# "key": value — JSON and dict literals.
_QUOTED_KEY = re.compile(r"[\"'](?P<key>[A-Za-z_][A-Za-z0-9_]{2,63})[\"']\s*:")
# --flag-name / --flag_name on a command line.
_CLI_FLAG = re.compile(r"(?<![\w-])--(?P<key>[A-Za-z][A-Za-z0-9_-]{2,63})")

_CONFIG_SUFFIXES = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
)

# Structural words that appear in `x:`/`x =` position but configure nothing.
_STOPWORDS = frozenset(
    {
        "self",
        "cls",
        "args",
        "kwargs",
        "type",
        "def",
        "let",
        "var",
        "const",
        "int",
        "str",
        "bool",
        "float",
        "dict",
        "list",
        "tuple",
        "set",
        "none",
        "true",
        "false",
        "null",
        "case",
        "default",
        "http",
        "https",
        "note",
        "warning",
        "error",
        "todo",
        "returns",
        "raises",
        "param",
        "params",
        "result",
        "value",
        "data",
        "item",
        "items",
        "line",
        "text",
        "name",
        "path",
        "file",
        "url",
        "out",
        "res",
        "req",
        "obj",
        "key",
        "val",
    }
)


def _plausible(key: str) -> bool:
    lowered = key.lower()
    if lowered in _STOPWORDS or keyword.iskeyword(lowered):
        return False
    # A knob is named, not abbreviated: `fill_occlusion_holes`, `MAX_WORKERS`,
    # `keypointDensity`. A bare lowercase word is far more often a local.
    if "_" in key or key.isupper():
        return True
    return bool(re.search(r"[a-z][A-Z]", key))


def extract(file_path: str, changed: list[tuple[str, str]]) -> list[str]:
    """Configuration keys touched by the changed lines of one file.

    `changed` is `(sign, text)` pairs from a unified diff, where sign is `+`
    or `-`. Comment lines are skipped: a key mentioned in a comment was not
    turned, and a stale comment is exactly the kind of thing that sends someone
    chasing a knob that no longer exists.
    """
    if not file_path.endswith(_CONFIG_SUFFIXES):
        return []

    found: dict[str, None] = {}
    for _sign, text in changed:
        stripped = text.strip()
        if not stripped or stripped.startswith(("#", "//", "*", "/*")):
            continue
        for pattern in (_QUOTED_KEY, _ASSIGNMENT, _CLI_FLAG):
            for match in pattern.finditer(text):
                key = match.group("key")
                normalized = key.replace("-", "_") if pattern is _CLI_FLAG else key
                if _plausible(normalized):
                    found[normalized] = None
    return list(found)
