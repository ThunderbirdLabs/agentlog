"""What a prompt is about, in anchor terms.

Per-turn injection lives or dies on precision. This runs on *every* prompt, and
a block that appears when it should not is worse than one that never appears:
it burns context, trains the reader to skim past it, and eventually gets the
hook turned off.

So the bar is an exact anchor match wherever possible — a path the person
actually typed, a setting name they actually named. Free-text search is the
fallback, and it is deliberately narrow.
"""

from __future__ import annotations

import re

# A path: has a slash, or is a bare filename with a code-ish extension.
_PATH = re.compile(
    r"(?:[\w.\-]+/)+[\w.\-]+\.\w{1,10}|\b[\w.\-]+\.(?:py|ts|tsx|js|jsx|go|rs|rb|java|kt|sql|ya?ml|toml|json|sh)\b"
)
# A setting: snake_case, SCREAMING_CASE, or camelCase. Same shape the settings
# anchor extractor accepts, so the two agree on what counts as a knob.
_SETTING = re.compile(
    r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{4,}")

# Words that carry no retrieval signal. Kept short on purpose: this is a stop
# list for a fallback path, not an attempt at language modelling.
_STOP = frozenset(
    [
        "about",
        "after",
        "again",
        "against",
        "because",
        "before",
        "being",
        "below",
        "between",
        "both",
        "should",
        "could",
        "would",
        "there",
        "these",
        "those",
        "through",
        "under",
        "until",
        "while",
        "where",
        "which",
        "whose",
        "write",
        "wrote",
        "running",
        "please",
        "thanks",
        "maybe",
        "might",
        "their",
        "theirs",
        "check",
        "checked",
        "checking",
        "change",
        "changed",
        "changes",
        "create",
        "created",
        "creating",
        "delete",
        "deleted",
        "update",
        "updated",
        "using",
        "used",
        "file",
        "files",
        "code",
        "error",
        "errors",
        "issue",
        "issues",
        "problem",
        "problems",
        "right",
        "think",
        "thing",
        "things",
        "stuff",
        "still",
        "just",
        "like",
        "really",
        "claude",
        "agent",
        "session",
        "repo",
        "branch",
        "commit",
    ]
)


def paths_in(prompt: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _PATH.finditer(prompt):
        seen[match.group(0).strip("`'\"(),;:")] = None
    return list(seen)


def settings_in(prompt: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _SETTING.finditer(prompt):
        token = match.group(0)
        # Two characters is not a knob, and a bare word is handled by search.
        if len(token) >= 4:
            seen[token] = None
    return list(seen)


def query_for(prompt: str) -> str:
    """An FTS5 query built from the prompt's distinctive words.

    OR rather than AND: a person describing a symptom rarely uses the same
    words the record does, so requiring all of them returns nothing. Ranking
    and the record cap handle the looseness.
    """
    words = []
    for match in _WORD.finditer(prompt):
        word = match.group(0).lower()
        if word in _STOP or word in words:
            continue
        words.append(word)
    if len(words) < 2:
        # One distinctive word is not enough to justify interrupting a turn.
        return ""
    # Quote each term: an unescaped hyphen or quote is an FTS5 syntax error.
    return " OR ".join(f'"{w}"' for w in words[:12])
