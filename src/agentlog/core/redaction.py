"""Transcript-grade scrubbing.

The highest-risk component in the tool. Transcripts are a worse threat model
than commit messages: people paste whole `.env` files, database URLs with
inline passwords, customer records, internal hostnames.

Two rules govern everything here.

1. **Fail closed.** If scrubbing raises, the caller drops the segment. A lost
   record is a minor loss; a written credential is not.
2. **Nothing reaches the network unscrubbed.** The only way to obtain a
   `Scrubbed` value is to call `scrub()`. `external.anthropic` accepts
   `Scrubbed` and nothing else, so "redaction ran first" is enforced by the
   type, not by convention. `tests/test_ordering.py` asserts it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentlog.core.errors import RedactionError

# --------------------------------------------------------------------------
# Pass 1 patterns
# --------------------------------------------------------------------------

_SECRET_WORDS = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "credential",
    "private",
    "key",
)

# Recognizable key shapes. Each is prefix-anchored on purpose: a generic
# high-entropy detector would eat git SHAs, base64 payloads, and UUIDs, and
# ordinary prose has to survive this pass untouched.
_KEY_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("anthropic_or_openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("stripe_key", re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{10,}")),
    ("stripe_webhook_secret", re.compile(r"\bwhsec_[A-Za-z0-9]{16,}")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{30,}")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    (
        "bearer_token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]{16,}"),
    ),
)

# scheme://user:password@host — covers postgres, mysql, mongodb, redis, amqp
# and the `+driver` variants (postgresql+asyncpg://...).
_CONNECTION_STRING = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.\-]*)://([^\s/:@]+):([^\s/@]+)@")

# KEY=value / key: value where the identifier names a secret. Not anchored to
# the start of a line: `run with DB_PASSWORD=hunter2 npm start` is exactly the
# shape that shows up mid-sentence in a transcript.
_ASSIGNMENT = re.compile(
    r"(?<![\w.])(?:export[ \t]+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"[ \t]*(?P<sep>[:=])[ \t]*(?P<value>\"[^\"\n]*\"|'[^'\n]*'|\S+)"
)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Deliberately conservative: separators are required, so version strings
# (1.2.3), dates (2026-08-11), and bare digit runs do not match.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[ .\-]?)?(?:\(\d{3}\)[ .\-]?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?![\w])"
)

_FENCE = re.compile(r"(?m)^(?P<fence>```+)[^\n]*\n(?P<body>.*?)^(?P=fence)[ \t]*$", re.DOTALL)
_ENVISH_LINE = re.compile(r"^[ \t]*(?:export[ \t]+)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=")


def _identifier_names_a_secret(key: str) -> bool:
    """Decide whether `key` in `key=value` is a secret name.

    Three accepted shapes, in order of how much they look like configuration
    rather than English:

      * all-caps env style        -> `API_TOKEN`, `SECRET`
      * snake case                -> `api_key`, `db_password`
      * the bare word, lowercased -> `password`, `token`

    The bare-word rule requires lowercase on purpose. A capitalized English
    word such as `Token` is prose, not configuration, which is what keeps
    "Token: 42 tokens used across the run." intact.
    """
    lowered = key.lower()
    if not any(word in lowered for word in _SECRET_WORDS):
        return False
    if key.isupper():
        return True
    if "_" in key:
        return True
    return key.islower() and lowered in _SECRET_WORDS


def _scrub_assignments(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group("key")
        if not _identifier_names_a_secret(key):
            return match.group(0)
        return f"{key}{match.group('sep')}[REDACTED:secret]"

    return _ASSIGNMENT.sub(repl, text)


def _scrub_env_fences(text: str) -> str:
    """Blank the body of fenced blocks that are predominantly `KEY=value`.

    A pasted `.env` file is the single most common way a whole credential set
    lands in a transcript, and per-line rules only catch the lines whose names
    look secret. If the block *is* an env file, none of it survives.
    """

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        lines = [line for line in body.splitlines() if line.strip()]
        if len(lines) < 2:
            return match.group(0)
        envish = sum(1 for line in lines if _ENVISH_LINE.match(line))
        if envish < 2 or envish * 2 < len(lines):
            return match.group(0)
        fence = match.group("fence")
        return f"{fence}\n[REDACTED:env block]\n{fence}"

    return _FENCE.sub(repl, text)


def _scrub_connection_strings(text: str) -> str:
    return _CONNECTION_STRING.sub(r"\1://[REDACTED:credentials]@", text)


def _scrub_key_shapes(text: str) -> str:
    for label, pattern in _KEY_SHAPES:
        text = pattern.sub(f"[REDACTED:{label}]", text)
    return text


def _scrub_pii(text: str) -> str:
    text = _EMAIL.sub("[REDACTED:email]", text)
    text = _PHONE.sub("[REDACTED:phone]", text)
    return text


def scrub_text(text: str) -> str:
    """Run every rule over `text`.

    Order matters: structural rules (PEM blocks, env fences) run before
    line-level ones so a block is replaced wholesale rather than picked at, and
    connection strings run before assignments so `DATABASE_URL=postgres://u:p@h`
    loses its password even though `DATABASE_URL` is not a secret-sounding name.
    """
    text = _scrub_key_shapes(text)  # PEM blocks first — they span lines.
    text = _scrub_env_fences(text)
    text = _scrub_connection_strings(text)
    text = _scrub_assignments(text)
    text = _scrub_pii(text)
    return text


# --------------------------------------------------------------------------
# The Scrubbed capability token
# --------------------------------------------------------------------------

_MINT = object()


@dataclass(frozen=True)
class Scrubbed:
    """Text that has been through `scrub()`.

    Constructed only by `scrub()`. `external.anthropic` requires this type, so
    no code path can send raw transcript text to the network — the ordering
    guarantee is structural rather than a comment asking people to be careful.
    """

    text: str
    _mint: object = None

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise RedactionError(
                "Scrubbed may only be constructed by redaction.scrub(); "
                "text reaching the network must be scrubbed first"
            )

    def __len__(self) -> int:
        return len(self.text)


def scrub(text: str) -> Scrubbed:
    """Pass 1. Scrub `text` before it can leave the machine.

    Raises `RedactionError` on any failure. Callers drop the segment rather
    than proceeding with text of unknown safety.
    """
    if not isinstance(text, str):
        raise RedactionError(f"expected str, got {type(text).__name__}")
    try:
        cleaned = scrub_text(text)
    except Exception as exc:  # noqa: BLE001 - fail closed on anything at all
        raise RedactionError(f"scrubber failed: {exc}") from exc
    return Scrubbed(cleaned, _MINT)


def scrub_record_field(value: str) -> str:
    """Pass 2. Scrub a model-produced field before it reaches the log.

    The model is told never to reproduce literals, but a record is written to
    a file a user may commit, so the same rules run again on the way out.
    """
    if not isinstance(value, str):
        raise RedactionError(f"expected str, got {type(value).__name__}")
    try:
        return scrub_text(value)
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise RedactionError(f"scrubber failed on record field: {exc}") from exc
