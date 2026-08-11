"""Ordinary prose and code must survive untouched.

Over-redaction is the failure mode that makes the log useless: a summary
reading "tried [REDACTED] before [REDACTED]" costs a model call and tells a
future agent nothing. These cases must come through byte-identical.
"""

from __future__ import annotations

import pytest

from agentlog.core.redaction import scrub_text

CLEAN = [
    # Prose that mentions secrets without containing any
    "The API key is stored in the vault, not in the repo.",
    "We rotate the token every 90 days.",
    "The password reset flow was broken for SSO users.",
    "Token: 42 tokens used across the run.",
    "Streaming the export timed out, so we moved to a background job.",
    "The private method on the service was doing too much.",
    "Check the secret manager before adding a new credential.",
    # Versions, dates, ids, hashes — the classic false-positive family
    "Bumped fastapi from 0.109.2 to 0.115.0",
    "Released on 2026-08-11 after the 1.2.3 tag",
    "commit a1b2c3d4e5f6789012345678901234567890abcd looks fine",
    "run id 12345678901234567890 finished in 4.5s",
    "Exit code 1 on line 220 of 512",
    "The container listens on 127.0.0.1:8080",
    # Code that isn't a secret
    "def get_weather(location: str, unit: str = 'celsius') -> str:",
    "const apiClient = createClient({ retries: 3 })",
    "if response.status_code == 429: backoff()",
    "@app.get('/api/v1/health')",
    "import { useState } from 'react'",
    "npm install --save-dev @types/node",
    "SELECT id, name FROM users WHERE active = true",
    "assert result == {'ok': True}",
    # Paths and identifiers
    "src/features/inspections/service.py:142",
    "See docs/AGENT_PROMPT_GUIDE.md for the format",
    "The branch is feat/THU-142-inspection-export",
    "POST /api/v1/inspections/{id}/export",
    # Env-shaped lines whose names are not secrets
    "LOG_LEVEL=debug",
    "PORT=8080",
    "NODE_ENV=production",
    "FEATURE_EXPORT_V2=true",
]


@pytest.mark.parametrize("text", CLEAN)
def test_clean_text_is_untouched(text: str) -> None:
    assert scrub_text(text) == text


def test_a_normal_paragraph_is_untouched() -> None:
    text = (
        "We spent an hour on the export endpoint. Streaming the response timed out, "
        "so we bumped the worker limit, and it still timed out. Moving it to a "
        "background job worked. The tests pass now.\n\n"
        "The interesting part is that neither failure is in git — only the fix got "
        "committed, so the next person to open this file has no idea streaming was "
        "already tried."
    )
    assert scrub_text(text) == text


def test_code_fence_that_is_not_env_survives() -> None:
    text = "```python\ndef render(self):\n    return self.template.render()\n```"
    assert scrub_text(text) == text


def test_short_fence_with_one_assignment_survives() -> None:
    """One assignment is a code sample, not a pasted .env."""
    text = "```\nPORT=8080\n```"
    assert scrub_text(text) == text


def test_url_without_credentials_is_untouched() -> None:
    assert scrub_text("https://api.example.com/v1/things") == "https://api.example.com/v1/things"
    assert scrub_text("postgres://db.internal:5432/app") == "postgres://db.internal:5432/app"
