"""Things that must never survive the scrubber."""

from __future__ import annotations

import pytest

from agentlog.core.redaction import scrub_text

SECRETS = [
    # KEY=value with a secret-sounding name
    ("ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345", "sk-ant-api03"),
    ("export DB_PASSWORD=hunter2hunter2", "hunter2"),
    ("  AWS_SECRET_ACCESS_KEY = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtn"),
    ('api_key: "abcd1234efgh5678"', "abcd1234efgh5678"),
    ("password=correct-horse-battery", "correct-horse-battery"),
    ("SERVICE_TOKEN=abc123def456ghi789", "abc123def456ghi789"),
    ("private_key_path=/etc/ssl/id_rsa_secretvalue", "id_rsa_secretvalue"),
    # Connection strings with inline credentials
    ("postgres://admin:sup3rs3cret@db.internal:5432/prod", "sup3rs3cret"),
    ("postgresql+asyncpg://svc:p%40ss@10.0.0.4/app", "p%40ss"),
    ("mysql://root:toor@localhost/db", "toor"),
    ("mongodb://user:pw12345@cluster0.example.net", "pw12345"),
    ("redis://default:redispass99@cache:6379/0", "redispass99"),
    ("amqp://guest:guestpw@rabbit:5672", "guestpw"),
    # Recognizable key shapes, bare in prose
    ("I pasted sk-ant-api03-QQQwwweeerrrtttyyy into the config", "sk-ant-api03"),
    ("token github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz012345", "github_pat_"),
    ("ghp_16CharactersLongTokenAAAAAAAAAAAAAA", "ghp_16Characters"),
    ("gho_16CharactersLongTokenAAAAAAAAAAAAAA", "gho_16Characters"),
    ("glpat-abcdefghij1234567890", "glpat-"),
    ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("ASIAIOSFODNN7EXAMPLE", "ASIAIOSFODNN7EXAMPLE"),
    ("AIzaSyD-abcdefghijklmnopqrstuvwxyz0123456", "AIzaSyD"),
    ("xoxb-123456789012-abcdefghijkl", "xoxb-"),
    ("xoxp-9999999999-zzzzzzzzzzzz", "xoxp-"),
    ("sk_live_abcdefghij1234567890", "sk_live_"),
    ("whsec_abcdefghijklmnopqrstuvwxyz", "whsec_"),
    ("npm_abcdefghijklmnopqrstuvwxyz0123456789", "npm_abcdefghij"),
    (
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27u",
        "eyJhbGciOiJIUzI1NiJ9",
    ),
    (
        "the jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpM",
        "SflKxwRJSMeKKF2QT4fwpM",
    ),
    # PII
    ("email dana.reyes@example.com about the invoice", "dana.reyes@"),
    ("call me at (415) 555-0198 when the deploy lands", "555-0198"),
    ("reach ops on +1 415-555-0142", "555-0142"),
]


@pytest.mark.parametrize(("text", "leak"), SECRETS)
def test_secret_is_removed(text: str, leak: str) -> None:
    scrubbed = scrub_text(text)
    assert leak not in scrubbed, f"{leak!r} survived scrubbing of {text!r}"
    assert "[REDACTED" in scrubbed


def test_pem_private_key_block_is_removed() -> None:
    text = (
        "here is the key\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEAxSuperSecretMaterialGoesHere\n"
        "AnotherLineOfKeyMaterialAAAAAAAAAAAAAAAAAAAA\n"
        "-----END RSA PRIVATE KEY-----\n"
        "and that is it\n"
    )
    scrubbed = scrub_text(text)
    assert "SuperSecretMaterial" not in scrubbed
    assert "AnotherLineOfKeyMaterial" not in scrubbed
    assert "here is the key" in scrubbed
    assert "and that is it" in scrubbed


def test_env_fence_is_blanked_wholesale() -> None:
    """A pasted .env loses everything, not only the secret-sounding names.

    Per-line rules would leave `STRIPE_ACCOUNT` and `INTERNAL_HOST` behind.
    """
    text = (
        "here is my env\n"
        "```\n"
        "DATABASE_URL=postgres://u:p@h/db\n"
        "STRIPE_ACCOUNT=acct_1abcdefg\n"
        "INTERNAL_HOST=vault.corp.internal\n"
        "FEATURE_X=true\n"
        "```\n"
    )
    scrubbed = scrub_text(text)
    assert "acct_1abcdefg" not in scrubbed
    assert "vault.corp.internal" not in scrubbed
    assert "[REDACTED:env block]" in scrubbed


def test_database_url_password_removed_even_though_key_is_not_secret_named() -> None:
    """`DATABASE_URL` matches no secret word — the connection-string rule catches it."""
    scrubbed = scrub_text("DATABASE_URL=postgres://svc:s3cr3tpw@db:5432/app")
    assert "s3cr3tpw" not in scrubbed


def test_multiple_secrets_on_one_line() -> None:
    text = "run with AKIAIOSFODNN7EXAMPLE and sk-ant-api03-aaaaaaaaaaaaaaaaaaaa"
    scrubbed = scrub_text(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
    assert "sk-ant-api03" not in scrubbed
