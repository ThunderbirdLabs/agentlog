"""The only file in agentlog that imports the Anthropic SDK.

Two rules hold this boundary:

1. Nothing else imports `anthropic`. Swapping SDK versions, or the model
   provider, touches this file and no other.
2. `classify` accepts `Scrubbed` and refuses anything else. That is what makes
   "redaction runs before the network call" a property of the code rather than
   a convention — see `tests/test_ordering.py`.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

from agentlog.core.errors import ConfigError
from agentlog.core.logging import get_logger
from agentlog.core.redaction import Scrubbed

log = get_logger("external.anthropic")

T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = 2048


class ModelClient(Protocol):
    """What extraction needs from a model. Tests substitute a fake."""

    def classify(self, system: str, user: Scrubbed, schema: type[T]) -> T | None: ...


def _require_scrubbed(user: object) -> Scrubbed:
    if not isinstance(user, Scrubbed):
        raise TypeError(
            "classify() requires redaction.Scrubbed; transcript text must be "
            f"scrubbed before any network call (got {type(user).__name__})"
        )
    return user


class AnthropicClient:
    """Thin wrapper over the Messages API.

    Structured outputs are used rather than asking for JSON in prose: the
    schema is enforced server-side, so a shape error becomes impossible instead
    of becoming a parsing bug.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        # Imported here so `import agentlog` works without the SDK installed —
        # every command except `capture --stage extract` is fully offline.
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                "the `anthropic` package is required for extraction; install agentlog's dependencies"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Secrets live in the environment; "
                "agentlog never reads or writes them to disk."
            )
        self._model = model
        self._client = anthropic.Anthropic(api_key=key)

    def classify(self, system: str, user: Scrubbed, schema: type[T]) -> T | None:
        scrubbed = _require_scrubbed(user)
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": scrubbed.text}],
            output_format=schema,
        )
        parsed = response.parsed_output
        if parsed is None:
            log.warning("model returned no parseable output (stop_reason=%s)", response.stop_reason)
            return None
        return parsed
