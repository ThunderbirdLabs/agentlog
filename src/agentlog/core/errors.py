"""Error types. Every failure path in agentlog raises one of these deliberately."""

from __future__ import annotations


class AgentlogError(Exception):
    """Base for every error agentlog raises."""


class ConfigError(AgentlogError):
    """Configuration is missing or invalid."""


class TranscriptError(AgentlogError):
    """A transcript could not be located or opened.

    Note: a *malformed line* inside a transcript is not an error — the parser
    skips it and continues. This is raised only when the file itself is
    unusable (unreadable, not a file).
    """


class RedactionError(AgentlogError):
    """The scrubber failed.

    Raised when redaction cannot complete. The pipeline treats this as a hard
    stop for the segment in question: an unscrubbed segment is never passed
    onward and never written. Losing a record is acceptable; writing a
    credential is not.
    """


class ExtractionError(AgentlogError):
    """The model returned something that could not be validated into records."""


class GitError(AgentlogError):
    """A git invocation failed in a way that leaves anchors unresolvable."""
