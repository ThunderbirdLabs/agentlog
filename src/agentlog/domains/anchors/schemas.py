"""Anchor types.

Anchors are computed, never generated. The model never produces a key: a model
that invents keys calls one feature three names across three sessions and the
timeline fragments.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Anchors(BaseModel):
    model_config = ConfigDict(frozen=True)

    files: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    # Configuration keys the change turned. For work against a library or an
    # SDK this is the anchor that matters — the knob, not the file that set it.
    settings: tuple[str, ...] = ()
    branch: str | None = None
    # The branch-to-issue link is the feature key. `feat/THU-142-export` gives a
    # stable feature identity for free — no clustering, no labelling, no drift.
    issue: str | None = None
    commits: tuple[str, ...] = ()
    head_sha: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.files or self.routes or self.symbols or self.settings or self.issue)
