# agentlog

Reads AI coding agent transcripts, extracts what was tried and what failed, keys those records to files and routes, and feeds them back to a future agent working on the same code.

Git records what changed. The chat records why, and every dead end along the way. Nobody joins them, so an agent opening a repo three weeks later re-tries the thing that already failed.

## The failure this fixes

You spend an hour on an export endpoint. You try streaming the response, it times out. You bump the worker limit, still times out. You switch to a background job, it works, you commit `fix: export timeout`.

Git has one commit. Both failures are gone, because broken versions don't get committed. Three weeks later an agent opens that file and tries streaming.

The transcript had all three attempts. This tool reads it.

## Status

Working end to end: capture, anchors, redaction, store, retrieval, hooks, and
install. Verified against real sessions — 38 records extracted from a month of
work on a photogrammetry worker, including the failed memory-limit experiments
and the calibration setting that had been dropped between two versions.

Not done: per-turn injection (`UserPromptSubmit`), staleness tagging, and a
reader for anything other than Claude Code. Extraction quality is the weakest
part — it currently produces more decisions and notes than dead ends, which is
backwards from the intent, and it still names files the anchors already cover.

## Quickstart

Two ways in. Both end in the same place.

### Type three commands

```bash
uv tool install git+https://github.com/OWNER/agentlog     # or: pipx install git+...
cd your-repo
agentlog init                                             # --with ../sibling-repo if work spans two
agentlog backfill --days 30                               # dry run: what's there, what it would cost
```

Then start a **new** Claude Code session — hooks are read at session start, so
the one you are in now will not have them.

### Or hand this to your agent

```
Set up agentlog in this repository.

1. Install the CLI with `uv tool install git+https://github.com/OWNER/agentlog`.
   If uv is not available, use pipx. Do not pip install into the project venv.
2. Run `agentlog init` from the repo root. If this project's work regularly
   spans a sibling repo — a frontend, a worker, an infra repo — pass
   `--with <path-to-sibling>`, then run init in that repo too with `--with`
   pointing back here.
3. Run `agentlog backfill --days 30`. This is a dry run: it reports how many
   past sessions it found and what extracting them would cost.
4. Tell me what it found and stop there.

Do not drain the queue or pass --write without asking me first — that step
spends money or plan tokens.
```

The last line matters. Everything up to it is free and local; the extraction
step is the only part that costs anything, and an agent should not decide that
on your behalf.

### After setup

Capture runs on its own from then on. To turn queued work into records:

```bash
agentlog drain            # needs ANTHROPIC_API_KEY, ~half a cent per work unit
```

or run `/agentlog-drain` inside a Claude Code session, which does the same
extraction on your existing plan with no key at all.

Then:

```bash
agentlog file src/worker/pipeline.py    # what happened to this file, oldest first
agentlog setting image_scale            # every session that turned this knob
agentlog search "densification crash"
```

## How it works

**Anchors are computed, never generated.** Files, routes, symbols, branch, issue, and commits come from git and from parsing source. The model never produces a key — a model that invents keys calls one feature three names across three sessions and the timeline fragments.

**Redaction runs before the network, and the code enforces it.** `redaction.scrub()` is the only way to obtain a `Scrubbed` value, and `external/anthropic.py` accepts nothing else. Swapping the order does not compile past the tests. Scrubbing failures drop the segment rather than proceeding: a lost record is a minor loss, a written credential is not.

**One model call per segment, Haiku tier.** This is classification, not reasoning. The response is validated against a pydantic schema before anything is written; a malformed response retries once, then the segment is skipped with a warning.

**Agent reasoning is not available, and the design does not assume it is.** Transcripts on current models store `thinking` blocks whose text is empty — `display` defaults to `"omitted"`, so the reasoning never reaches disk. Verified against a real 172-turn session: 51 thinking blocks, all empty. The signals extraction actually runs on are assistant text, tool calls, and tool errors. That is enough — a failed test and an abandoned approach are both observable — but anything built on top of this should not expect the agent's reasoning to be there.

**No embeddings, no vector store.** Deliberate. Missing a record leaves you exactly where you are today, but returning a near-match tells the agent "Redis caching deadlocked here" about a different endpoint, and now it avoids a valid approach for a fake reason. Keyword and exact anchor matching — poor recall, perfect precision — is the correct trade.

## Configuration

Defaults are in `core/config.py`; a repo overrides them in `.agentlog/config.json`. The Anthropic key is read from `ANTHROPIC_API_KEY` at the point of use and is never written to disk.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

The suite sets `pythonpath = ["src"]`, so it runs against the source tree and does not depend on the editable install being wired up. If the `agentlog` console script cannot find the package, `PYTHONPATH=src python -m agentlog.cli ...` always works.

Tests never hit the network: extraction is always given a fake client, and anchor resolution is offline against a fixture git repo. Fixture transcripts are hand-written and contain no real secrets.

## Prior art

This is a crowded space. Several tools already do most of what agentlog does, and two do large parts of it better-established. Read this section before deciding agentlog is worth using.

**[projectmem](https://github.com/riponcm/projectmem)** ([paper](https://arxiv.org/abs/2606.12329), Malo & Qiu) is the closest in intent: local-first, append-only plain text, MCP-native, and explicitly built to warn an agent before it repeats an approach that already failed. It is further along than this and has a published evaluation. The difference is where records come from — projectmem is logged deliberately, by the agent calling `record_attempt()` or a human running `pjm attempt`. agentlog reads the transcript instead, which trades projectmem's precision for coverage of the attempts nobody thought to log.

**[claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler)** already does transcript-first capture on the same hooks, at the same moment — session end and pre-compaction — and extracts with the Claude Agent SDK, which means it runs on a Claude subscription with no API key. If you want this category of tool today, start there. It organises what it finds into LLM-written concept articles rather than keying it to anything computed.

**[claude-memory-extractor](https://github.com/obra/claude-memory-extractor)** also reads `~/.claude/projects` JSONL directly, chunks long sessions, and extracts lessons with model-generated tags. Also in the space: [claude_memory](https://github.com/codenamev/claude_memory) (hooks + MCP + SQLite), [claude-remember](https://github.com/Digital-Process-Tools/claude-remember), and [code-session-memory](https://github.com/djannot/code-session-memory).

So transcript-first capture is not novel, and neither is compaction-boundary timing. What is actually different here, as far as their public documentation shows:

- **The model never produces a key.** Anchors — files, routes, symbols, config settings, branch, issue — are computed from git and from parsing source. Every other tool above either has the model generate tags and organisation, or keys on file paths and full text. A model that invents keys calls one feature three names across three sessions and the timeline fragments.
- **Configuration keys are anchors, and both sides of a rename are kept.** When work is against a library or an SDK the knob is the unit of work, and the name you search six weeks later is the one that was in use when it last worked — which only exists on the removed side of a diff. Nothing else found does this.
- **Secrets are scrubbed before the network, enforced by the type system.** None of the tools above mention redaction at all, and all of them ship transcript text to a model. Transcripts contain pasted `.env` files.

Everything else — local-first, append-only plain text, no telemetry, hook-driven capture, running on a subscription — is convergent, and was arrived at independently rather than first.

Also relevant: **Codified Context** ([arXiv:2602.20478](https://arxiv.org/abs/2602.20478)), **ESAA** ([arXiv:2602.23193](https://arxiv.org/abs/2602.23193)) on event sourcing for SE agents, and **Reflexion** ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) on verbal feedback from failed trials within a task.

## License

MIT.
