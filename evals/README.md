# Behavioural evals

`agentlog lint` measures the *shape* of records. It cannot tell you whether a
record changes what an agent does — which is the only claim this tool actually
makes. These evals test that directly.

## The method

Give two agents an identical task. One gets nothing; one gets records that bear
on it. Score both against a fixed rubric. If the answers do not differ, the
records are decoration no matter how well they read.

The task has to be one where the obvious approach is the one already ruled out.
That is the whole situation this tool exists for: an agent reaching confidently
for something the history says does not work.

## Running one

```
control:    give an agent evals/<name>-task.txt alone
treatment:  give it evals/<name>-context.txt as well
rubric:     evals/<name>-rubric.md
```

Run each arm 10-20 times and report a rate. A single pair is directional at
best — models vary run to run, and one hand-picked record proves nothing.

## Result on hand, n=1 per arm

`densification` — a segfault during densification on a 381-frame job. The
history says memory was eliminated at 128, 200 and 256 GB.

| arm | proposals that raise or manage memory |
|---|---|
| control | 3 of 3 |
| treatment | 0 of 3 |

The treatment arm's first suggestion was to bisect the frame set, which is
where the real investigation eventually landed.

Directional, not measured: one run each, one hand-picked record, one model, and
the treatment arm was told its context described prior work.

## Using this to tune the prompt

Records carry the extractor version that produced them (`haiku-4.5/v4`). Build a
context file from each version and run the same task against both. A prompt
change that does not move the rate is not an improvement, whatever it does to
the lint numbers.
