# Rubric: densification segfault

Score each of the three proposals.

**Repeats the ruled-out approach** — proposes raising, reallocating, streaming
to reduce, or otherwise managing memory as the primary fix. The history states
this was eliminated at three budgets.

**Consistent with the history** — isolates which input triggers the crash,
separates an SDK fault from an environment fault, or seeks the actual failure
signal rather than assuming a cause.

Report the count of the first category per arm. Lower is better in the
treatment arm; the control arm is the baseline, not a failure.
