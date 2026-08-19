# BACE / Exact Batch-ERV debugging snapshot

This repository is a source-only snapshot prepared to reproduce and diagnose the
BACE ALFWorld training issue described in
[`当前问题说明-Exact-Batch-ERV指数爆炸.md`](./当前问题说明-Exact-Batch-ERV指数爆炸.md).

## Current blocker

During a 4×H100 stability run, training completed steps 1 and 2 but stalled in
step 3 before branch replay. The current Exact Batch-ERV implementation builds
the Cartesian product of all per-anchor allocations and only then filters by the
global branch quota. In the captured case, a task with quota 1 and 26 usable
anchors caused approximately 1.69 trillion candidate states to be enumerated,
although only 26 single-branch allocations were feasible.

The relevant implementation is:

- `verl-agent-src/recipe/bace_gigpo/batch_erv.py`
- `verl-agent-src/recipe/bace_gigpo/coordinator.py`
- `verl-agent-src/recipe/bace_gigpo/rollout_collector.py`

Relevant focused tests are under `verl-agent-src/tests/bace_gigpo/`.

## Repository layout

- `verl-agent-src/`: migrated verl-agent source plus the BACE implementation,
  launchers, and tests.
- `BACE-work-2/`: current method and parameter specifications.
- `BACE-work/`: earlier design, implementation, and audit notes.
- `deploy/`: H100 environment and smoke-test helpers.
- `issues/`: related algorithm investigations.

The original upstream project is
[`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent). This migrated
snapshot has no original `.git` metadata, so it is published as a diagnostic
snapshot rather than a clean upstream patch series.

## What is intentionally excluded

Experiment artifacts, checkpoints, Slurm logs, core dumps, Python caches,
models, datasets, local environments, and the `ICLR2027/` paper workspace are
excluded. ALFWorld data and Qwen model weights must be supplied separately.

The H100 launch scripts contain site-specific absolute paths and Slurm account
settings from the original RWTH deployment. Review and replace these values
before running on another cluster; they are configuration examples, not portable
defaults.

## Focused regression tests

From `verl-agent-src/`, with the project dependencies available:

```bash
pytest -q tests/bace_gigpo
```

The issue document contains the observed runtime evidence, failure location, and
a proposed quota-aware dynamic-programming replacement that preserves Exact
Batch-ERV tie semantics.
