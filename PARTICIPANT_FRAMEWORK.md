# Participant Framework — Milestone 1

This document describes the participant-side preparation for an autonomous ML research agent. It is separate from the hackathon starter-kit README so the official starter instructions remain unchanged.

## Purpose

`framework/` is an execution foundation for the research agent, not a recommender model itself. The first milestone makes experimentation reproducible and protects the official evaluation contract before any LLM-driven work begins.

It provides:

- validation-only random, popularity, and FM baseline runners; it neither evaluates nor records test metrics;
- an SHA-256 fingerprint check for the protected `evaluate.py` file;
- append-only experiment logs, validation-best FM checkpoints, and run state;
- read-only environment, NumPy, and dataset preflight checks.

## Setup

Use Python 3.12 or newer, then install the minimal dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the environment check and reproduce validation baselines:

```bash
python -m framework.doctor
python -m framework.preflight --run-baselines
```

Run one baseline directly:

```bash
python -m framework.runner --experiment fm
```

Each run writes state, append-only iteration records, metrics, and (for FM) a validation-best `model.npz` checkpoint to `runs/<run-id>/`. The `runs/` directory is intentionally Git-ignored.

## Current boundaries

- Do not edit `evaluate.py`; the framework detects changes and stops.
- Development work is limited to the train/validation split. Test evaluation and submissions remain explicit participant actions.
- The framework currently contains no LLM client, autonomous coding loop, or package-install mechanism.

## Next milestones

1. Add experiment specifications/results, token accounting, and convergence criteria.
2. Add bounded execution: worktree isolation, dependency profiles, timeouts, and recovery records.
3. Add a provider-neutral LLM adapter, compact role prompts, and allow-listed research/code/evaluation tools.
4. Run a supervised single-experiment pilot before enabling bounded autonomous iterations.
