# Participant Framework — Milestone 2

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
- The framework contains no LLM client or autonomous coding loop. The pilot accepts a human-written instruction and executes exactly one experiment.

## Supervised pilot

Create an `ExperimentSpec` JSON instruction with these required review fields:

```json
{
  "hypothesis": "A pairwise objective will improve ranking quality.",
  "git_diff": "",
  "description": "Train one candidate model on train and score valid.",
  "result_compare": "Compare primary against the FM validation baseline.",
  "next_steps": "Keep the change only if primary improves; otherwise try the next hypothesis.",
  "command": ["python", "experiment.py"],
  "seed": 42,
  "dependency_profile": "base",
  "result_file": "experiment_result.json",
  "artifacts": ["model.bin"]
}
```

The command runs in Docker with no network, bounded memory/CPU/processes, and a timeout. It must write `experiment_result.json` inside the workspace, including numeric `metrics` (including `primary`) and a `status`. Protected judge files are not mounted into the container. The framework persists stdout, stderr, the instruction, git diff, modified files, result artifacts, failures, and recovery attempts under `runs/`.

Run it with:

```bash
python -m framework.pilot --spec experiment.json --baseline-primary 0.6016
```

Token and cost fields are supported in the result contract but intentionally have no configured prices or limits yet. Approved dependency profiles are declarations only; the framework never installs arbitrary packages.

## Next milestones

1. Connect the LLM with structured prompts, safe tool calls, and research/implementation/reviewer roles.
2. Enable bounded autonomous iterations only after the supervised pilot is validated.
