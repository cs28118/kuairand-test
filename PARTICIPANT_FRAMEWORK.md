# Participant Framework — Milestone 3

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
- An LLM can propose exactly one `ExperimentSpec`, but it cannot execute it. A human must separately approve the saved proposal.
- There is no autonomous iteration, retry, or research loop.

## LLM-supervised proposal

Set a provider credential and a pinned model in your terminal environment or a local, Git-ignored `.env` file. The framework launcher loads the project-local `.env` automatically, while process environment variables take precedence. Then request one proposal:

```bash
python -m framework.propose --goal "Test one small pairwise-ranking hypothesis on validation data."
```

For an organization-provided OpenAI-compatible gateway, set `OPENAI_BASE_URL` to its API root (usually ending in `/v1`). The wrapper then calls `<OPENAI_BASE_URL>/responses`; leave it unset to use the public OpenAI API.

The wrapper sends a structured prompt containing the project rules, validation baselines, allowed files, result contract, and strict `ExperimentSpec` JSON schema. It rejects non-JSON responses, unknown schema fields, non-Python commands, unsafe paths, unapproved dependency profiles, and any attempt to modify a protected or non-allowed file.

The command prints a run ID and writes `runs/<run-id>/proposal.json`. Inspect that file before approving it:

```bash
python -m framework.propose --approve-run run-1 --approval-note "Reviewed command and patch."
```

Approval is an explicit second command. Only this command passes the saved spec to `framework.pilot`; the Docker pilot keeps its network, resource, file, and evaluator protections. `audit.jsonl` records the complete LLM request, raw response, validation decision, human approval, and final result.

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

## Next milestone

Enable a bounded autonomous research loop only after two or three LLM-generated, human-approved experiments have completed successfully.
