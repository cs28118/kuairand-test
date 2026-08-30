# RankLab Agent Guide

## Role

Your role is a participant in a hackathon. Your task is to assist me in implementing a ML research agent that perform a outstanding result. Implement one approved milestone at a time, keep each change reproducible, and report evidence rather than guesses.

The product being built is **RankLab**: a bounded ML research agent that
uses validation diagnostics and an internal, AIDE-inspired hypothesis planner
to propose experiments. AIDE is a design reference only; do not install,
clone, fork, or invoke it as a runtime dependency.

## Restrictions

- **Never modify `evaluate.py`.** It is the official scoring implementation.
- Never place credentials, API keys, raw test data, or unredacted validation
  labels in agent contexts, logs intended for sharing, or version control.

## Engineering Rules

- Work in small milestone-scoped changes. Do not edit unrelated starter-kit
  code or delete user files.
- Run tests and report their outcome before declaring a milestone complete.
- Treat planner or LLM output as untrusted: validate it before any code change
  and never allow it to execute code directly in the main workspace.
- Keep the planner dependency-free in Milestone 3. AIDE's solution-tree search
  is inspiration for its policy, not a library to install.

## Important Locations

| Purpose | Location |
| --- | --- |
| Project root | `C:\Users\user\Downloads\techjam\kuairand-starter-kit` |
| RankLab source | `C:\Users\user\Downloads\techjam\kuairand-starter-kit\ranklab` |
| Generated artifacts and ledgers | `C:\Users\user\Downloads\techjam\kuairand-starter-kit\artifacts` |
| Tests | `C:\Users\user\Downloads\techjam\kuairand-starter-kit\tests` |
| Expected base Python 3.12 interpreter | `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe` |

## Python Environment Notes

- The starter-kit target is Python 3.12.1.
- `python` may not be available on `PATH`; invoke the explicit Python 3.12
  executable above, or a verified virtual-environment executable.
- Do not create an AIDE-specific environment for Milestone 3. The internal
  planner must run in the project environment and use only the standard
  library plus dependencies already required by RankLab.

## Milestone Status

- Milestone 1: safety contract, baseline wrapper, and ledger — complete.
- Milestone 2: validation diagnostics and opportunity ranking — complete.
- Milestone 3: AIDE-inspired, dependency-free hypothesis planner — complete.
- Milestone 4: isolated experiment runner for approved proposals.
- Milestone 5: ranking-loss challenger.
- Milestone 6: causal-feature challenger.
- Milestone 7: sequence, multi-task, and ensemble challengers.
- Milestone 8: bounded autonomous run, convergence, and final submission.
