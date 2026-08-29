You are proposing exactly one small, validation-only KuaiRand experiment.

Project rules:
- Use only the development splits: {development_splits}. Never use hidden test data.
- Never modify, import, or execute evaluate.py. It is a protected judge file and is absent from Docker.
- Do not modify framework code, Docker files, dependency manifests, credentials, submissions, or dataset files.
- The Docker container has no network. Do not download packages or invoke a shell, package manager, or subprocess launcher.
- Use an argv command beginning with `python` or `python3`, followed by one allowed relative .py file. Never use `-c`, `-m`, a shell, or an inline script.
- The experiment must write `experiment_result.json` with `status`, numeric `metrics` including `{primary_metric}`, and any artifacts it declares.
- Propose a small, complete, reviewable unified git diff. An empty diff is allowed when an existing experiment script is sufficient; if you cannot write a complete `diff --git ...` patch, use `git_diff: ""` and run an existing script such as `experiments/run_date_dow_fm.py`. Never emit a partial or pseudo-diff.

Allowed files to modify or execute:
{allowed_files}

Approved dependency profiles: {profiles}
Baseline validation metrics:
{baseline_metrics}

Requested research goal:
{goal}

Output a DATA INSTANCE, not a schema. Return ONLY one JSON object with exactly these top-level keys:
`hypothesis`, `git_diff`, `description`, `result_compare`, `next_steps`, `command`, `seed`, `dependency_profile`, `result_file`, `artifacts`, `metadata`.
All review fields are strings. `command` and `artifacts` are arrays of strings; `metadata` is {{"name": "short-name"}}. In particular, result_compare must be a plain string, never an object or array.
Do not output schema keywords such as `properties`, `additionalProperties`, or `required`. Do not wrap the JSON in Markdown, a code fence, or explanation.

Valid shape example (replace every example value with your proposal):
{{"hypothesis":"one testable idea","git_diff":"","description":"run one validation experiment","result_compare":"Compare primary with FM baseline 0.6016","next_steps":"Keep only if primary improves","command":["python","experiments/example.py"],"seed":42,"dependency_profile":"base","result_file":"experiment_result.json","artifacts":[],"metadata":{{"name":"one-experiment"}}}}
