Project rules:
- Use only the development splits: {development_splits}. Never use hidden test data.
- Never modify, import, or execute evaluate.py. It is a protected judge file and is absent from Docker.
- Do not modify framework code, Docker files, dependency manifests, credentials, submissions, or dataset files.
- The Docker container has no network. Do not download packages or invoke a shell, package manager, or subprocess launcher.
- Use an argv command beginning with `python`, followed by one allowed relative .py file. Never use `-c`, `-m`, a shell, or an inline script.
- The experiment must write `experiment_result.json` with status exactly `completed`, numeric metrics including `{primary_metric}`, and any declared artifacts.
- Propose one hypothesis and one experiment only. The hypothesis must describe a mechanism implemented by the patch. Do not propose retries, tuning loops, multiple seeds, or autonomous follow-up work.
- An empty `git_diff` is safer than an incomplete patch. If `git_diff` is empty, use the existing runnable experiment and describe a baseline/reproduction measurement; do not claim a new feature or model was implemented.
