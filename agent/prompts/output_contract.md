Return ONLY one JSON object conforming to the ExperimentSpec contract. Output a
DATA INSTANCE, not a schema. No
Markdown, code fence, prose, or explanation.

The object must contain exactly these top-level keys:
`hypothesis`, `git_diff`, `description`, `result_compare`, `next_steps`,
`command`, `seed`, `dependency_profile`, `result_file`, `artifacts`, `metadata`.

JSON rules:
- Use strict JSON: double quotes, no comments, no trailing commas, and no duplicate keys.
- Do not put literal line breaks inside a JSON string. Encode patch line breaks with the single JSON escape `\n` (one backslash followed by `n`); never emit `\\n`. Encode each patch line break this way.
- All review fields, including `git_diff`, `result_compare`, and `next_steps`, are strings. `command` and `artifacts` are arrays of strings. `metadata` is {{"name":"short-name"}}.
- `result_compare must be a plain string`; `next_steps` must also be a plain
  string. Neither field may be an object or array.

Patch and command rules:
- `git_diff` must be either `""` or a complete patch beginning with `diff --git a/<path> b/<path>`, followed by valid `---`, `+++`, and hunk lines. A complete patch has both file headers and at least one hunk; `+def`, `-line`, or `@@` by itself is not a patch.
- A code snippet beginning with `+def`, `-line`, or `@@` alone is invalid. Never emit a partial or pseudo-diff.
- If you cannot produce a complete patch exactly, use `git_diff: ""` and `command: ["python", "experiments/run_date_dow_fm.py"]`. Do not invent a command path.
- If the requested research goal explicitly asks for a code or feature change, `git_diff` must be non-empty and must implement that requested change. Use the empty-diff fallback only when the goal is a reproduction or measurement; never describe an unimplemented change as if it exists.
- result_file must be exactly `experiment_result.json`.

Metric names in comparisons must be the repository names `GAUC`, `nDCG@5`, and
`primary`; do not invent names such as `GAUC@5`.

Final self-check before emitting the object:
- The first non-whitespace character is `{{` and the last is `}}`.
- Every string is on one logical JSON line; all embedded newlines use the single
  JSON escape `\n`, and all embedded quotes are escaped.
- The object has exactly the eleven keys listed above, with no schema keywords
  such as `properties`, `additionalProperties`, or `required`.
- `result_compare` and `next_steps` are plain JSON strings, never objects or
  arrays. The command is an argv array, never a shell command string.

Valid shape example (replace every value):
{{"hypothesis":"one testable idea","git_diff":"","description":"run one validation experiment","result_compare":"Compare primary with FM baseline","next_steps":"Keep only if primary improves","command":["python","experiments/run_date_dow_fm.py"],"seed":42,"dependency_profile":"base","result_file":"experiment_result.json","artifacts":[],"metadata":{{"name":"one-experiment"}}}}
