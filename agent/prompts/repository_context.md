Repository facts (use these facts; do not invent APIs or files):

Allowed files to modify or execute:
{allowed_files}

Existing runnable experiment:
- `experiments/run_date_dow_fm.py` imports `data.load`, `data.encode`, `baseline.FM`, and the development-only `safe_evaluate` implementation.
- It reads `EXPERIMENT_SEED`, uses only train/valid, and writes the required `experiment_result.json` through `EXPERIMENT_RESULT_PATH`.
- Prefer this script when no complete patch is needed. It is the only command path guaranteed to exist in the pilot image.

Useful existing APIs:
- `data.load(data_dir)` returns `train`, `valid`, and `test`; use only `train` and `valid`.
- `data.encode(splits)` returns encoded arrays and the feature dimension.
- `baseline.FM` provides `step()` and `predict()` for the existing NumPy FM.
- `experiments/safe_evaluate.py` is the development-only evaluator available inside the pilot.

Approved dependency profiles: {profiles}
Do not reference `DataPipeline`, `BaseModel`, LightGBM, Torch, pandas, or other APIs unless the proposal's complete patch also adds and verifies them.

Propose one real, testable ranking improvement for KuaiRand validation. Use a
complete unified git diff on an allowed file only if you can produce it exactly;
otherwise use an empty `git_diff` and the existing experiment script. Do not
invent a command path, and evaluate only the train and valid development splits.
