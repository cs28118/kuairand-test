Repository facts (use these facts; do not invent APIs or files):

Allowed files to modify or execute:
{allowed_files}

Existing runnable experiment:
- `experiments/run_date_dow_fm.py` imports `data.load`, `data.encode`, `baseline.FM`, and the development-only `safe_evaluate` implementation.
- It reads `EXPERIMENT_SEED`, uses only train/valid, and writes the required `experiment_result.json` through `EXPERIMENT_RESULT_PATH`.
- It does not currently write `model.npz`; set `artifacts` to `[]` unless the complete patch explicitly adds checkpoint creation.
- Prefer this script when no complete patch is needed. It is the only command path guaranteed to exist in the pilot image.
- Despite its filename, the current script is still the baseline FM pipeline: `data.py` currently encodes only `user_id`, `video_id`, `author_id`, `tab`, and `dur_bucket`. It does not currently implement a day-of-week feature.

Useful existing APIs:
- `data.load(data_dir)` returns `train`, `valid`, and `test`; use only `train` and `valid`.
- `data.encode(splits)` returns encoded arrays and the feature dimension.
- `baseline.FM` provides `step()` and `predict()` for the existing NumPy FM.
- `experiments/safe_evaluate.py` is the development-only evaluator available inside the pilot.

Exact current `data.py` anchors (patch against these lines, not a generic
dataset implementation):
```python
LABEL = 'long_view'
SPLITS = {{'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}}
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
# each loaded row is (date, user_id, video_id, author_id, tab, duration_ms, label)
def encode(splits):
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])
    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]
```
The complete file uses these five categorical fields, vocabulary offsets, and
returns `(encoded_splits, dimension)`. It has no timestamps, dictionaries, or
`train.txt`/`valid.txt` format. Do not invent any of those structures.

Exact current `experiments/run_date_dow_fm.py` change anchor for the requested
FM capacity experiment:
```python
    encoded, dimension = encode(development)
    x_train, y_train, _ = encoded["train"]
    x_valid, y_valid, users_valid = encoded["valid"]
    model = FM(dimension, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
```
If changing FM capacity, emit this exact minimal diff (including hunk header
and six-line counts), changing only the `k=16` token to `k=8`:
```diff
diff --git a/experiments/run_date_dow_fm.py b/experiments/run_date_dow_fm.py
--- a/experiments/run_date_dow_fm.py
+++ b/experiments/run_date_dow_fm.py
@@ -24,5 +24,5 @@
     encoded, dimension = encode(development)
     x_train, y_train, _ = encoded["train"]
     x_valid, y_valid, users_valid = encoded["valid"]
-    model = FM(dimension, k=16, lr=0.001, seed=seed)
+    model = FM(dimension, k=8, lr=0.001, seed=seed)
     rng = np.random.default_rng(seed)
```
Preserve these lines exactly; do not add a function signature, imports, or
training-loop changes.

Approved dependency profiles: {profiles}
Do not reference `DataPipeline`, `BaseModel`, LightGBM, Torch, pandas, or other APIs unless the proposal's complete patch also adds and verifies them.

Propose one real, testable ranking improvement for KuaiRand validation. Use a
complete unified git diff on an allowed file only if you can produce it exactly;
otherwise use an empty `git_diff` and the existing experiment script. Do not
invent a command path, and evaluate only the train and valid development splits.

Never claim that a feature, model, or enhancement exists unless the supplied
`git_diff` actually implements it. An empty diff is a reproduction only.
