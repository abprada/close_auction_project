# W&B Model Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `04) train_wandb.ipynb` — five W&B Bayesian sweeps (30 trials each, optimizing held-out IC) that tune the model shortlist from `03) models_and_feature_selection.ipynb`, ending in a single comparison table of each model's best config and full metric set.

**Architecture:** One notebook, eight sequential sections. Four sklearn models (`HuberRegressor`, `Lasso`, `GradientBoostingRegressor`, `RandomForestRegressor`) share one generic `train()`-function factory and a `Pipeline(impute → scale → model)`. The MLP is a separate small `torch.nn` feedforward network with its own per-fold training loop (early stopping, per-epoch logging on one fold). All five reuse the same 10-fold expanding-window CV folds, rebuilt once at the top of the notebook. A final cell pulls each sweep's best run via the W&B API into one table.

**Tech Stack:** pandas, scikit-learn, PyTorch, Weights & Biases (`wandb`), scipy (`spearmanr`), Jupyter/nbformat.

**Spec:** `docs/superpowers/specs/2026-08-19-wandb-model-tuning-design.md`

## Global Constraints

- Working directory for every command: `~/repos/close_auction_project`.
- Python/Jupyter binaries: `.venv/bin/python3` / `.venv/bin/jupyter` — this is a `uv`-managed venv with **no `pip` on PATH**. To add a new package: `uv pip install --python .venv/bin/python3 <package>`. `wandb` and `torch` are already installed and in `requirements.txt` — no action needed.
- **Notebooks are never edited with a text-editing tool.** Every change is a small Python script using `nbformat` that reads the `.ipynb`, appends/modifies cells, and writes it back — then the notebook is executed for real.
- Canonical execution command: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`. After every execution: (1) confirm the command's own output contains no `CellExecutionError` / traceback, (2) read the actual printed output of the new cells via a small `nbformat`-based read script (never just trust "it ran") before considering a step done.
- **Runtime grows task over task.** Because verification always re-executes the whole notebook top to bottom, later tasks re-run every earlier task's already-completed sweep too. Tasks 4 onward should be run with `run_in_background: true` and polled rather than blocked on. Before touching the real notebook, validate new `train()` logic cheaply in a throwaway script (direct call, or a 2-trial `wandb.agent(..., count=2)`) — catches bugs without paying the 30-trial cost.
- W&B auth is already configured: `wandb login` was run by the user in a separate terminal, credentials are cached in `~/.netrc`. Every `wandb.login()` call in this plan takes **no arguments** — never pass or read an API key.
- W&B project name (all five sweeps): `"close-auction-model-tuning"`.
- Every sweep: `"method": "bayes"`, `"metric": {"name": "eval_ic", "goal": "maximize"}`, `count=30` in the real (non-dry-run) `wandb.agent()` call.
- Data: `data/train.csv` (parse `date` as datetime), `data/columns.json`. Target column: `TARGET_COL = columns["target_cols"][0]` (`"target_bps_vs_mid"`). Features: `FINAL_FEATURE_COLS = columns["final_feature_cols"]` — exactly `["near_price_vs_mid_bps", "urgency", "ref_price_vs_mid_bps", "far_price_vs_mid_bps"]`.
- The loaded training DataFrame is named `train_df` in this notebook (**not** `train`) — `train` is reserved for the wandb sweep callback function name used throughout, and shadowing it with a DataFrame would break every sweep task.
- CV folds: identical construction to `03) models_and_feature_selection.ipynb`'s cell (`MIN_TRAIN_DAYS = 7`, `DELTA_DAYS = 1`, walk-forward on `train_df["date"]`) — 10 folds, indices 0–9, fold 9 is the largest (16 fit days). Reproduced fresh in this notebook, not imported from `03)`.

---

### Task 1: Notebook foundation — data, CV folds, wandb login, shared sklearn train-fn factory

**Files:**
- Modify: `04) train_wandb.ipynb` (currently just a title cell, `# 04) Train with W&B`)
- Test: none (no pytest in this project — notebooks are verified by execution + reading output, per Global Constraints)

**Interfaces:**
- Consumes: nothing (first real task)
- Produces (module-level names later tasks depend on, all live in the single notebook kernel):
  - `train_df: pd.DataFrame`, `TARGET_COL: str`, `FINAL_FEATURE_COLS: list[str]`
  - `folds: list[tuple[pd.Series, pd.Series]]` — 10 `(fit_mask, eval_mask)` boolean-mask pairs over `train_df`
  - `make_sklearn_train_fn(estimator_class) -> Callable[[], None]` — factory used by Tasks 2–5
  - Already-imported names available to every later task in-kernel: `json, np, pd, wandb, spearmanr, SimpleImputer, StandardScaler, Pipeline, mean_squared_error, r2_score`

- [ ] **Step 1: Write the nbformat script that builds this task's cells**

Write to `/tmp/build_04_task1.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## Setup

Loads `data/train.csv`/`data/columns.json` fresh (same convention as every other notebook here),
logs in to W&B (credentials already cached in `~/.netrc` -- no key ever passed here), and rebuilds
the exact 10-fold expanding-window CV split from `03)`: walk-forward, `MIN_TRAIN_DAYS=7` warm-up,
`DELTA_DAYS=1` step, no lookahead.

The loaded DataFrame is `train_df`, not `train` -- `train` is reserved for the per-model function
each W&B sweep calls.'''
))

cells.append(nbf.v4.new_code_cell(
'''import json

import numpy as np
import pandas as pd
import wandb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("data/train.csv", parse_dates=["date"])
with open("data/columns.json") as f:
    columns = json.load(f)

TARGET_COL = columns["target_cols"][0]
FINAL_FEATURE_COLS = columns["final_feature_cols"]

wandb.login()

print(f"train_df: {train_df.shape[0]} rows, target = {TARGET_COL!r}")
print(f"{len(FINAL_FEATURE_COLS)} final features:", FINAL_FEATURE_COLS)'''
))

cells.append(nbf.v4.new_code_cell(
'''unique_dates = np.sort(train_df["date"].unique())
MIN_TRAIN_DAYS = 7
DELTA_DAYS = 1

folds = []
for fold_idx, eval_start in enumerate(range(MIN_TRAIN_DAYS, len(unique_dates), DELTA_DAYS)):
    fit_dates = unique_dates[:eval_start]
    eval_dates = unique_dates[eval_start:eval_start + DELTA_DAYS]
    assert fit_dates.max() < eval_dates.min(), "eval dates must all fall after fit dates"

    fit_mask = train_df["date"].isin(fit_dates)
    eval_mask = train_df["date"].isin(eval_dates)
    folds.append((fit_mask, eval_mask))

    print(f"fold {fold_idx}: fit {fit_mask.sum():>4} rows ({len(fit_dates)} days, "
          f"{pd.Timestamp(fit_dates.min()).date()} to {pd.Timestamp(fit_dates.max()).date()}) | "
          f"eval {eval_mask.sum():>4} rows ({len(eval_dates)} days, "
          f"{pd.Timestamp(eval_dates.min()).date()} to {pd.Timestamp(eval_dates.max()).date()})")'''
))

cells.append(nbf.v4.new_markdown_cell(
'''## Shared sklearn sweep runner

One `train()`-function factory reused by every sklearn model's sweep (`HuberRegressor`, `Lasso`,
`GradientBoostingRegressor`, `RandomForestRegressor`): builds `impute -> scale -> model`, fits/evals
across all 10 folds, logs the full metric set. `random_state` is fixed to `0` automatically for
models that accept it (not swept -- a determinism knob, not a tunable choice).'''
))

cells.append(nbf.v4.new_code_cell(
'''def make_sklearn_train_fn(estimator_class):
    def train():
        wandb.init()
        config = dict(wandb.config)
        if "random_state" in estimator_class().get_params():
            config["random_state"] = 0
        estimator = estimator_class(**config)

        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])

        fit_mses, eval_mses = [], []
        fit_r2s, eval_r2s = [], []
        fit_ics, eval_ics = [], []
        for fit_mask, eval_mask in folds:
            X_fit = train_df.loc[fit_mask, FINAL_FEATURE_COLS]
            y_fit = train_df.loc[fit_mask, TARGET_COL]
            X_eval = train_df.loc[eval_mask, FINAL_FEATURE_COLS]
            y_eval = train_df.loc[eval_mask, TARGET_COL]

            model.fit(X_fit, y_fit)
            pred_fit = model.predict(X_fit)
            pred_eval = model.predict(X_eval)

            fit_mses.append(mean_squared_error(y_fit, pred_fit))
            eval_mses.append(mean_squared_error(y_eval, pred_eval))
            fit_r2s.append(r2_score(y_fit, pred_fit))
            eval_r2s.append(r2_score(y_eval, pred_eval))
            fit_ics.append(spearmanr(y_fit, pred_fit).statistic)
            eval_ics.append(spearmanr(y_eval, pred_eval).statistic)

        wandb.log({
            "fit_mse": np.mean(fit_mses),
            "eval_mse_mean": np.mean(eval_mses),
            "eval_mse_std": np.std(eval_mses),
            "fit_r2": np.mean(fit_r2s),
            "eval_r2": np.mean(eval_r2s),
            "fit_ic": np.mean(fit_ics),
            "eval_ic": np.mean(eval_ics),
        })
        wandb.finish()
    return train

print("make_sklearn_train_fn ready")'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task1.py`
Expected: `wrote 5 total cells to 04) train_wandb.ipynb`

- [ ] **Step 3: Execute the notebook for real**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected: no `CellExecutionError` in the command output.

- [ ] **Step 4: Read the actual output and verify**

Run:
```
.venv/bin/python3 -c "
import nbformat
nb = nbformat.read(open('04) train_wandb.ipynb'), as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        for o in c.get('outputs', []):
            if 'text' in o:
                print(f'--- cell {i} ---'); print(o['text'])
            if o.get('output_type') == 'error':
                print(f'!!! cell {i} ERROR !!!', o.get('ename'), o.get('evalue'))
"
```
Expected: `train_df: 8443 rows, target = 'target_bps_vs_mid'`; `4 final features: [...]`; 10 fold lines identical in shape to `03)`'s (fold 0: 7 fit days → fold 9: 16 fit days); `make_sklearn_train_fn ready`. No `ERROR` lines. `wandb.login()` must not have printed an interactive prompt (would indicate `~/.netrc` isn't being picked up — stop and re-check auth if so).

- [ ] **Step 5: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add 04) foundation: data/CV folds, wandb login, shared sklearn sweep runner"
```

---

### Task 2: HuberRegressor sweep

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `make_sklearn_train_fn`, `folds`, `train_df`, `TARGET_COL`, `FINAL_FEATURE_COLS` (Task 1)
- Produces: `huber_sweep_id: str` (kernel-scoped variable, consumed by Task 8)

- [ ] **Step 1: Cheap dry-run validation (outside the notebook)**

Write to `/tmp/dryrun_huber.py`:

```python
import json
import numpy as np
import pandas as pd
import wandb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("data/train.csv", parse_dates=["date"])
columns = json.load(open("data/columns.json"))
TARGET_COL = columns["target_cols"][0]
FINAL_FEATURE_COLS = columns["final_feature_cols"]

unique_dates = np.sort(train_df["date"].unique())
folds = []
for eval_start in range(7, len(unique_dates), 1):
    fit_dates = unique_dates[:eval_start]
    eval_dates = unique_dates[eval_start:eval_start + 1]
    folds.append((train_df["date"].isin(fit_dates), train_df["date"].isin(eval_dates)))

def make_sklearn_train_fn(estimator_class):
    def train():
        wandb.init()
        config = dict(wandb.config)
        if "random_state" in estimator_class().get_params():
            config["random_state"] = 0
        estimator = estimator_class(**config)
        model = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler()), ("model", estimator)])
        fit_mses, eval_mses, fit_r2s, eval_r2s, fit_ics, eval_ics = [], [], [], [], [], []
        for fit_mask, eval_mask in folds:
            X_fit, y_fit = train_df.loc[fit_mask, FINAL_FEATURE_COLS], train_df.loc[fit_mask, TARGET_COL]
            X_eval, y_eval = train_df.loc[eval_mask, FINAL_FEATURE_COLS], train_df.loc[eval_mask, TARGET_COL]
            model.fit(X_fit, y_fit)
            pred_fit, pred_eval = model.predict(X_fit), model.predict(X_eval)
            fit_mses.append(mean_squared_error(y_fit, pred_fit))
            eval_mses.append(mean_squared_error(y_eval, pred_eval))
            fit_r2s.append(r2_score(y_fit, pred_fit)); eval_r2s.append(r2_score(y_eval, pred_eval))
            fit_ics.append(spearmanr(y_fit, pred_fit).statistic)
            eval_ics.append(spearmanr(y_eval, pred_eval).statistic)
        wandb.log({"fit_mse": np.mean(fit_mses), "eval_mse_mean": np.mean(eval_mses),
                    "eval_mse_std": np.std(eval_mses), "fit_r2": np.mean(fit_r2s),
                    "eval_r2": np.mean(eval_r2s), "fit_ic": np.mean(fit_ics), "eval_ic": np.mean(eval_ics)})
        wandb.finish()
    return train

wandb.login()
sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "epsilon": {"distribution": "log_uniform_values", "min": 1.1, "max": 3.0},
        "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e1},
    },
}
sweep_id = wandb.sweep(sweep_config, project="close-auction-model-tuning")
wandb.agent(sweep_id, function=make_sklearn_train_fn(HuberRegressor), count=2)
print("dry-run sweep_id:", sweep_id)
```

Run: `.venv/bin/python3 /tmp/dryrun_huber.py`
Expected: no traceback; ends printing `dry-run sweep_id: <id>`; 2 completed runs visible at the printed W&B URL.

- [ ] **Step 2: Write the nbformat script for the real cells**

Write to `/tmp/build_04_task2.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## HuberRegressor sweep (best linear)

`epsilon` controls where Huber's loss switches from quadratic to linear (robustness to outliers);
`alpha` is the L2 penalty strength. 30-trial Bayesian search, optimizing held-out IC.'''
))

cells.append(nbf.v4.new_code_cell(
'''from sklearn.linear_model import HuberRegressor

huber_sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "epsilon": {"distribution": "log_uniform_values", "min": 1.1, "max": 3.0},
        "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e1},
    },
}

huber_sweep_id = wandb.sweep(huber_sweep_config, project="close-auction-model-tuning")
wandb.agent(huber_sweep_id, function=make_sklearn_train_fn(HuberRegressor), count=30)
print("huber_sweep_id:", huber_sweep_id)'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 3: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task2.py`
Expected: `wrote 7 total cells to 04) train_wandb.ipynb`

- [ ] **Step 4: Execute the notebook for real (background this — Task 1's cells re-run too)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`.

- [ ] **Step 5: Read the actual output and verify**

Use the same read snippet as Task 1 Step 4. Expected: all of Task 1's output unchanged, plus a line `huber_sweep_id: <id>` with no `ERROR` anywhere. Spot-check that `<id>`'s sweep in the W&B project shows 30 completed runs.

- [ ] **Step 6: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add HuberRegressor sweep to 04) -- 30 trials, bayes, optimizing eval_ic"
```

---

### Task 3: Lasso sweep

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `make_sklearn_train_fn`, `folds`, `train_df`, `TARGET_COL`, `FINAL_FEATURE_COLS` (Task 1)
- Produces: `lasso_sweep_id: str` (consumed by Task 8)

- [ ] **Step 1: Cheap dry-run validation**

Same pattern as Task 2 Step 1: copy `/tmp/dryrun_huber.py` to `/tmp/dryrun_lasso.py`, replace the `HuberRegressor` import with `from sklearn.linear_model import Lasso`, replace `sweep_config["parameters"]` with:

```python
    "parameters": {
        "alpha": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e1},
    },
```

and replace `make_sklearn_train_fn(HuberRegressor)` with `make_sklearn_train_fn(Lasso)`.

Run: `.venv/bin/python3 /tmp/dryrun_lasso.py`
Expected: no traceback; prints `dry-run sweep_id: <id>`; 2 completed runs at the printed URL.

- [ ] **Step 2: Write the nbformat script**

Write to `/tmp/build_04_task3.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## Lasso sweep (baseline)

Plain L1-penalized linear regression -- the explicit floor to compare the other four models
against. Only `alpha` to tune. 30-trial Bayesian search, optimizing held-out IC.'''
))

cells.append(nbf.v4.new_code_cell(
'''from sklearn.linear_model import Lasso

lasso_sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "alpha": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e1},
    },
}

lasso_sweep_id = wandb.sweep(lasso_sweep_config, project="close-auction-model-tuning")
wandb.agent(lasso_sweep_id, function=make_sklearn_train_fn(Lasso), count=30)
print("lasso_sweep_id:", lasso_sweep_id)'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 3: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task3.py`
Expected: `wrote 9 total cells to 04) train_wandb.ipynb`

- [ ] **Step 4: Execute the notebook for real (background)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`.

- [ ] **Step 5: Read the actual output and verify**

Same read snippet as Task 1 Step 4. Expected: all prior output unchanged, plus `lasso_sweep_id: <id>`, no `ERROR`. Spot-check 30 completed runs in that sweep.

- [ ] **Step 6: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add Lasso sweep to 04) -- baseline, 30 trials, bayes, optimizing eval_ic"
```

---

### Task 4: GradientBoostingRegressor sweep

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `make_sklearn_train_fn`, `folds`, `train_df`, `TARGET_COL`, `FINAL_FEATURE_COLS` (Task 1)
- Produces: `gb_sweep_id: str` (consumed by Task 8)

- [ ] **Step 1: Cheap dry-run validation**

Copy `/tmp/dryrun_huber.py` to `/tmp/dryrun_gb.py`, replace the import with
`from sklearn.ensemble import GradientBoostingRegressor`, replace `sweep_config["parameters"]` with:

```python
    "parameters": {
        "n_estimators": {"distribution": "int_uniform", "min": 50, "max": 400},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 6},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-3, "max": 3e-1},
        "subsample": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 50},
    },
```

and `make_sklearn_train_fn(GradientBoostingRegressor)`.

Run: `.venv/bin/python3 /tmp/dryrun_gb.py`
Expected: no traceback; prints `dry-run sweep_id: <id>`; 2 completed runs at the printed URL.

- [ ] **Step 2: Write the nbformat script**

Write to `/tmp/build_04_task4.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## GradientBoostingRegressor sweep (best boosted tree)

`min_samples_leaf`'s range goes wide and regularization-leaning on purpose -- `03)`'s shortlist run
showed the untuned tree models overfitting badly, so the sweep needs room to find configs that
generalize, not just variations near sklearn's overfit-prone defaults. 30-trial Bayesian search,
optimizing held-out IC.'''
))

cells.append(nbf.v4.new_code_cell(
'''from sklearn.ensemble import GradientBoostingRegressor

gb_sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"distribution": "int_uniform", "min": 50, "max": 400},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 6},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-3, "max": 3e-1},
        "subsample": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 50},
    },
}

gb_sweep_id = wandb.sweep(gb_sweep_config, project="close-auction-model-tuning")
wandb.agent(gb_sweep_id, function=make_sklearn_train_fn(GradientBoostingRegressor), count=30)
print("gb_sweep_id:", gb_sweep_id)'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 3: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task4.py`
Expected: `wrote 11 total cells to 04) train_wandb.ipynb`

- [ ] **Step 4: Execute the notebook for real (background)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`. This is the first noticeably slow full run (GradientBoosting at up to 400 estimators, 10 folds, 30 trials, plus re-running Huber+Lasso) -- give it real time before checking.

- [ ] **Step 5: Read the actual output and verify**

Same read snippet as Task 1 Step 4. Expected: all prior output unchanged, plus `gb_sweep_id: <id>`, no `ERROR`. Spot-check 30 completed runs in that sweep.

- [ ] **Step 6: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add GradientBoostingRegressor sweep to 04) -- 30 trials, bayes, optimizing eval_ic"
```

---

### Task 5: RandomForestRegressor sweep

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `make_sklearn_train_fn`, `folds`, `train_df`, `TARGET_COL`, `FINAL_FEATURE_COLS` (Task 1)
- Produces: `rf_sweep_id: str` (consumed by Task 8)

- [ ] **Step 1: Cheap dry-run validation**

Copy `/tmp/dryrun_huber.py` to `/tmp/dryrun_rf.py`, replace the import with
`from sklearn.ensemble import RandomForestRegressor`, replace `sweep_config["parameters"]` with:

```python
    "parameters": {
        "n_estimators": {"distribution": "int_uniform", "min": 50, "max": 400},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 15},
        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 50},
        "max_features": {"values": ["sqrt", "log2", 0.5, 1.0]},
    },
```

and `make_sklearn_train_fn(RandomForestRegressor)`.

Run: `.venv/bin/python3 /tmp/dryrun_rf.py`
Expected: no traceback; prints `dry-run sweep_id: <id>`; 2 completed runs at the printed URL.

- [ ] **Step 2: Write the nbformat script**

Write to `/tmp/build_04_task5.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## RandomForestRegressor sweep (best bagged tree)

Same regularization-leaning `min_samples_leaf` range as GradientBoosting above, for the same
reason -- `03)`'s shortlist run showed this exact model overfitting badly at sklearn defaults
(fit MSE 3.0 vs. eval MSE 15.4, a 5x gap). 30-trial Bayesian search, optimizing held-out IC.'''
))

cells.append(nbf.v4.new_code_cell(
'''from sklearn.ensemble import RandomForestRegressor

rf_sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"distribution": "int_uniform", "min": 50, "max": 400},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 15},
        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 50},
        "max_features": {"values": ["sqrt", "log2", 0.5, 1.0]},
    },
}

rf_sweep_id = wandb.sweep(rf_sweep_config, project="close-auction-model-tuning")
wandb.agent(rf_sweep_id, function=make_sklearn_train_fn(RandomForestRegressor), count=30)
print("rf_sweep_id:", rf_sweep_id)'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 3: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task5.py`
Expected: `wrote 13 total cells to 04) train_wandb.ipynb`

- [ ] **Step 4: Execute the notebook for real (background)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`.

- [ ] **Step 5: Read the actual output and verify**

Same read snippet as Task 1 Step 4. Expected: all prior output unchanged, plus `rf_sweep_id: <id>`, no `ERROR`. Spot-check 30 completed runs in that sweep.

- [ ] **Step 6: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add RandomForestRegressor sweep to 04) -- 30 trials, bayes, optimizing eval_ic"
```

---

### Task 6: PyTorch MLP architecture + per-fold training loop (standalone, not yet swept)

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `folds`, `train_df`, `TARGET_COL`, `FINAL_FEATURE_COLS` (Task 1)
- Produces:
  - `MLPNet(nn.Module)` — constructor `MLPNet(input_dim: int, hidden_layer_sizes: tuple[int, ...], activation: str)`, `forward(x) -> Tensor` of shape `(batch,)`
  - `train_one_fold_torch(X_fit, y_fit, X_eval, y_eval, hidden_layer_sizes, activation, alpha, learning_rate_init, max_epochs=200, patience=20, log_epochs=False) -> tuple[np.ndarray, np.ndarray]` — returns `(pred_fit, pred_eval)` using best early-stopped weights; consumed by Task 7

- [ ] **Step 1: Write the nbformat script**

Write to `/tmp/build_04_task6.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## MLP: PyTorch architecture + per-fold training loop

sklearn's `MLPRegressor` only supports `{identity, logistic, tanh, relu}` activations and has no
clean hook for per-epoch validation MSE across arbitrary CV folds -- both needed here (leaky ReLU,
and a real train/eval-vs-epoch curve). Built as a small `torch.nn` feedforward net instead. The
final layer is always a plain `nn.Linear` with no activation after it -- linear output for a
continuous regression target.

`train_one_fold_torch` runs one fold: Adam optimizer, MSE loss, early stopping on that fold's eval
MSE (patience 20, max 200 epochs, keeps the best-epoch weights). `log_epochs=True` logs a real
per-epoch curve to W&B -- used for fold 9 only in Task 7, never here.'''
))

cells.append(nbf.v4.new_code_cell(
'''import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, activation):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers = []
        prev_size = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(act_cls())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))  # linear output, no activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold_torch(X_fit, y_fit, X_eval, y_eval, hidden_layer_sizes, activation,
                          alpha, learning_rate_init, max_epochs=200, patience=20,
                          log_epochs=False):
    device = torch.device("cpu")
    X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=device)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32, device=device)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.float32, device=device)

    model = MLPNet(X_fit.shape[1], tuple(hidden_layer_sizes), activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init, weight_decay=alpha)
    loss_fn = nn.MSELoss()

    best_eval_mse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_fit_t), y_fit_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_mse = loss_fn(model(X_fit_t), y_fit_t).item()
            eval_mse = loss_fn(model(X_eval_t), y_eval_t).item()

        if log_epochs:
            wandb.log({"epoch": epoch, "fold9_train_mse": train_mse, "fold9_eval_mse": eval_mse})

        if eval_mse < best_eval_mse:
            best_eval_mse = eval_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_fit = model(X_fit_t).numpy()
        pred_eval = model(X_eval_t).numpy()
    return pred_fit, pred_eval

print("MLPNet and train_one_fold_torch ready")'''
))

cells.append(nbf.v4.new_markdown_cell(
'''**Sanity check** -- run one fold (fold 9, the largest) directly, no W&B, no sweep, just confirming
it trains and the loss actually goes down before wiring this into a 30-trial sweep.'''
))

cells.append(nbf.v4.new_code_cell(
'''_fit_mask, _eval_mask = folds[9]
_X_fit_raw = train_df.loc[_fit_mask, FINAL_FEATURE_COLS].to_numpy()
_y_fit = train_df.loc[_fit_mask, TARGET_COL].to_numpy()
_X_eval_raw = train_df.loc[_eval_mask, FINAL_FEATURE_COLS].to_numpy()
_y_eval = train_df.loc[_eval_mask, TARGET_COL].to_numpy()

_imputer = SimpleImputer(strategy="median").fit(_X_fit_raw)
_X_fit_imp = _imputer.transform(_X_fit_raw)
_X_eval_imp = _imputer.transform(_X_eval_raw)
_scaler = StandardScaler().fit(_X_fit_imp)
_X_fit_scaled = _scaler.transform(_X_fit_imp)
_X_eval_scaled = _scaler.transform(_X_eval_imp)

_pred_fit, _pred_eval = train_one_fold_torch(
    _X_fit_scaled, _y_fit, _X_eval_scaled, _y_eval,
    hidden_layer_sizes=(64, 32), activation="relu",
    alpha=1e-3, learning_rate_init=1e-2,
)
print("sanity-check fold 9 -- fit MSE:", mean_squared_error(_y_fit, _pred_fit),
      " eval MSE:", mean_squared_error(_y_eval, _pred_eval))'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task6.py`
Expected: `wrote 17 total cells to 04) train_wandb.ipynb`

- [ ] **Step 3: Execute the notebook for real (background)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`.

- [ ] **Step 4: Read the actual output and verify**

Same read snippet as Task 1 Step 4. Expected: all prior output unchanged, plus `MLPNet and train_one_fold_torch ready` and a `sanity-check fold 9 -- fit MSE: <number>  eval MSE: <number>` line with finite, non-NaN numbers roughly in the same order of magnitude as the sklearn models' fold-9 fit/eval MSE (single-digit to low-double-digit bps^2). No `ERROR`. If either number is `nan` or absurdly large, stop and debug before Task 7 — do not wire a broken training loop into a 30-trial sweep.

- [ ] **Step 5: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add PyTorch MLP architecture + per-fold training loop to 04), sanity-checked on fold 9"
```

---

### Task 7: MLP sweep (wires Task 6 into W&B, per-epoch logging on fold 9)

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `MLPNet`, `train_one_fold_torch` (Task 6); `folds`, `train_df`, `TARGET_COL`, `FINAL_FEATURE_COLS` (Task 1)
- Produces: `mlp_sweep_id: str` (consumed by Task 8)

- [ ] **Step 1: Cheap dry-run validation (outside the notebook)**

Write to `/tmp/dryrun_mlp.py` — this duplicates `MLPNet`/`train_one_fold_torch` from Task 6 plus a
minimal `mlp_train()` for a fast 2-trial check:

```python
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("data/train.csv", parse_dates=["date"])
columns = json.load(open("data/columns.json"))
TARGET_COL = columns["target_cols"][0]
FINAL_FEATURE_COLS = columns["final_feature_cols"]

unique_dates = np.sort(train_df["date"].unique())
folds = []
for eval_start in range(7, len(unique_dates), 1):
    fit_dates = unique_dates[:eval_start]
    eval_dates = unique_dates[eval_start:eval_start + 1]
    folds.append((train_df["date"].isin(fit_dates), train_df["date"].isin(eval_dates)))

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}

class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, activation):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers, prev_size = [], input_dim
        for size in hidden_layer_sizes:
            layers += [nn.Linear(prev_size, size), act_cls()]
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_one_fold_torch(X_fit, y_fit, X_eval, y_eval, hidden_layer_sizes, activation,
                          alpha, learning_rate_init, max_epochs=200, patience=20, log_epochs=False):
    X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
    y_eval_t = torch.tensor(y_eval, dtype=torch.float32)
    model = MLPNet(X_fit.shape[1], tuple(hidden_layer_sizes), activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init, weight_decay=alpha)
    loss_fn = nn.MSELoss()
    best_eval_mse, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train(); optimizer.zero_grad()
        loss = loss_fn(model(X_fit_t), y_fit_t); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mse = loss_fn(model(X_fit_t), y_fit_t).item()
            eval_mse = loss_fn(model(X_eval_t), y_eval_t).item()
        if log_epochs:
            wandb.log({"epoch": epoch, "fold9_train_mse": train_mse, "fold9_eval_mse": eval_mse})
        if eval_mse < best_eval_mse:
            best_eval_mse, best_state, bad_epochs = eval_mse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return model(X_fit_t).numpy(), model(X_eval_t).numpy()

def mlp_train():
    wandb.init()
    config = wandb.config
    fit_mses, eval_mses, fit_r2s, eval_r2s, fit_ics, eval_ics = [], [], [], [], [], []
    for fold_idx, (fit_mask, eval_mask) in enumerate(folds):
        X_fit_raw = train_df.loc[fit_mask, FINAL_FEATURE_COLS].to_numpy()
        y_fit = train_df.loc[fit_mask, TARGET_COL].to_numpy()
        X_eval_raw = train_df.loc[eval_mask, FINAL_FEATURE_COLS].to_numpy()
        y_eval = train_df.loc[eval_mask, TARGET_COL].to_numpy()
        imputer = SimpleImputer(strategy="median").fit(X_fit_raw)
        X_fit_imp, X_eval_imp = imputer.transform(X_fit_raw), imputer.transform(X_eval_raw)
        scaler = StandardScaler().fit(X_fit_imp)
        X_fit_scaled, X_eval_scaled = scaler.transform(X_fit_imp), scaler.transform(X_eval_imp)
        pred_fit, pred_eval = train_one_fold_torch(
            X_fit_scaled, y_fit, X_eval_scaled, y_eval,
            hidden_layer_sizes=config["hidden_layer_sizes"], activation=config["activation"],
            alpha=config["alpha"], learning_rate_init=config["learning_rate_init"],
            log_epochs=(fold_idx == 9),
        )
        from sklearn.metrics import mean_squared_error, r2_score
        fit_mses.append(mean_squared_error(y_fit, pred_fit)); eval_mses.append(mean_squared_error(y_eval, pred_eval))
        fit_r2s.append(r2_score(y_fit, pred_fit)); eval_r2s.append(r2_score(y_eval, pred_eval))
        fit_ics.append(spearmanr(y_fit, pred_fit).statistic); eval_ics.append(spearmanr(y_eval, pred_eval).statistic)
    wandb.log({"fit_mse": np.mean(fit_mses), "eval_mse_mean": np.mean(eval_mses),
                "eval_mse_std": np.std(eval_mses), "fit_r2": np.mean(fit_r2s),
                "eval_r2": np.mean(eval_r2s), "fit_ic": np.mean(fit_ics), "eval_ic": np.mean(eval_ics)})
    wandb.finish()

wandb.login()
mlp_sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "hidden_layer_sizes": {"values": [[16], [32], [64], [128], [32, 16], [64, 32],
                                            [128, 64], [64, 32, 16], [128, 64, 32], [100, 50]]},
        "activation": {"values": ["relu", "leaky_relu"]},
        "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e0},
        "learning_rate_init": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
    },
}
sweep_id = wandb.sweep(mlp_sweep_config, project="close-auction-model-tuning")
wandb.agent(sweep_id, function=mlp_train, count=2)
print("dry-run sweep_id:", sweep_id)
```

Run: `.venv/bin/python3 /tmp/dryrun_mlp.py`
Expected: no traceback; prints `dry-run sweep_id: <id>`; 2 completed runs at the printed URL, each
showing logged `epoch`/`fold9_train_mse`/`fold9_eval_mse` points in its run history (confirms
per-epoch logging actually fired for fold 9).

- [ ] **Step 2: Write the nbformat script for the real cells**

Write to `/tmp/build_04_task7.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## MLP sweep (best neural net)

Wires `MLPNet`/`train_one_fold_torch` into a real W&B sweep. Same impute/scale-per-fold discipline
as the sklearn models (fit-fold statistics only, applied to both sides). Fold 9 -- the largest --
gets its per-epoch train/eval MSE logged to W&B on every trial; the other 9 folds still contribute
their final-epoch (best early-stopped) metrics to the aggregate. 30-trial Bayesian search,
optimizing held-out IC.'''
))

cells.append(nbf.v4.new_code_cell(
'''def mlp_train():
    wandb.init()
    config = wandb.config

    fit_mses, eval_mses = [], []
    fit_r2s, eval_r2s = [], []
    fit_ics, eval_ics = [], []
    for fold_idx, (fit_mask, eval_mask) in enumerate(folds):
        X_fit_raw = train_df.loc[fit_mask, FINAL_FEATURE_COLS].to_numpy()
        y_fit = train_df.loc[fit_mask, TARGET_COL].to_numpy()
        X_eval_raw = train_df.loc[eval_mask, FINAL_FEATURE_COLS].to_numpy()
        y_eval = train_df.loc[eval_mask, TARGET_COL].to_numpy()

        imputer = SimpleImputer(strategy="median").fit(X_fit_raw)
        X_fit_imp = imputer.transform(X_fit_raw)
        X_eval_imp = imputer.transform(X_eval_raw)
        scaler = StandardScaler().fit(X_fit_imp)
        X_fit_scaled = scaler.transform(X_fit_imp)
        X_eval_scaled = scaler.transform(X_eval_imp)

        pred_fit, pred_eval = train_one_fold_torch(
            X_fit_scaled, y_fit, X_eval_scaled, y_eval,
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation=config["activation"],
            alpha=config["alpha"],
            learning_rate_init=config["learning_rate_init"],
            log_epochs=(fold_idx == 9),
        )

        fit_mses.append(mean_squared_error(y_fit, pred_fit))
        eval_mses.append(mean_squared_error(y_eval, pred_eval))
        fit_r2s.append(r2_score(y_fit, pred_fit))
        eval_r2s.append(r2_score(y_eval, pred_eval))
        fit_ics.append(spearmanr(y_fit, pred_fit).statistic)
        eval_ics.append(spearmanr(y_eval, pred_eval).statistic)

    wandb.log({
        "fit_mse": np.mean(fit_mses),
        "eval_mse_mean": np.mean(eval_mses),
        "eval_mse_std": np.std(eval_mses),
        "fit_r2": np.mean(fit_r2s),
        "eval_r2": np.mean(eval_r2s),
        "fit_ic": np.mean(fit_ics),
        "eval_ic": np.mean(eval_ics),
    })
    wandb.finish()

mlp_sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval_ic", "goal": "maximize"},
    "parameters": {
        "hidden_layer_sizes": {"values": [[16], [32], [64], [128], [32, 16], [64, 32],
                                            [128, 64], [64, 32, 16], [128, 64, 32], [100, 50]]},
        "activation": {"values": ["relu", "leaky_relu"]},
        "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e0},
        "learning_rate_init": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
    },
}

mlp_sweep_id = wandb.sweep(mlp_sweep_config, project="close-auction-model-tuning")
wandb.agent(mlp_sweep_id, function=mlp_train, count=30)
print("mlp_sweep_id:", mlp_sweep_id)'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 3: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task7.py`
Expected: `wrote 19 total cells to 04) train_wandb.ipynb`

- [ ] **Step 4: Execute the notebook for real (background — this is the slowest task so far)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`.

- [ ] **Step 5: Read the actual output and verify**

Same read snippet as Task 1 Step 4. Expected: all prior output unchanged, plus `mlp_sweep_id: <id>`,
no `ERROR`. Spot-check 30 completed runs in that sweep, and that at least one run's history contains
`fold9_train_mse`/`fold9_eval_mse` points across multiple epochs (not just a single value).

- [ ] **Step 6: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add MLP sweep to 04) -- PyTorch, 30 trials, bayes, optimizing eval_ic, fold-9 epoch curves"
```

---

### Task 8: Final comparison table

**Files:**
- Modify: `04) train_wandb.ipynb`

**Interfaces:**
- Consumes: `huber_sweep_id`, `lasso_sweep_id`, `gb_sweep_id`, `rf_sweep_id`, `mlp_sweep_id` (Tasks 2–5, 7)
- Produces: printed `comparison_df` (this is the notebook's deliverable — nothing downstream consumes it in code)

- [ ] **Step 1: Write the nbformat script**

Write to `/tmp/build_04_task8.py`:

```python
import nbformat as nbf

PATH = "04) train_wandb.ipynb"
nb = nbf.read(open(PATH), as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell(
'''## Comparison: best run per model

Pulls each sweep's best run (by its own optimize target, `eval_ic`) via the W&B API and prints one
row per model: every metric logged during training, plus that model's winning hyperparameters.'''
))

cells.append(nbf.v4.new_code_cell(
'''api = wandb.Api()
entity = api.default_entity

sweep_ids = {
    "HuberRegressor (best linear)": huber_sweep_id,
    "Lasso (baseline)": lasso_sweep_id,
    "GradientBoostingRegressor (best boosted)": gb_sweep_id,
    "RandomForestRegressor (best bagged)": rf_sweep_id,
    "MLP (best neural net, PyTorch)": mlp_sweep_id,
}

METRIC_KEYS = ["fit_mse", "eval_mse_mean", "eval_mse_std", "fit_r2", "eval_r2", "fit_ic", "eval_ic"]

comparison_rows = []
for model_name, sweep_id in sweep_ids.items():
    sweep = api.sweep(f"{entity}/close-auction-model-tuning/{sweep_id}")
    best_run = sweep.best_run()
    row = {"model": model_name}
    row.update({k: best_run.summary_metrics.get(k) for k in METRIC_KEYS})
    row["best_hyperparameters"] = dict(best_run.config)
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
pd.set_option("display.max_colwidth", None)
print(comparison_df.to_string(index=False))'''
))

nb["cells"] = cells
nbf.write(nb, open(PATH, "w"))
print(f"wrote {len(cells)} total cells to {PATH}")
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python3 /tmp/build_04_task8.py`
Expected: `wrote 21 total cells to 04) train_wandb.ipynb`

- [ ] **Step 3: Execute the notebook for real (background — full cumulative run, all 5 sweeps)**

Run in background: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace "04) train_wandb.ipynb"`
Expected on completion: no `CellExecutionError`.

- [ ] **Step 4: Read the actual output and verify**

Same read snippet as Task 1 Step 4. Expected: all prior tasks' output present, plus a 5-row table
(one row per model) with populated numeric values in every `METRIC_KEYS` column (no `NaN`/`None`)
and a non-empty `best_hyperparameters` dict per row. No `ERROR` anywhere in the whole notebook.

- [ ] **Step 5: Commit**

```bash
git add "04) train_wandb.ipynb"
git commit -m "Add final comparison table to 04) -- best run per model, full metrics + hyperparameters"
```

---

## Self-Review Notes

**Spec coverage:**
- Auth via `~/.netrc`, no `.env` → Task 1 (`wandb.login()`, no args). ✓
- IC as sweep objective for all five sweeps → every sweep's `metric` dict in Tasks 2, 3, 4, 5, 7. ✓
- Five separate sweeps, shared sklearn train-fn factory → Task 1 (`make_sklearn_train_fn`) + Tasks 2–5. ✓
- Per-model sklearn search spaces (table in spec) → Tasks 2–5, parameter dicts copied verbatim. ✓
- MLP: PyTorch, leaky ReLU, linear output layer, expanded layer sizes, per-fold training with early stopping, fold-9-only per-epoch logging → Tasks 6–7. ✓
- Full metric set logged (not just optimize target) → `make_sklearn_train_fn` (Task 1) and `mlp_train` (Task 7) both log all 7 metrics. ✓
- Final comparison table with full metrics + hyperparameters, pulled via W&B API → Task 8. ✓
- 30 trials, Bayesian, `close-auction-model-tuning` project → present in every sweep config. ✓

**Type consistency:** `train_one_fold_torch`'s signature (Task 6) matches its call site in `mlp_train` (Task 7) exactly — same parameter names, same positional/keyword usage. `make_sklearn_train_fn`'s return signature (a zero-arg closure) matches how `wandb.agent(..., function=...)` is called in Tasks 2–5.
