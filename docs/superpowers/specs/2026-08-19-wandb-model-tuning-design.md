# W&B hyperparameter/architecture tuning for the model shortlist

## Context

`03) models_and_feature_selection.ipynb` ends with a LazyPredict scan over the
final 4-feature set (`FINAL_FEATURE_COLS` in `data/columns.json`) and picks one
model per family as a shortlist: `HuberRegressor` (best linear), `Lasso`
(baseline), `MLPRegressor` (best neural net), `GradientBoostingRegressor`
(best boosted tree), `RandomForestRegressor` (best bagged tree). All five are
still at sklearn defaults. `04) train_wandb.ipynb` exists but is empty except
for a title, reserved for this work.

Goal: tune architecture (where applicable) and hyperparameters for all five
shortlisted models using Weights & Biases sweeps, evaluated with real rigor
(the same day-respecting, no-lookahead CV already used everywhere else in
this project), and end with a clear "best config per model" comparison.

## Auth

`wandb login` already run by the user in a separate terminal (not through the
coding agent), caching credentials to `~/.netrc`. The notebook calls
`wandb.login()` with no arguments -- picks up the cached credentials silently,
no interactive prompt, works fine under headless `jupyter nbconvert --execute`.
No `.env` file, no API key ever passed through or read by the coding agent.

## Sweep objective: IC, not MSE

All five sweeps optimize `metric: {name: eval_ic, goal: maximize}` --
held-out Spearman correlation between prediction and target, averaged across
the 10 CV folds -- not `eval_mse_mean`. This project's strategy trades on
direction and relative magnitude (cross the spread on the predicted-strongest
names, exit via MOC), not on hitting an exact bps value, and IC directly
measures ranking quality where MSE is dominated by getting the bulk of
near-zero rows slightly-more-right. This doesn't change any model's internal
training loss (all five still fit by minimizing their own default loss,
usually MSE-flavored) -- it only changes which hyperparameter configuration
the sweep reports as "best."

## Structure

Five separate W&B sweeps, one per model -- not one unified sweep with a
model-type branch, since each model's tunable space is genuinely different
(MLP has real architecture choices; the others don't).

For the four sklearn models (`HuberRegressor`, `Lasso`,
`GradientBoostingRegressor`, `RandomForestRegressor`):

1. A `sweep_config` dict: `method: bayes`, `metric: {name: eval_ic, goal: maximize}`,
   and a `parameters` block matching that model's search space (below).
2. A `train()` function, called by `wandb.agent()` for each trial:
   - Starts a run via `wandb.init()` (sweep-managed, so `wandb.config` holds
     the trial's sampled hyperparameters).
   - Builds `Pipeline([("impute", SimpleImputer(strategy="median")),
     ("scale", StandardScaler()), ("model", <estimator>(**wandb.config))])`.
   - Evaluates across the same 10-fold expanding-window CV folds used in
     `03)` (`MIN_TRAIN_DAYS=7`, `DELTA_DAYS=1`, walk-forward, no lookahead) --
     rebuilt here from `data/train.csv`/`data/columns.json` fresh, since `04)`
     loads its own data independently like every other notebook in this
     project.
   - Logs `eval_ic` (the sweep's optimize target), plus `fit_mse`,
     `eval_mse_mean`, `eval_mse_std`, `fit_r2`, `eval_r2`, `fit_ic` via
     `wandb.log(...)` -- full metric set, not just the target.
3. `wandb.sweep(sweep_config, project="close-auction-model-tuning")` then
   `wandb.agent(sweep_id, function=train, count=30)` -- 30 trials per model,
   Bayesian search.

The MLP sweep follows the same sweep/agent/30-trials shell but its `train()`
function is different -- see below.

Five sweeps x 30 trials x 10 folds = 1,500 total model fits (plus the MLP's
own per-epoch training within each fold). Expect a runtime of tens of
minutes, dominated by the tree models and the MLP.

## Per-model search spaces (sklearn models)

| Model | Parameters |
|---|---|
| `HuberRegressor` | `epsilon`: log-uniform 1.1-3.0; `alpha`: log-uniform 1e-5-1e1 |
| `Lasso` | `alpha`: log-uniform 1e-4-1e1 |
| `GradientBoostingRegressor` | `n_estimators`: int-uniform 50-400; `max_depth`: int-uniform 2-6; `learning_rate`: log-uniform 1e-3-3e-1; `subsample`: uniform 0.5-1.0; `min_samples_leaf`: int-uniform 1-50 |
| `RandomForestRegressor` | `n_estimators`: int-uniform 50-400; `max_depth`: int-uniform 2-15; `min_samples_leaf`: int-uniform 1-50; `max_features`: categorical `sqrt`/`log2`/`0.5`/`1.0` |

`min_samples_leaf`'s wide, regularization-leaning range on both tree models is
deliberate: `03)`'s shortlist run showed `RandomForestRegressor` overfitting
badly at sklearn defaults (fit MSE 3.0 vs. eval MSE 15.4, a 5x gap), so the
sweep needs room to find configs that generalize, not just variations near an
overfit-prone default.

## MLP: PyTorch, not sklearn

sklearn's `MLPRegressor` only supports `activation in {identity, logistic,
tanh, relu}` and offers no clean hook for per-epoch validation MSE across
arbitrary CV folds -- both required below. The MLP is a small `torch.nn`
feedforward network instead (new dependency: `torch`). The other four models
are unaffected.

**Architecture** (all swept):

- `hidden_layer_sizes`: categorical over `(16,)`, `(32,)`, `(64,)`, `(128,)`,
  `(32,16)`, `(64,32)`, `(128,64)`, `(64,32,16)`, `(128,64,32)`, `(100,50)`.
- `activation`: categorical `relu` / `leaky_relu` (`nn.LeakyReLU` default
  negative slope 0.01), applied after every hidden `nn.Linear` layer.
- **Final layer is always `nn.Linear(prev_size, 1)` with no activation
  applied after it** -- linear output for a continuous regression target,
  never gated through relu/leaky_relu/anything else.
- `alpha`: log-uniform 1e-5-1e0 (L2 weight decay, passed to the optimizer).
- `learning_rate_init`: log-uniform 1e-4-1e-1 (Adam).

**Training loop per fold**: max 200 epochs, Adam optimizer, MSE loss,
early stopping with patience 20 on that fold's eval MSE (stop if no
improvement for 20 consecutive epochs, keep the best-epoch weights).

**Per-epoch logging**: for fold 9 only (the largest fold, 16 fit days) --
`wandb.log({"epoch": e, "fold9_train_mse": ..., "fold9_eval_mse": ...})`
every epoch, giving a real train/eval MSE-vs-epoch curve in the W&B UI for
that trial. The other 9 folds still train and report their final metrics
(after early stopping) into the same aggregate `eval_mse_mean`/`eval_ic`/etc.
as the sklearn models, just without per-epoch logging -- logging full curves
for all 10 folds x up to 200 epochs x 30 trials would be on the order of
60,000 log points for this model alone and unreadable as a dashboard.

## Wrap-up

After all five sweeps finish, pull each sweep's best run via the W&B API
(`wandb.Api().sweep(sweep_id).best_run()`) and print a final table with one
row per model, showing every metric logged during training: `fit_mse`,
`eval_mse_mean`, `eval_mse_std`, `fit_r2`, `eval_r2`, `fit_ic`, `eval_ic`,
plus that model's best hyperparameters. This table is the notebook's actual
deliverable -- picking a final single model/config is a separate next step,
not decided here.

## Testing / validation

Executed the normal way for this project: `jupyter nbconvert --to notebook
--execute --inplace`, verify zero cell errors, then read the actual printed
final comparison table (not just "it ran") before considering this done.
