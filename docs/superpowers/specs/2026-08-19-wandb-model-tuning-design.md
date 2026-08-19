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

## Structure

Five separate W&B sweeps, one per model -- not one unified sweep with a
model-type branch, since each model's tunable space is genuinely different
(MLP has real architecture choices; the others don't).

For each model:

1. A `sweep_config` dict: `method: bayes`, `metric: {name: eval_mse_mean, goal: minimize}`,
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
   - Logs `eval_mse_mean` (the sweep's optimize target), `eval_mse_std`,
     `fit_mse`, `eval_r2`, `eval_ic` via `wandb.log(...)`.
3. `wandb.sweep(sweep_config, project="close-auction-model-tuning")` then
   `wandb.agent(sweep_id, function=train, count=30)` -- 30 trials per model,
   Bayesian search.

Five sweeps x 30 trials x 10 folds = 1,500 total model fits. Expect a runtime
of tens of minutes, dominated by the tree models (`GradientBoostingRegressor`,
`RandomForestRegressor`) at larger `n_estimators` draws.

## Per-model search spaces

| Model | Parameters |
|---|---|
| `HuberRegressor` | `epsilon`: log-uniform 1.1-3.0; `alpha`: log-uniform 1e-5-1e1 |
| `Lasso` | `alpha`: log-uniform 1e-4-1e1 |
| `MLPRegressor` | `hidden_layer_sizes`: categorical over `(32,)`, `(64,)`, `(32,16)`, `(64,32)`, `(100,)`, `(128,64)`; `activation`: categorical `relu`/`tanh`; `alpha`: log-uniform 1e-5-1e0; `learning_rate_init`: log-uniform 1e-4-1e-1 |
| `GradientBoostingRegressor` | `n_estimators`: int-uniform 50-400; `max_depth`: int-uniform 2-6; `learning_rate`: log-uniform 1e-3-3e-1; `subsample`: uniform 0.5-1.0; `min_samples_leaf`: int-uniform 1-50 |
| `RandomForestRegressor` | `n_estimators`: int-uniform 50-400; `max_depth`: int-uniform 2-15; `min_samples_leaf`: int-uniform 1-50; `max_features`: categorical `sqrt`/`log2`/`0.5`/`1.0` |

`min_samples_leaf`'s wide, regularization-leaning range on both tree models is
deliberate: `03)`'s shortlist run showed `RandomForestRegressor` overfitting
badly at sklearn defaults (fit MSE 3.0 vs. eval MSE 15.4, a 5x gap), so the
sweep needs room to find configs that generalize, not just variations near an
overfit-prone default. `MLPRegressor`'s `max_iter` is fixed high (2000) with
sklearn's default early-stopping-free training, not swept -- convergence
budget, not a tunable choice here.

## Wrap-up

After all five sweeps finish, pull each sweep's best run via the W&B API
(`wandb.Api().sweep(sweep_id).best_run()`) and print a final table: model,
best hyperparameters, best `eval_mse_mean`/`eval_r2`/`eval_ic`. This table is
the notebook's actual deliverable -- picking a final single model/config is a
separate next step, not decided here.

## Testing / validation

Executed the normal way for this project: `jupyter nbconvert --to notebook
--execute --inplace`, verify zero cell errors, then read the actual printed
final comparison table (not just "it ran") before considering this done.
