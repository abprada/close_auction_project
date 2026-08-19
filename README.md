# Closing Auction Prediction

Research project predicting Nasdaq closing-cross price moves from pre-close order-imbalance
data, and backtesting a long/short strategy built on those predictions.

## Data

`data/202503_imbalance.tar` — Nasdaq closing-auction imbalance feed, 21 trading days
(2025-03-03 to 2025-03-31), snapshots every 10s from 15:50-15:55 ET and every 1s from 15:55 to
16:00 ET. 500 symbols/day (562 unique across the month, universe rotates daily), ~4.1M rows raw.

**Target**: `target_bps_vs_mid = (close_price / mid - 1) * 1e4`, where `mid = (bid + ask) / 2` as
of the last pre-close snapshot. Idealized and cost-free by construction — chosen over raw
`close_price` because price levels aren't comparable across symbols, and over targets referenced
to Nasdaq's own auction-price estimates (`near_price`/`far_price`) because that's a different
question from predicting the move itself.

**Final feature set** (selected in `03)` via lasso bootstrap t-stats + SHAP/SAGE on a LightGBM
model): `near_price_vs_mid_bps`, `urgency`, `ref_price_vs_mid_bps`, `far_price_vs_mid_bps`. Full
feature glossary and selection rationale in `02)`'s Feature Reference and `03)`'s Conclusion.

## Workflow

Notebooks run in numbered order, each building on the last:

1. **`01) data_exploration.ipynb`** — data inventory, univariate analysis, data quality issues,
   lookahead-bias check (`cross` is the auction print itself, delivered in-band pre-close as
   `NaN` — never forward-fill it), and panel construction. Findings written up in
   `1) data_exploration_report.md`.
2. **`02) features_and_target.ipynb`** — feature engineering (imbalance/spread/urgency,
   ADV-normalized sizes, short-horizon momentum and rolling vol), target definition, train/test
   split, correlation analysis, pruning of non-stationary/scale-variant raw price levels.
3. **`03) models_and_feature_selection.ipynb`** — expanding-window CV, elastic net with
   bootstrap t-stats, a small LightGBM model with SHAP/SAGE, cross-checked against each other to
   pick the final 4-feature set, then a LazyPredict scan to shortlist model families.
4. **`04) train_wandb.ipynb`** — Bayesian hyperparameter sweeps (Weights & Biases, 100 trials
   each) over the shortlisted families: Huber, Lasso, GradientBoosting, RandomForest, and a
   PyTorch MLP. Best configs per family saved to `best_hyperparameters.json`.
5. **`05) backtest.ipynb`** — long/short bucket backtest using the tuned models, walk-forward
   folds, run across three entry snapshot times (15:59:59, 15:59:00, 15:55:00) to check whether
   performance survives trading earlier than the last second before close.

## Repository layout

```
01) data_exploration.ipynb          EDA, data quality, lookahead check
1) data_exploration_report.md       Write-up of 01)'s findings
02) features_and_target.ipynb       Feature engineering, target, train/test split
03) models_and_feature_selection.ipynb   CV, elastic net, SHAP/SAGE, final feature set
04) train_wandb.ipynb               W&B hyperparameter sweeps per model family
05) backtest.ipynb                  Long/short backtest, walk-forward, robustness check
best_hyperparameters.json           Best config per model family, from 04)
data/                               Raw + processed data (mostly gitignored, see below)
docs/                               Take-home brief (Nasdaq closing auction PDF)
research/                           Background papers, gitignored — see research/README.md
requirements.txt
```

`data/` is gitignored except `panel_clean.csv`, `train.csv`, `test.csv`, and `columns.json` — the
raw `.tar` and any other derived files are excluded. `research/*.pdf` is gitignored too;
`research/README.md` has full citations and links to what's there.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`04)` logs sweeps to Weights & Biases — run `wandb login` once before executing that notebook.

## Working notes

- No production concerns here — this is research/EDA code, optimized for correctness of
  conclusions over robustness to edge cases.
- Lookahead bias is treated as the primary risk throughout (see `01)` Section 4 in particular);
  each notebook that touches CV or the target documents how it avoids leaking future information
  into a fold.
