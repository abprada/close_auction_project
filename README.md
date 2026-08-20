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

**Result** (see `06)`): an MLP + quintile long/short bucket strategy entered at `15:58:00` posts
Sharpe 14.5 (annualized) on the 17-day train period and a directionally consistent but not
statistically significant Sharpe 5.6 on the 4-day held-out test period (`p = 0.53`). A promising
research result, not yet a validated trading strategy — execution feasibility in particular is
still open (NASDAQ's MOC/LOC order cutoff is `15:55:00`, three minutes before this entry).

## Workflow

Notebooks run in numbered order, each building on the last:

1. **`01) data_exploration.ipynb`** — data inventory, univariate analysis, data quality issues,
   lookahead-bias check (`cross` is the auction print itself, delivered in-band pre-close as
   `NaN` — never forward-fill it), and panel construction. Findings written up in
   `docs/1) data_exploration_report.md`.
2. **`02) features_and_target.ipynb`** — feature engineering (imbalance/spread/urgency,
   ADV-normalized sizes, short-horizon momentum and rolling vol), target definition, train/test
   split, correlation analysis, pruning of non-stationary/scale-variant raw price levels.
3. **`03) models_and_feature_selection.ipynb`** — expanding-window CV, elastic net with
   bootstrap t-stats, a small LightGBM model with SHAP/SAGE, cross-checked against each other to
   pick the final 4-feature set, then a LazyPredict scan to shortlist model families.
4. **`04) train_wandb.ipynb`** — Bayesian hyperparameter sweeps (Weights & Biases, 100 trials
   each) over the shortlisted families: Huber, Lasso, GradientBoosting, RandomForest, and a
   PyTorch MLP. Best configs per family saved to `best_hyperparameters.json`.
5. **`05) backtest.ipynb`** — long/short bucket backtest using the tuned models: realized P&L
   (long fills at `ask`, short at `bid`, both exit via a guaranteed-fill Market-on-Close order at
   `close_price`, not the idealized mid-based training target), walk-forward folds, swept across
   4 bucket-construction schemes (50/50, quartile, quintile, decile) × 5 tuned model classes × 6
   entry timings (`15:55:00` through `15:59:00` one minute apart, plus the terminal `15:59:59`) to
   check whether performance survives trading earlier than the last second before close. Adds an
   execution-latency check (repricing an already-selected bucket 1/5/10s after the signal) to
   confirm the chosen entry timing isn't a zero-latency artifact. Lands on **MLP + quintile
   buckets + `15:58:00` entry** as the final recipe.
6. **`06) test.ipynb`** — final validation of that recipe: identical backtest run on both the
   17-day train period (walk-forward retrained) and the 4-day held-out test period (single model
   frozen after training once on all of train), with full P&L/Sharpe/t-stat/hit-rate stats plus
   stress tests (removing vs. winsorizing top winners, execution lag, bootstrap Sharpe CI, max
   drawdown, sub-period stability). Also where the MLP training loop's eval-fold leakage — present
   in `05)`'s copy — gets fixed, so this notebook's numbers are the ones to trust.

## Repository layout

```
01) data_exploration.ipynb          EDA, data quality, lookahead check
02) features_and_target.ipynb       Feature engineering, target, train/test split
03) models_and_feature_selection.ipynb   CV, elastic net, SHAP/SAGE, final feature set
04) train_wandb.ipynb               W&B hyperparameter sweeps per model family
05) backtest.ipynb                  Long/short backtest, walk-forward, entry-timing +
                                     execution-lag robustness, picks the final recipe
06) test.ipynb                      Final recipe validated on train + held-out test
best_hyperparameters.json           Best config per model family, from 04)
data/                               Raw + processed data (mostly gitignored, see below)
docs/                               Take-home brief + 01)'s write-up (data_exploration_report.md)
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
