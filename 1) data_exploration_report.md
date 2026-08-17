# Data Exploration Report — Nasdaq Closing Auction Imbalance Data

**Source**: `data/202503_imbalance.tar` — 21 daily `csv.gz` files, one per trading day, 2025-03-03 to 2025-03-31
**Companion notebook**: `01) data_exploration.ipynb`

## 1. Data Inventory

- **4,103,085 rows** across 21 trading days, ~1 GB in memory once loaded
- **562 unique symbols** total, but **exactly 500 symbols every single day** — the universe rotates day to day rather than staying fixed
- **18 columns**: `ts` (UTC), `local_time` (ET, tz-aware string), `symbol`, `side`, `adv`, `ask`, `ask_qty`, `bid`, `bid_qty`, `cross`, `far_price`, `near_price`, `open`, `paired_shares`, `ref_price`, `shares`, plus a derived `date` and `local_clock`
- **Cadence matches the documented spec exactly**: 10-second snapshots from 15:50–15:55 ET, 1-second snapshots from 15:55 ET onward, confirmed directly on sample data rather than assumed
- Median snapshot count per symbol-day: 391 (30 at 10s cadence + ~361 at 1s cadence). A small number of symbol-days run longer — see Section 4.
- Two structural quirks in the raw CSV, both handled at load time:
  - The header row repeats `ts` (`ts,...,ts,local_time`) — the duplicate is dropped
  - `local_time`'s UTC offset changes mid-month at the March 9 DST transition (`-05:00` → `-04:00`), which breaks a naive `pd.to_datetime` on that column (mixed offsets). The wall-clock `HH:MM:SS` is instead extracted via string slicing into `local_clock`, which is DST-safe and sufficient for all close-relative filtering used here.

## 2. Univariate Analysis

- `shares`, `paired_shares`, `adv`, and both size columns (`bid_qty`/`ask_qty`) are heavily right-skewed, log-normal-ish distributions spanning several orders of magnitude across symbols — small- and large-cap names sit on completely different scales.
- `imbalance_ratio` (`shares / (paired_shares + shares)`) has a large spike at 0 — consistent with the `side == 'NONE'` rows where the auction is already balanced at the reference price — and a long right tail for names with genuine unmatched demand.
- `near_price - far_price`, examined only in the 15:55–16:00 window where both fields are live, is concentrated near zero with a moderate spread, i.e. the two theoretical closing prices (auction-only vs. auction+book) mostly agree but diverge meaningfully for a subset of names.

## 3. Multivariate Patterns

Correlations were computed against the actual auction outcome, not just among pre-close variables in isolation. This required building the eventual model panel structure — one row per symbol-day, using the last snapshot *before* 16:00:00 ET as features, joined to the realized `cross` price as the outcome (renamed `close_price`) — since correlating raw pre-close columns against a column that's 85% null otherwise wouldn't be meaningful.

- **10,500 of 10,500 symbol-days matched cleanly** (500 symbols × 21 days) — every symbol-day had both a valid pre-close snapshot and a known close.
- Restricted to raw variables only (no engineered ratios or signed features, per instruction — feature engineering is a later, separate step): the price-level columns (`ask`, `bid`, `ref_price`, `open`, `near_price`, `far_price`, `close_price`) are all highly collinear with each other. This is expected and not very informative on its own — it mostly reflects which stock a row belongs to (a \$400 name vs. a \$5 name), not auction dynamics.
- The size columns (`shares`, `paired_shares`, `adv`, `ask_qty`, `bid_qty`) correlate weakly with `close_price`, which also makes sense — raw share counts don't map to price level.
- Neither block says much yet about what predicts the *closing move* specifically, since nothing here is constructed relative to a reference point. That's explicitly deferred to the feature engineering phase.

## 4. Data Quality Issues

- **Zero fully duplicated rows, zero negative prices or sizes, zero crossed markets** (`bid > ask`) — the feed is clean on all three dimensions.
- `side` includes a fourth category beyond the documented BUY/SELL/NONE: `INSUFFICIENT` (~1% of rows), indicating the auction couldn't fully match — carried as its own state rather than collapsed into `NONE`.
- **The ~0.3% null block** in `side`, `shares`, `paired_shares`, `near_price`, `far_price` is not random missingness — it is exactly one coherent phenomenon. All 10,502 affected rows are either the `local_clock == 16:00:00` auction-print row for a symbol-day (10,500 of them — imbalance fields simply stop applying once the auction has executed) or one of two delayed-close stragglers (below). Verified by checking that these five columns are null in the identical row set, not independently.
- **Delayed/extended auctions**: at least two symbol-days continue publishing well past the nominal 16:00:00 close. `WBA` on 2025-03-06 runs to **17:45 ET — 105 minutes past close** — with `cross` already known partway through, and there's one further case at 16:35:00. This is very likely a real closing-auction delay (individual names can be held up by LULD or imbalance-driven halts), not corrupted data, but it means "the last row per symbol-day" is not a safe proxy for "the last pre-close row" — see Section 5.

## 5. Lookahead Risk Check

This is the sharpest trap in the dataset, and it's a hard mechanical fact rather than a matter of judgment.

- `cross` is `NaN` for every row before `16:00:00` ET, confirmed on a real symbol-day (`TSLQ`, 2025-03-03) down to the exact transition timestamp, then becomes a single constant value for the remainder of the day. It is the auction print itself, delivered in-band in the same feed as the pre-close imbalance data — not a separate historical/label file.
- Consequences for any modeling work:
  - `cross` must never appear in a feature vector. Trivially true pre-close (it's `NaN`), but a naive `.ffill()` per symbol-day would silently manufacture a perfect feature — never forward-fill this column.
  - Rows at or after `local_clock == 16:00:00` must be excluded from feature construction entirely, not just the `cross` column within them — for the delayed-close symbol-days in Section 4, those later rows describe post-auction (or extended-auction) state that a strategy entering after 15:50 and exiting at 16:00 could never have observed.
  - `near_price`/`far_price` are legitimately `0.0` (not missing) before 15:55 ET — confirmed directly: identically zero pre-15:55, never `NaN`, and live/moving from exactly 15:55 onward. This is Nasdaq not disseminating those fields yet, not an artifact to clean.
  - The eventual model panel must join on `(symbol, date)` only, using the last snapshot strictly before `16:00:00` — this prevents any later day's `adv`/`open`/etc. from leaking into an earlier day's row.
  - Not verifiable from this file alone: whether `adv` is genuinely point-in-time or could be restated after the fact. Flagged as an open item rather than assumed either way.

## Conclusions — Is the Data Clean?

Structurally, yes: no duplicates, no negative values, no crossed markets, cadence matches spec. But several things that looked like data-quality issues on first pass turned out, once checked, to be mechanical facts about how the feed works rather than defects — that distinction matters, and none of the following is "cleaning" in the sense of dropping bad rows or imputing garbage. It's *handling*.

### Handling rules for all future work in this repo

1. Drop the duplicate `ts` header column on load (already handled in `load_all()`).
2. Treat `near_price`/`far_price` == 0.0 before 15:55 ET as "not yet disseminated," never as a true price. Do not impute or interpolate through it.
3. Never forward-fill or impute `cross` — it is the target, valid only at/after `16:00:00`.
4. Any feature-construction row selection must filter strictly on `local_clock < 16:00:00`, never on "last row per symbol-day."
5. The ~0.3% null block in `side`/`shares`/`paired_shares`/`near_price`/`far_price` is fully explained and disappears automatically once rule 4 is applied — no imputation needed.
6. Decide explicitly on the two delayed-close symbol-days before modeling: drop as a different microstructure regime, or keep with a flag feature. Currently undecided.
7. Normalize before comparing across symbols — raw price and share-count scales span orders of magnitude.
8. Remember the price-level columns are highly collinear; including several raw, unmodified, together would be redundant.
9. When normalizing or scaling in the feature engineering phase, compute any statistic (z-score, mean, std) only on the training data available at that point in time — never over the full dataset including future dates, which is its own form of lookahead leakage.

### Open items — not resolved by this EDA

- Whether `adv` is point-in-time or restatable — a question for the data provider, not answerable from this file.
- Whether to exclude or flag the two delayed-close symbol-days.
- Target definition (raw close level vs. a move relative to some reference price) — deliberately deferred; no target or derived features exist yet in the notebook.

### Next step

Feature engineering, informed directly by the findings above — not started yet.
