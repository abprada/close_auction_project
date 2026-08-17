# Data Exploration Report — Nasdaq Closing Auction Imbalance Data

**Source**: `data/202503_imbalance.tar` — 21 daily `csv.gz` files, one per trading day, 2025-03-03 to 2025-03-31
**Companion notebook**: `01) data_exploration.ipynb`

## 1. Data Inventory

- **4,103,085 rows** across 21 trading days, ~800 MB in memory once loaded
- **562 unique symbols** total, but **exactly 500 symbols every single day** — the universe rotates day to day rather than staying fixed
- **18 columns**: `ts` (UTC), `local_time` (ET, tz-aware string), `symbol`, `side`, `adv`, `ask`, `ask_qty`, `bid`, `bid_qty`, `cross`, `far_price`, `near_price`, `open`, `paired_shares`, `ref_price`, `shares`, plus a derived `date` and `local_clock`
- **Cadence matches the documented spec exactly**: 10-second snapshots from 15:50–15:55 ET, 1-second snapshots from 15:55 ET onward, confirmed directly on sample data rather than assumed
- Two structural quirks in the raw CSV, both handled at load time:
  - The header row repeats `ts` (`ts,...,ts,local_time`) — pandas renames the second to `ts.1` on read rather than erroring, so a naive duplicate-column check finds nothing to drop; dropped explicitly by name instead, after confirming `ts` and `ts.1` are byte-identical across all 21 files
  - `local_time`'s UTC offset changes mid-month at the March 9 DST transition (`-05:00` → `-04:00`), which breaks a naive `pd.to_datetime` on that column (mixed offsets). The wall-clock `HH:MM:SS` is instead extracted via string slicing into `local_clock`, which is DST-safe and sufficient for all close-relative filtering used here.

## 2. Univariate Analysis

- `shares`, `paired_shares`, `adv`, and both size columns (`bid_qty`/`ask_qty`) are heavily right-skewed, log-normal-ish distributions spanning several orders of magnitude across symbols — small- and large-cap names sit on completely different scales.
- `imbalance_ratio` (`shares / (paired_shares + shares)`) has a large spike at 0 — consistent with the `side == 'NONE'` rows where the auction is already balanced at the reference price — and a long right tail for names with genuine unmatched demand.
- `near_price - far_price`, examined only in the 15:55–16:00 window where both fields are live, is concentrated near zero with a moderate spread, i.e. the two theoretical closing prices (auction-only vs. auction+book) mostly agree but diverge meaningfully for a subset of names.

## 3. Data Quality Issues

- **Zero fully duplicated rows, zero negative prices or sizes** — clean on both dimensions. **1,134 rows do show a crossed market** (`bid > ask`) — see Section 5 for where these concentrate.
- `side` includes a fourth category beyond the documented BUY/SELL/NONE: `INSUFFICIENT` (~0.5% of rows), indicating the auction couldn't fully match — carried as its own state rather than collapsed into `NONE`.
- **The ~0.3% null block** in `side`, `shares`, `paired_shares`, `near_price`, `far_price` is not random missingness — it is exactly one coherent phenomenon. All 10,502 affected rows are either the `local_clock == 16:00:00` auction-print row for a symbol-day (10,500 of them — imbalance fields simply stop applying once the auction has executed) or one of two delayed-close stragglers (below). Verified by checking that these five columns are null in the identical row set, not independently.
- **A separate, smaller null block**: `bid`/`ask`/`bid_qty`/`ask_qty` are null in 558 rows (0.01%) — quotes not yet posted in the first minute of the pre-close window, mostly on the two dead-book symbol-days below (`ALTR`, `AZPN`), plus 6 other symbols with a brief delay. Confirmed resolved by 15:59:59 in every case — zero nulls in the final panel.
- **Snapshot-duration check**: median snapshot count per symbol-day is 391 (30 at 10s cadence + ~361 at 1s cadence). A handful of symbol-days run longer — `WBA` on 2025-03-06 continues publishing to **17:45 ET, 105 minutes past close**, with `cross` already known partway through. This means "the last row per symbol-day" is not a safe proxy for "the last pre-close row." A full census of every symbol-day's post-close behavior, including a second delayed case (`AAOI`) and a distinct ETP-related issue, is in Section 5.

## 4. Lookahead Risk Check

This is the sharpest trap in the dataset, and it's a hard mechanical fact rather than a matter of judgment.

- `cross` is `NaN` for every row before `16:00:00` ET, confirmed on a real symbol-day (`TSLQ`, 2025-03-03) down to the exact transition timestamp, then becomes a single constant value for the remainder of the day. It is the auction print itself, delivered in-band in the same feed as the pre-close imbalance data — not a separate historical/label file.
- Consequences for any modeling work:
  - `cross` must never appear in a feature vector. Trivially true pre-close (it's `NaN`), but a naive `.ffill()` per symbol-day would silently manufacture a perfect feature — never forward-fill this column.
  - Rows at or after `local_clock == 16:00:00` must be excluded from feature construction entirely, not just the `cross` column within them — for the delayed-close symbol-days in Section 3, those later rows describe post-auction (or extended-auction) state that a strategy entering after 15:50 and exiting at 16:00 could never have observed.
  - `near_price`/`far_price` are legitimately `0.0` (not missing) before 15:55 ET — confirmed directly: identically zero pre-15:55, never `NaN`, and live/moving from exactly 15:55 onward. This is Nasdaq not disseminating those fields yet, not an artifact to clean.
  - The eventual model panel (built in Section 5) must join on `(symbol, date)` only, using the last snapshot strictly before `16:00:00` — this prevents any later day's `adv`/`open`/etc. from leaking into an earlier day's row.
- **`adv` provenance check**: `adv` (trailing 20-day average traded shares) changes value on 550 of 562 symbols across the month. The 12 symbols where it looks constant all turn out to be names that only appear on a single day in the file — trivially constant, not a static field. For symbols present the full 21 days, `adv` drifts smoothly and incrementally day to day, which is what a genuine rolling calculation recomputed fresh each day looks like, not a snapshot reused across the month. This is consistent with `adv` being point-in-time and safe to use as a feature — not formal proof (a rolling figure could in principle still be computed with hindsight over the full extract), but there is no evidence of leakage, and the drift pattern is the right shape.

## 5. Post-Close Delay Census

The Section 3 finding on `WBA` came from a snapshot-count outlier check that only surfaced the single largest case. A direct census of every symbol-day's last `local_clock` value shows the full picture, across all 10,500 symbol-days:

- **10,428 (99.3%)** run a routine 5-second-cadence tail from 16:00:00 to 16:05:00 with `shares`/`paired_shares` pinned at 0, `side == 'NONE'`, and `cross` already fixed. This is a post-close heartbeat, not new information — it's already excluded by the `local_clock < 16:00:00` rule, so it's not a new risk.
- **70 (0.67%)** stop exactly at 16:00:00 with no tail at all. 65 of these belong to just four tickers — `SPY`, `IWM`, `USO`, `FNGA` — and account for every single day those tickers appear in the file. All four are ETPs primarily traded on NYSE Arca, not Nasdaq, and `cross` is a 0.0 placeholder for them most days (100% for `USO`/`FNGA`, 90% for `IWM`, 62% for `SPY`) rather than a genuine close. **These four tickers should be excluded from the target-prediction task** — `cross` isn't a reliable label for them.
- The remaining 5 occurrences (`ALNY`, `ALTR`, `AZPN`, `NVDD`, `RAA`) are **not noise — one coherent failure mode.** All five show `side == 'INSUFFICIENT'` from the first message at 15:50:00 through the last pre-close snapshot at 15:59:59 — no matchable book ever existed, not just at the close, so no real close was ever established. Two of the five — `ALNY` and `AZPN` — additionally show a crossed book (314 and 222 rows respectively) during that dead window; that's where two of the four crossed-market symbol-days below actually came from. Excluded from the panel on that pre-close feature (`side == 'INSUFFICIENT'` in the last snapshot) rather than on the realized close — the dead book is visible before 16:00:00, so this isn't a hindsight-based exclusion, and it needs the same treatment as the ETPs: excluded, not imputed.
- **2 (0.02%)** are genuine extended delays: `WBA` (2025-03-06, resumes at 17:40 after a ~95-minute gap) and `AAOI` (2025-03-13, resumes at 16:30 after a ~25-minute gap). Both show real, changing `ref_price`/`shares` activity resuming well after 16:00 while `cross` stays fixed at the already-printed value — a delayed reopening auction layered on top of the original close, not corrupted data.
  - `WBA`'s `ref_price` jumps from 10.60 to 11.25, which lines up closely with the real-world announcement that day of Walgreens Boots Alliance's take-private acquisition by Sycamore Partners at $11.45/share cash — very likely the market repricing around the deal terms.
  - `AAOI`'s `ref_price`/`bid` run from 17.60 to 28.00 while `ask` stays frozen at 16.06 the whole time (a stale one-sided quote under an illiquid reopening, not a clean two-sided repricing). **Catalyst confirmed**: Applied Optoelectronics entered a warrant agreement with Amazon that day, granting Amazon the right to purchase up to 7.9 million shares at a $23.6954 strike, with 1.32 million warrant shares vesting immediately — that strike sits squarely inside the 17.60-28.00 range the reopening traded through.
  - Neither changes the handling rules already in place: the last valid pre-16:00:00 snapshot for both symbol-days is unaffected by the later gap.

**The 1,134 crossed-market rows flagged in Section 3** turn out to sit inside exactly these four symbol-days and nowhere else in the entire file: `AAOI` (301 rows, during its reopening), `ALNY` (314 rows, during its failed auction), `AZPN` (222 rows, during its failed auction), `WBA` (297 rows, during its reopening). A crossed book during a volatile reopening or a failed-match auction isn't surprising — it's confined entirely to the days already flagged as anomalous for other reasons.

**Building the panel**: one row per symbol-day — the last pre-close snapshot (features) joined to the realized close (target) — is the panel that carries into feature engineering. Built directly with what this census established: the pre-16:00:00 boundary needs no special handling (`cross` takes exactly one value per symbol-day across the entire file, zero exceptions, so the heartbeat and reopening tails are never reachable). Rows are dropped on the four ETPs, and on `side == 'INSUFFICIENT'` in the last pre-close snapshot (65 + 5 rows) — a feature of the pre-close book, not the realized close, even though it happens to coincide exactly with the five symbol-days whose `close_price` would have come out 0.0. That distinction matters: excluding rows because the target turned out invalid is fine for training, but this filter is stronger — the dead book is observable *before* 16:00:00, so nothing here relies on hindsight. That takes the panel from **10,500 to 10,430 symbol-days**. `WBA` and `AAOI` remain in the panel, each with one valid price, unaffected by their post-close reopening activity. No engineered or derived columns are built here — correlation analysis and feature construction are deliberately deferred to feature engineering. `cross` is dropped before saving (always `NaN` by construction in a pre-close snapshot — `close_price` is the only target column). Saved to `data/panel_clean.csv` (10,430 rows, 19 columns) — small enough to persist as-is rather than rebuilt from the raw tar each time.

## Conclusions — Is the Data Clean?

Structurally, yes: no duplicates, no negative values, cadence matches spec, and the 1,134 crossed-market rows are confined entirely to the four anomalous symbol-days above. But several things that looked like data-quality issues on first pass turned out, once checked, to be mechanical facts about how the feed works rather than defects — that distinction matters, and none of the following is "cleaning" in the sense of dropping bad rows or imputing garbage. It's *handling*.

The panel carried forward for modeling is **10,430 symbol-days** — the original 10,500 minus the 65 rows belonging to the four excluded ETPs and 5 more failed-auction rows with a fake 0.0 close (Section 5).

### Handling rules for all future work in this repo

1. Drop the duplicate `ts` header column on load (already handled in `load_all()`).
2. Treat `near_price`/`far_price` == 0.0 before 15:55 ET as "not yet disseminated," never as a true price. Do not impute or interpolate through it.
3. Never forward-fill or impute `cross` — it is the target, valid only at/after `16:00:00`.
4. Any feature-construction row selection must filter strictly on `local_clock < 16:00:00`, never on "last row per symbol-day." `WBA` and `AAOI` both have delayed/extended auctions that keep publishing well past the nominal close with `cross` already known (Section 5); using the true last row for those would leak.
5. The ~0.3% null block in `side`/`shares`/`paired_shares`/`near_price`/`far_price` is fully explained and disappears automatically once rule 4 is applied — no imputation needed.
6. Normalize before comparing across symbols — raw price and share-count scales span orders of magnitude.
7. The raw price-level columns (`ask`, `bid`, `ref_price`, `open`, `near_price`, `far_price`, `close_price`) are the same stock's price sampled at different points and are likely highly collinear — check before including several unmodified together as features; this EDA doesn't quantify it, that's feature-engineering work.
8. When normalizing or scaling in the feature engineering phase, compute any statistic (z-score, mean, std) only on the training data available at that point in time — never over the full dataset including future dates, which is its own form of lookahead leakage.
9. Exclude `SPY`, `IWM`, `USO`, `FNGA` from the modeling universe — non-Nasdaq-primary-listed ETPs whose `cross` is mostly a 0.0 placeholder, not a real close (Section 5). Also drop any panel row where the last pre-close snapshot shows `side == 'INSUFFICIENT'` — a dead book, visible before 16:00:00 (five more symbol-days, Section 5). Filter on this pre-close feature, not on `close_price == 0.0` directly — the two happen to coincide here, but the feature-based rule is the one that's actually ex-ante-justified.
10. `WBA` (2025-03-06) and `AAOI` (2025-03-13) had genuine delayed-reopening auctions well after 16:00; the last pre-16:00:00 snapshot for both is still valid and safe to use per rule 4 — no special-casing needed beyond what's already in place.

### Open items — not resolved by this EDA

- Target definition (raw close level vs. a move relative to some reference price) — deliberately deferred; no target or derived features exist in this notebook, by design.

### Next step

Feature engineering — not started. Correlation analysis, signed/derived features, and target definition all belong there, informed by the panel and handling rules above.
