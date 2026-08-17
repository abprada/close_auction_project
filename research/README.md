# Research — Closing Auction Alpha

Downloaded for background reading before feature engineering. Not part of the take-home deliverable.

## Downloaded

- **`01_esann2024_predicting_closing_cross_nasdaq.pdf`** — Predicting the Closing Cross Auction Results at the NASDAQ Stock Exchange (ESANN 2024). Same Optiver/Nasdaq imbalance data as ours (imbalance size, far/near price, 1s snapshots over the last 10 min). Deep learning vs. SVR.
- **`02_imperial_predicting_us_returns_closing_imbalance.pdf`** — Predicting US stock returns using closing auction imbalance data (Imperial College thesis). S&P 500/Nasdaq names, imbalance predicting forward returns, not just the close price.
- **`03_arxiv_dynamical_regularities_auctions.pdf`** — Dynamical regularities of US equities opening and closing auctions (arXiv 1802.01921). Statistical structure of imbalance/volume at the open and close.
- **`04_arxiv_heavy_tailed_closing_auctions.pdf`** — Heavy tailed distributions in closing auctions (arXiv 2012.10145). Distributional shape of closing-auction quantities — relevant to whether raw values need log/winsorizing before modeling.

## Not downloadable — access blocked or paywalled

- **Jegadeesh & Wu, "Closing Auctions: Nasdaq versus NYSE"** (*Journal of Financial Economics*, 2022) — paywalled on ScienceDirect. Key finding worth knowing without reading the full paper: order imbalance is largest at first dissemination (15:50 ET) and decays steadily into the close; on Nasdaq it tends to clear almost entirely by 16:00. Imbalance contributes real price discovery for small caps, only weak evidence for large caps. https://www.sciencedirect.com/science/article/abs/pii/S0304405X21005092
- **Goyal, Jegadeesh, Wu, "Price Impact: Continuous Trading, Closing Auctions, and Opening Auctions"** (SSRN) — blocked by a Cloudflare bot challenge, not fetchable by a script. https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4692944_code2427035.pdf?abstractid=4300417&mirid=1

## Practitioner resources (not papers, links only)

- **Optiver "Trading at the Close" Kaggle competition** — same Nasdaq imbalance fields as ours, thousands of public solutions. https://www.kaggle.com/competitions/optiver-trading-at-the-close
- **1st place solution writeup**: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- **14th place solution writeup**: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/clash-royale-14th-place-solution-for-the-optiver-t

Recurring pattern across public solutions: gradient boosting (XGBoost/LightGBM) outperformed deep learning; heavy use of lagged features per stock across the pre-close window (not just the last snapshot), rolling stats (mean/std/skew/kurtosis) of price and size, and basis-point price changes implied by the imbalance rather than raw levels.
