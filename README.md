# Stock Market Analysis

Predicting intraday (open-to-close) stock returns from pre-market syndicate news using TF-IDF features and Ridge regression. Includes preprocessing of anonymized offering news (2021–2024), model training, evaluation, and portfolio simulation.

**Author:** Jiun Tseng  
**Method:** TF-IDF + Ridge Regression  
**Data:** Syndicate Offering News & Prices (2021–2024)  

---

## Key Features
- Pre-market news vectorization (headlines + article body) with TF-IDF
- Ridge regression for return prediction
- Forward-offset merge to align news with first trading day
- Threshold-based long/short portfolio simulation
- Full reproducibility with fixed random seed
- Includes visualizations and statistical summaries

---

## Approach
- **Features:** TF-IDF vectorization  
  - Full-text: max 5,000 features  
  - Headline-only (baseline): max 3,000 features  
- **Model:** Ridge regression (α=1.0)  
- **Train/Test Split:** Train = pre-2024, Test = 2024  
- **Reproducibility:** Vectorizers fit on training data only; numpy/Python seeds fixed (seed=42)

---

## Data & Preprocessing
- **Pre-market filter:** Only news timestamped before 09:30 NYC time, simulating trades at the open  
- **Price cleaning:** Zero/negative prices and extreme return outliers (>500%, <-95%) removed (2,474 rows dropped from 1.13M)  
- **Forward offset merge:** Same-day merge recovers only 1,054 rows (~28%). Forward-looking merge up to 5 calendar days recovers 3,771 rows (~3.6× increase). Remaining ~2,400 unmatched articles are likely cancelled offerings  

---

## Results

| Metric                               | Value        |
|-------------------------------------|-------------|
| Total matched rows                   | 3,771       |
| Train / Test split                   | 2,114 / 1,657 |
| Predicted vs Actual Correlation      | 0.189       |
| Directional Accuracy (Full-text)    | 54.4%       |
| Directional Accuracy (Headline)     | 54.7%       |
| Long/Short Portfolio (any signal)   | +591%       |
| Long/Short Portfolio (|pred| >1%)   | +1,004%     |
| Equal-weight Benchmark               | -87%        |

---

## Key Insights
- **Structural Post-IPO Underperformance:** The universe is dominated by declining returns; equal-weight long-only strategies fail (-87%)  
- **Headlines carry most of the signal:** Headline-only model overperforms compared to full-text analysis (54.7% vs 54.4%). This is because the text has been summarized by the author of said article. 
- **Sparse Universe & Thresholding:** With ~2 symbols/day, top-N selection degenerates. Threshold-based trading (>1% predicted return) significantly improves portfolio outcomes  
- **Alpha Concentrated on Losers:** The model identifies declining stocks more reliably than winners, producing ~3× more sell signals than buy (1,029 vs 324)  

---

## Output Files
- `predicted_signals.csv` — Buy/sell/hold signal per stock/day (test set)  
- `top_symbols.csv` / `bottom_symbols.csv` — Top/bottom 20 symbols by predicted return  
- `statistical_summary.csv` — All key metrics in one place  
- `cumulative_portfolio.png` — Long/short portfolio vs equal-weight benchmark  
- `scatter_pred_vs_actual.png` — Predicted vs actual return scatter  

---
