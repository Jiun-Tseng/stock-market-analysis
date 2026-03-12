#Syndicate Offerings NLP Return Predictor
#Trains a TF-IDF + Ridge model on pre-market offering news to predict
#intraday stock returns, then simulates a threshold-based long/short portfolio.

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridges
from sklearn.metrics import mean_squared_error

# CONFIG
SEED         = 42
DATA_PATH    = "/Users/jjtseng/Documents/syndicate-data"
OUT          = os.path.join(DATA_PATH, "analysis_output")
PRICE_FILE   = "temp_prices_2021_2024_anon.tsv"
TRAIN_CUTOFF = "2024-01-01"
BUY_THRESH   = 0.01
SELL_THRESH  = -0.01
MAX_RETURN   = 5.0
OFFSET_DAYS  = 5

np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

NEWS_FILES = [
    f"temp_offerings_{y}_anon.tsv" for y in range(2021, 2025)
    if os.path.exists(os.path.join(DATA_PATH, f"temp_offerings_{y}_anon.tsv"))
]

# LOAD
news   = pd.concat([pd.read_csv(os.path.join(DATA_PATH, f), sep="\t")
                    for f in NEWS_FILES], ignore_index=True)
prices = pd.read_csv(os.path.join(DATA_PATH, PRICE_FILE), sep="\t")

news["timestamp"] = pd.to_datetime(news["timestamp"])
news["date"]      = news["timestamp"].dt.date
news              = news[news["timestamp"].dt.time < pd.to_datetime("09:30").time()]
prices["date"]    = pd.to_datetime(prices["date"]).dt.date

# CLEAN PRICES
n_raw            = len(prices)
prices           = prices[(prices["open"] > 0) & (prices["close"] > 0)]
prices["return"] = (prices["close"] - prices["open"]) / prices["open"]
prices           = prices[prices["return"].between(-0.95, MAX_RETURN)]
print(f"[prices]  {n_raw:,} raw -> {len(prices):,} clean  ({n_raw - len(prices):,} dropped)")

# MERGE  (same-day + up to OFFSET_DAYS forward)
same_day  = news.merge(prices, on=["symbol", "date"], how="inner")
unmatched = news[~news.index.isin(same_day.index)].copy()
batches   = [same_day]
print(f"[merge]   same-day: {len(same_day):,} rows")

for d in range(1, OFFSET_DAYS + 1):
    if unmatched.empty:
        break
    unmatched["_shifted"] = (pd.to_datetime(unmatched["date"]) +
                              pd.Timedelta(days=d)).dt.date
    m = unmatched.merge(prices, left_on=["symbol", "_shifted"],
                        right_on=["symbol", "date"],
                        how="inner", suffixes=("_news", ""))
    m["date"] = m["date_news"]
    m.drop(columns=["date_news", "_shifted"], errors="ignore", inplace=True)
    batches.append(m)
    unmatched = unmatched[~unmatched.index.isin(m.index)]
    print(f"[merge]   +{d}d offset: {len(m):,} rows")

data = pd.concat(batches, ignore_index=True)
print(f"[merge]   total: {len(data):,}  |  unmatched (cancelled offerings): {len(news) - len(data):,}\n")

# FEATURES
body_col     = "article" if "article" in data.columns else "body"
data["text"] = (data["headline"] + " " + data[body_col]).apply(
    lambda t: re.sub(r"\s+", " ",
              re.sub(r"[^a-zA-Z ]", " ",
              re.sub(r"http\S+", "", str(t).lower()))).strip()
)
data["date"] = pd.to_datetime(data["date"])

train = data[data["date"] <  TRAIN_CUTOFF].copy()
test  = data[data["date"] >= TRAIN_CUTOFF].copy()


# MODELS
vec   = TfidfVectorizer(max_features=5000, stop_words="english")
model = Ridge(alpha=1.0).fit(vec.fit_transform(train["text"]), train["return"])
test["predicted_return"] = model.predict(vec.transform(test["text"]))

vec_h   = TfidfVectorizer(max_features=3000, stop_words="english")
model_h = Ridge(alpha=1.0).fit(vec_h.fit_transform(train["headline"]), train["return"])
dir_acc_head = np.mean(
    np.sign(model_h.predict(vec_h.transform(test["headline"]))) == np.sign(test["return"])
)

# EVALUATE
y_true  = test["return"].values
y_pred  = test["predicted_return"].values
mse     = mean_squared_error(y_true, y_pred)
corr    = np.corrcoef(y_true, y_pred)[0, 1]
dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

print(f"[model]   full-text -- MSE: {mse:.4f} | Corr: {corr:.4f} | Dir Acc: {dir_acc:.4f}")
print(f"[model]   headline  -- Dir Acc: {dir_acc_head:.4f}\n")

# SIGNALS + SYMBOL RANKINGS
test["signal"] = np.where(test["predicted_return"] > BUY_THRESH,  "buy",
                 np.where(test["predicted_return"] < SELL_THRESH, "sell", "hold"))
test[["date", "symbol", "predicted_return", "signal"]].to_csv(
    os.path.join(OUT, "predicted_signals.csv"), index=False)

sym_preds = test.groupby("symbol")["predicted_return"].mean().sort_values(ascending=False)
sym_preds.head(20).to_csv(os.path.join(OUT, "top_symbols.csv"))
sym_preds.tail(20).to_csv(os.path.join(OUT, "bottom_symbols.csv"))

# PORTFOLIO SIMULATION
daily = (test.groupby(["symbol", "date"])
             .agg(predicted_return=("predicted_return", "mean"),
                  actual_return=("return", "mean"))
             .reset_index()
             .sort_values("date"))

def run_portfolio(df, threshold=0.0):
    def day_pnl(g):
        longs  = g[g["predicted_return"] >  threshold]["actual_return"]
        shorts = g[g["predicted_return"] < -threshold]["actual_return"]
        if longs.empty and shorts.empty:
            return 0.0
        lr = longs.mean()   if not longs.empty  else 0.0
        sr = -shorts.mean() if not shorts.empty else 0.0
        return (0.5 * lr + 0.5 * sr) if (not longs.empty and not shorts.empty) else lr + sr
    pnl = df.groupby("date").apply(day_pnl)
    return pnl, (1 + pnl).cumprod() - 1

equal_daily         = daily.groupby("date")["actual_return"].mean()
cum_equal           = (1 + equal_daily).cumprod() - 1
pnl_any,  cum_any   = run_portfolio(daily, threshold=0.0)
pnl_conf, cum_conf  = run_portfolio(daily, threshold=0.01)

# CHARTS
# 1 -- Cumulative returns
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(cum_any.index,   cum_any.values,   lw=2,   color="steelblue",  label="Long/Short any signal")
ax.plot(cum_conf.index,  cum_conf.values,  lw=2,   color="darkorange", label="Long/Short high confidence (>1%)")
ax.plot(cum_equal.index, cum_equal.values, lw=1.5, color="grey", ls="--", label="Equal-weight benchmark")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_title("Cumulative Returns: Long/Short Portfolio vs Equal Weight Benchmark", fontsize=11)
ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cumulative_portfolio.png"), dpi=150)
plt.close()

# 2 -- Scatter: predicted vs actual
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(daily["predicted_return"], daily["actual_return"],
           alpha=0.4, s=20, color="steelblue")
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_title(f"Predicted vs Actual Returns\ncorr = {corr:.3f}  |  dir acc = {dir_acc:.3f}")
ax.set_xlabel("Predicted Return"); ax.set_ylabel("Actual Return")
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "scatter_pred_vs_actual.png"), dpi=150)
plt.close()

# STATISTICAL SUMMARY
summary = {
    # Data
    "total_rows":               len(data),
    "distinct_symbols":         data["symbol"].nunique(),
    "distinct_dates":           data["date"].nunique(),
    "date_min":                 str(data["date"].min().date()),
    "date_max":                 str(data["date"].max().date()),
    "train_rows":               len(train),
    "test_rows":                len(test),
    "avg_articles_per_symbol":  round(data.groupby("symbol").size().mean(), 2),
    "avg_symbols_per_date":     round(data.groupby("date")["symbol"].nunique().mean(), 2),
    # Prices
    "price_rows_raw":           n_raw,
    "price_rows_clean":         len(prices),
    "price_mean_return":        round(prices["return"].mean(), 4),
    "price_std_return":         round(prices["return"].std(), 4),
    # Model
    "model_mse":                round(float(mse), 4),
    "model_corr":               round(float(corr), 4),
    "model_dir_acc":            round(float(dir_acc), 4),
    "headline_model_dir_acc":   round(float(dir_acc_head), 4),
    # Signals
    "signal_buy_count":         int((test["signal"] == "buy").sum()),
    "signal_sell_count":        int((test["signal"] == "sell").sum()),
    "signal_hold_count":        int((test["signal"] == "hold").sum()),
    # Portfolio
    "portfolio_ls_any_final":   round(float(cum_any.iloc[-1]),  4),
    "portfolio_ls_1pct_final":  round(float(cum_conf.iloc[-1]), 4),
    "portfolio_equal_wt_final": round(float(cum_equal.iloc[-1]), 4),
    "active_trade_days":        int((pnl_any != 0).sum()),
    # Test returns
    "test_return_mean":         round(float(daily["actual_return"].mean()), 4),
    "test_return_std":          round(float(daily["actual_return"].std()),  4),
    "test_return_min":          round(float(daily["actual_return"].min()),  4),
    "test_return_max":          round(float(daily["actual_return"].max()),  4),
}

pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(
    os.path.join(OUT, "statistical_summary.csv"), index=False)

sections = {
    "Data":         ["total_rows","distinct_symbols","distinct_dates","date_min","date_max",
                     "train_rows","test_rows","avg_articles_per_symbol","avg_symbols_per_date"],
    "Prices":       ["price_rows_raw","price_rows_clean","price_mean_return","price_std_return"],
    "Model":        ["model_mse","model_corr","model_dir_acc","headline_model_dir_acc"],
    "Signals":      ["signal_buy_count","signal_sell_count","signal_hold_count"],
    "Portfolio":    ["portfolio_ls_any_final","portfolio_ls_1pct_final",
                     "portfolio_equal_wt_final","active_trade_days"],
    "Test returns": ["test_return_mean","test_return_std","test_return_min","test_return_max"],
}
pad = max(len(k) for k in summary)
print("\n" + "=" * 55)
print("  STATISTICAL SUMMARY")
print("=" * 55)
for section, keys in sections.items():
    print(f"\n  {section}")
    for k in keys:
        print(f"    {k:<{pad}}  {summary[k]}")
print("=" * 55)

print(f"\nOutputs saved to: {OUT}")
print("  predicted_signals.csv      top_symbols.csv     bottom_symbols.csv")
print("  statistical_summary.csv")
print("  cumulative_portfolio.png   scatter_pred_vs_actual.png")