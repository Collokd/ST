# Predicting Short-Term Stock Market Movements Using Financial News Sentiment

CIS 635 · Data Mining and Knowledge Discovery · Fall 2026

**Research question:** Does news sentiment improve next-day directional prediction beyond historical price and volume features?

## Pipeline

| Step | Module | What it does |
|---|---|---|
| News acquisition | `newsstock/news_data.py` | GDELT DOC 2.0 API headlines (UTC timestamps), fetched in short windows; drops missing headlines, same-day syndicated duplicates and headlines that don't match the company's name pattern |
| Market acquisition | `newsstock/market_data.py` | Stooq daily OHLCV CSV; drops duplicate dates, missing closes and invalid prices |
| Session alignment | `newsstock/sessions.py` | Maps each headline to the first session whose 16:00 ET close follows publication (after-close, weekend and holiday news roll forward) |
| Sentiment | `newsstock/sentiment.py` | FinBERT (`ProsusAI/finbert`) positive/negative/neutral probabilities, aggregated per ticker/session (means, shares, net tone, news count) |
| Features & label | `newsstock/features.py` | Lagged returns, 5/20-day volatility, volume change and z-score, high-low range, SMA ratio; label = next session's close above today's close |
| Models | `newsstock/modeling.py` | Logistic regression (baseline), decision tree, random forest, SVM; chronological train/val/test split with a 1-day gap, tuning on validation, refit on train+val, one held-out test evaluation; scaling fit on training rows only |
| Reporting | `newsstock/reporting.py` | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, confusion matrices, market-only vs market+sentiment deltas, EDA plots |
| Clustering | `newsstock/clustering.py` | Exploratory k-means: k chosen by silhouette score, clusters compared to the up/down label with a cross-tab |

Every feature for session *t* uses only information available at *t*'s close, so there is no look-ahead leakage (`tests/test_pipeline.py` checks this).

## Running

**Google Colab (main deliverable):** `notebooks/stock_news_sentiment.ipynb` is self-contained and laid out like the final report (introduction, data, preprocessing and integration, methodology, results, discussion, conclusion, references). Open it at
https://colab.research.google.com/github/Collokd/ST/blob/claude/focused-fermat-ayoya3/notebooks/stock_news_sentiment.ipynb
(or upload it in Colab via *File → Upload notebook*), set `DATA_SOURCE`, and choose *Runtime → Run all*. It doesn't need anything else from this repo.

**Command line:** the `newsstock` package runs the same steps as a script and is what the tests check.

```bash
pip install -e ".[finbert,dev]"

# Quick offline check on synthetic data (results are not meaningful for real markets)
python -m newsstock.pipeline --source synthetic --no-finbert

# Real data: GDELT + Stooq, cached to data/raw/
python -m newsstock.pipeline --source online --tickers AAPL MSFT NVDA --start 2023-01-01 --end 2024-12-31 --cluster

# Pre-downloaded files: data/raw/market_<TICKER>.csv (Date,Open,High,Low,Close,Volume)
#                       data/raw/news_<TICKER>.csv   (published_utc,headline[,url,source])
python -m newsstock.pipeline --source csv

pytest
```

Outputs go to `outputs/`: `test_metrics.csv`, `market_vs_sentiment.csv`, `confusion_matrices.png`, `accuracy_comparison.png`, `f1_comparison.png`, `eda.png` (and `kmeans_profile.csv` with `--cluster`). Intermediate data goes to `data/processed/`.

## Notes on data sources

- **GDELT** (https://www.gdeltproject.org/): the DOC API covers 2017 onward, returns at most 250 articles per request, and asks for about one request every 5 seconds, so a multi-year pull for several tickers takes a while; results are cached per ticker.
- **Stooq** (https://stooq.com/): if the CSV endpoint asks for an API key, pass `--stooq-api-key`, or download the CSVs manually and use `--source csv`.
- Verify both sources' terms of use before acquiring data. The ticker universe and search terms are in `newsstock/config.py`.
- If FinBERT can't be loaded, the code falls back to a small finance word list and prints a warning; don't report those results as FinBERT results.
