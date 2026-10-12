"""Dataset 2: daily stock prices and volume from Stooq (https://stooq.com/).

Stooq lets you download a ticker's daily history as a CSV file. U.S. tickers
need a ".us" at the end, e.g. "aapl.us". If the download doesn't work (Stooq
sometimes asks for an API key), you can download the CSV files by hand and
read them with load_market_csv().
"""

import io

import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/"
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def download_from_stooq(ticker, start, end, api_key=None):
    params = {
        "s": ticker.lower() + ".us",
        "i": "d",  # daily data
        "d1": pd.Timestamp(start).strftime("%Y%m%d"),
        "d2": pd.Timestamp(end).strftime("%Y%m%d"),
    }
    if api_key:
        params["apikey"] = api_key

    response = requests.get(STOOQ_URL, params=params, timeout=30)
    response.raise_for_status()

    # A good response is a CSV that starts with the "Date" header.
    # Anything else is usually an error message or a captcha page.
    if not response.text.lower().startswith("date"):
        raise RuntimeError(f"Stooq didn't return a CSV for {ticker}: {response.text[:200]}")

    prices = pd.read_csv(io.StringIO(response.text))
    return clean_prices(prices, ticker)


def load_market_csv(path, ticker):
    return clean_prices(pd.read_csv(path), ticker)


def clean_prices(prices, ticker):
    """Tidy up the raw price table and remove rows we can't use."""
    prices = prices.rename(columns=lambda name: name.strip().lower())

    missing = {"date", *PRICE_COLUMNS} - set(prices.columns)
    if missing:
        raise ValueError(f"{ticker}: the price file is missing these columns: {sorted(missing)}")

    prices = prices[["date"] + PRICE_COLUMNS].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    for col in PRICE_COLUMNS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    # Drop days with no closing price, repeated dates, and impossible values.
    prices = prices.dropna(subset=["close"])
    prices = prices.drop_duplicates(subset="date", keep="last")
    prices = prices[prices["close"] > 0]
    prices = prices[prices["volume"].fillna(0) >= 0]

    prices.insert(0, "ticker", ticker.upper())
    return prices.sort_values("date").reset_index(drop=True)


def get_market_data(tickers, start, end, cache_dir, api_key=None):
    """Get prices for every ticker. Files are saved so we only download once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_prices = []

    for ticker in tickers:
        saved_file = cache_dir / f"market_{ticker}.csv"
        if saved_file.exists():
            prices = load_market_csv(saved_file, ticker)
        else:
            prices = download_from_stooq(ticker, start, end, api_key)
            prices.drop(columns="ticker").to_csv(saved_file, index=False)

        in_range = (prices["date"] >= start) & (prices["date"] <= end)
        all_prices.append(prices[in_range])

    return pd.concat(all_prices, ignore_index=True)
