"""Dataset 2: daily OHLCV stock data from Stooq (https://stooq.com/).

Stooq serves daily history as CSV at ``/q/d/l/?s=<symbol>&i=d``; U.S. tickers
use the ``.us`` suffix. Verify Stooq's terms of use before acquiring data. If
the endpoint is unavailable (Stooq may require an API key or captcha), download
the CSVs manually and load them with :func:`load_market_csv`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/"
OHLCV = ["open", "high", "low", "close", "volume"]


def fetch_stooq(ticker: str, start: str, end: str, api_key: str | None = None, timeout: int = 30) -> pd.DataFrame:
    params = {
        "s": f"{ticker.lower()}.us",
        "i": "d",
        "d1": pd.Timestamp(start).strftime("%Y%m%d"),
        "d2": pd.Timestamp(end).strftime("%Y%m%d"),
    }
    if api_key:
        params["apikey"] = api_key
    resp = requests.get(STOOQ_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if not text.lower().startswith("date"):
        raise RuntimeError(f"Unexpected Stooq response for {ticker}: {text[:200]!r}")
    return clean_market(pd.read_csv(io.StringIO(text)), ticker)


def load_market_csv(path: str | Path, ticker: str) -> pd.DataFrame:
    return clean_market(pd.read_csv(path), ticker)


def clean_market(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise columns, drop duplicate/missing rows and invalid prices."""
    df = df.rename(columns=lambda c: c.strip().lower())
    missing = {"date", *OHLCV} - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: market data missing columns {sorted(missing)}")
    df = df[["date", *OHLCV]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).drop_duplicates(subset="date", keep="last")
    df = df[(df["close"] > 0) & (df["volume"].fillna(0) >= 0)]
    df.insert(0, "ticker", ticker.upper())
    return df.sort_values("date").reset_index(drop=True)


def get_market_data(tickers: list[str], start: str, end: str, cache_dir: Path, api_key: str | None = None) -> pd.DataFrame:
    """Fetch (or read cached) market data for all tickers."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        path = cache_dir / f"market_{t}.csv"
        if path.exists():
            df = load_market_csv(path, t)
        else:
            df = fetch_stooq(t, start, end, api_key=api_key)
            df.drop(columns="ticker").to_csv(path, index=False)
        frames.append(df[(df["date"] >= start) & (df["date"] <= end)])
    return pd.concat(frames, ignore_index=True)
