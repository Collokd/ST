"""Dataset 1: financial news headlines from the GDELT DOC 2.0 API.

The DOC API (https://api.gdeltproject.org/api/v2/doc/doc) returns up to 250
articles per request with a ``seendate`` UTC timestamp, so the date range is
fetched in short windows. GDELT asks clients to keep to about one request every
five seconds. Verify GDELT's terms before acquiring data.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from .config import DEFAULT_UNIVERSE

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_window(query: str, start: pd.Timestamp, end: pd.Timestamp, max_records: int = 250, timeout: int = 60) -> list[dict]:
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    resp = requests.get(GDELT_DOC_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    if not resp.text.strip().startswith("{"):
        # GDELT returns plain-text errors (e.g. rate limiting, bad query).
        raise RuntimeError(f"GDELT error: {resp.text[:200]!r}")
    return resp.json().get("articles", [])


def fetch_gdelt(ticker: str, start: str, end: str, window_days: int = 3, pause: float = 5.5, query: str | None = None) -> pd.DataFrame:
    query = query or DEFAULT_UNIVERSE[ticker]["query"]
    rows: list[dict] = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end) + pd.Timedelta(days=1)
    while cur < stop:
        nxt = min(cur + pd.Timedelta(days=window_days), stop)
        for attempt in range(3):
            try:
                rows.extend(fetch_gdelt_window(query, cur, nxt))
                break
            except (requests.RequestException, RuntimeError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(pause * (attempt + 2))
        cur = nxt
        time.sleep(pause)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "published_utc", "headline", "url", "source"])
    df = pd.DataFrame({
        "published_utc": pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True),
        "headline": df["title"],
        "url": df.get("url"),
        "source": df.get("domain"),
    })
    return clean_news(df, ticker)


def clean_news(df: pd.DataFrame, ticker: str, pattern: str | None = None) -> pd.DataFrame:
    """Drop missing/duplicate headlines and those not clearly about the ticker."""
    df = df.copy()
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    df["headline"] = df["headline"].astype("string").str.strip()
    df = df.dropna(subset=["published_utc", "headline"])
    df = df[df["headline"].str.len() > 0]
    if pattern is None and ticker in DEFAULT_UNIVERSE:
        pattern = DEFAULT_UNIVERSE[ticker]["pattern"]
    if pattern:
        df = df[df["headline"].str.contains(pattern, flags=re.IGNORECASE, regex=True)]
    # Syndicated stories repeat the same headline across outlets on the same day: keep the first.
    norm = df["headline"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")
    day = df["published_utc"].dt.floor("D")
    df = (df.assign(_norm=norm, _day=day).sort_values("published_utc")
          .drop_duplicates(["_norm", "_day"]).drop(columns=["_norm", "_day"]))
    if "url" in df:
        df = df[df["url"].isna() | ~df["url"].duplicated()]
    df = df.drop(columns="ticker", errors="ignore")
    df.insert(0, "ticker", ticker.upper())
    return df.reset_index(drop=True)


def get_news_data(tickers: list[str], start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch (or read cached) headlines for all tickers."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        path = cache_dir / f"news_{t}.csv"
        if path.exists():
            df = clean_news(pd.read_csv(path), t)
        else:
            df = fetch_gdelt(t, start, end)
            df.to_csv(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
