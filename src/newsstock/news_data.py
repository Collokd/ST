"""Dataset 1: news headlines from the GDELT DOC 2.0 API (https://www.gdeltproject.org/).

Each GDELT request returns at most 250 articles, so we ask for a few days at
a time. GDELT asks users to wait about 5 seconds between requests, so pulling
a few years of news for several companies takes a while. That's why the
results are saved to a CSV file after the first download.
"""

import re
import time

import pandas as pd
import requests

from .config import COMPANIES

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def request_gdelt(query, start, end):
    """Ask GDELT for English articles matching `query` between two timestamps."""
    params = {
        "query": query + " sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    response = requests.get(GDELT_URL, params=params, timeout=60)
    response.raise_for_status()

    # When something goes wrong (too many requests, bad query), GDELT replies
    # with plain text instead of JSON.
    if not response.text.strip().startswith("{"):
        raise RuntimeError("GDELT error: " + response.text[:200])
    return response.json().get("articles", [])


def download_headlines(ticker, start, end, days_per_request=3, wait_seconds=5.5):
    query = COMPANIES[ticker]["query"]
    articles = []

    window_start = pd.Timestamp(start)
    last_day = pd.Timestamp(end) + pd.Timedelta(days=1)

    while window_start < last_day:
        window_end = min(window_start + pd.Timedelta(days=days_per_request), last_day)

        # Try each window up to 3 times, waiting a bit longer after each failure.
        for attempt in range(3):
            try:
                articles += request_gdelt(query, window_start, window_end)
                break
            except (requests.RequestException, RuntimeError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(wait_seconds * (attempt + 2))

        window_start = window_end
        time.sleep(wait_seconds)

    if not articles:
        return pd.DataFrame(columns=["ticker", "published_utc", "headline", "url", "source"])

    raw = pd.DataFrame(articles)
    headlines = pd.DataFrame({
        "published_utc": pd.to_datetime(raw["seendate"], format="%Y%m%dT%H%M%SZ", utc=True),
        "headline": raw["title"],
        "url": raw.get("url"),
        "source": raw.get("domain"),
    })
    return clean_headlines(headlines, ticker)


def clean_headlines(headlines, ticker, pattern=None):
    """Remove empty, duplicate and off-topic headlines."""
    headlines = headlines.copy()
    headlines["published_utc"] = pd.to_datetime(headlines["published_utc"], utc=True)
    headlines["headline"] = headlines["headline"].astype("string").str.strip()

    # 1. Drop rows without a timestamp or a headline.
    headlines = headlines.dropna(subset=["published_utc", "headline"])
    headlines = headlines[headlines["headline"].str.len() > 0]

    # 2. Keep only headlines that actually mention the company.
    if pattern is None and ticker in COMPANIES:
        pattern = COMPANIES[ticker]["pattern"]
    if pattern:
        mentions_company = headlines["headline"].str.contains(pattern, flags=re.IGNORECASE, regex=True)
        headlines = headlines[mentions_company]

    # 3. The same story is often reposted by many websites on the same day.
    #    We compare headlines in lowercase without punctuation and keep the
    #    earliest copy per day.
    headlines["simple_text"] = (headlines["headline"].str.lower()
                                .str.replace(r"[^a-z0-9 ]", "", regex=True)
                                .str.split().str.join(" "))
    headlines["day"] = headlines["published_utc"].dt.floor("D")
    headlines = headlines.sort_values("published_utc")
    headlines = headlines.drop_duplicates(subset=["simple_text", "day"])
    headlines = headlines.drop(columns=["simple_text", "day"])

    # 4. Same article URL twice? Keep one.
    if "url" in headlines:
        headlines = headlines[headlines["url"].isna() | ~headlines["url"].duplicated()]

    headlines = headlines.drop(columns="ticker", errors="ignore")
    headlines.insert(0, "ticker", ticker.upper())
    return headlines.reset_index(drop=True)


def get_news_data(tickers, start, end, cache_dir):
    """Get headlines for every ticker. Files are saved so we only download once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_headlines = []

    for ticker in tickers:
        saved_file = cache_dir / f"news_{ticker}.csv"
        if saved_file.exists():
            headlines = clean_headlines(pd.read_csv(saved_file), ticker)
        else:
            headlines = download_headlines(ticker, start, end)
            headlines.to_csv(saved_file, index=False)
        all_headlines.append(headlines)

    return pd.concat(all_headlines, ignore_index=True)
