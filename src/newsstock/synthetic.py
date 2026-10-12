"""Fake (synthetic) prices and headlines for testing the code without internet.

We deliberately build in a small link between today's news tone and
tomorrow's return, so we know there is a signal for the models to find.
This only checks that the code works - results on this data say nothing
about the real stock market.
"""

import numpy as np
import pandas as pd

from .config import COMPANIES

COMPANY_NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "Nvidia"}

POSITIVE_HEADLINES = [
    "{name} shares surge after earnings beat", "{name} gains on strong growth outlook",
    "Analysts upgrade {name} on record profit", "{name} rallies as demand jumps",
    "{name} raises guidance, stock rises",
]
NEGATIVE_HEADLINES = [
    "{name} shares plunge after earnings miss", "{name} falls on weak outlook",
    "Analysts downgrade {name} amid probe", "{name} slumps as losses widen",
    "{name} cuts forecast, stock drops",
]
NEUTRAL_HEADLINES = [
    "{name} to hold annual shareholder meeting", "{name} announces new product event date",
    "{name} names new board member", "What to watch for {name} this week", "{name} files quarterly report",
]


def fake_trading_days(start, end):
    """Weekdays, minus New Year's Day, July 4th and Christmas."""
    days = pd.bdate_range(start, end)
    holidays = [d for d in days if (d.month, d.day) in [(1, 1), (7, 4), (12, 25)]]
    return days.difference(pd.DatetimeIndex(holidays))


def make_synthetic(tickers=None, start="2022-01-01", end="2024-12-31", signal=0.006, seed=0):
    rng = np.random.default_rng(seed)
    tickers = tickers or list(COMPANIES)
    days = fake_trading_days(start, end)
    n_days = len(days)

    all_prices = []
    all_news = []

    for ticker in tickers:
        name = COMPANY_NAMES.get(ticker, ticker)

        # The hidden "news tone" of each day (positive = good news).
        tone = rng.normal(0, 1, n_days)

        # Daily returns are random noise, plus a small push from YESTERDAY's tone.
        returns = rng.normal(0.0003, 0.018, n_days)
        returns[1:] += signal * tone[:-1]

        close = 100 * np.exp(np.cumsum(returns))
        open_price = close * np.exp(rng.normal(0, 0.005, n_days))
        high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.008, n_days)))
        low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.008, n_days)))
        volume = rng.lognormal(16, 0.3, n_days) * (1 + 0.3 * np.abs(tone))

        all_prices.append(pd.DataFrame({
            "ticker": ticker, "date": days, "open": open_price, "high": high,
            "low": low, "close": close, "volume": volume.round(),
        }))

        # Headlines for a trading day are published between the previous close and this day's close.
        previous_day = days[0] - pd.Timedelta(days=1)
        for day, day_tone in zip(days, tone):
            window_start = (previous_day + pd.Timedelta(hours=16)).tz_localize("America/New_York")
            window_end = (day + pd.Timedelta(hours=16)).tz_localize("America/New_York")
            previous_day = day

            n_headlines = rng.poisson(3 * (1 + abs(day_tone)))  # bigger news days get more headlines
            window_seconds = (window_end - window_start).total_seconds() - 60
            chance_positive = 1 / (1 + np.exp(-1.5 * day_tone))

            for _ in range(n_headlines):
                if rng.random() < 0.35:
                    options = NEUTRAL_HEADLINES
                elif rng.random() < chance_positive:
                    options = POSITIVE_HEADLINES
                else:
                    options = NEGATIVE_HEADLINES
                published = window_start + pd.Timedelta(seconds=float(rng.uniform(0, window_seconds)))
                all_news.append({
                    "ticker": ticker,
                    "published_utc": published.tz_convert("UTC"),
                    "headline": rng.choice(options).format(name=name),
                    "url": None,
                    "source": "synthetic",
                })

    news = pd.DataFrame(all_news)
    # Copy 5% of the headlines to imitate the same story being reposted, so the cleaning step has work to do.
    news = pd.concat([news, news.sample(frac=0.05, random_state=seed)], ignore_index=True)
    return pd.concat(all_prices, ignore_index=True), news
