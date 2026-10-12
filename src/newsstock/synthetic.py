"""Synthetic market + news data for offline demos and tests.

The generator plants a weak link between headline tone and the *next* day's
return, so the pipeline has a known signal to detect. It is for checking the
code end to end only; results on it say nothing about real markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_UNIVERSE

_NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "Nvidia"}
_POS = ["{n} shares surge after earnings beat", "{n} gains on strong growth outlook", "Analysts upgrade {n} on record profit",
        "{n} rallies as demand jumps", "{n} raises guidance, stock rises"]
_NEG = ["{n} shares plunge after earnings miss", "{n} falls on weak outlook", "Analysts downgrade {n} amid probe",
        "{n} slumps as losses widen", "{n} cuts forecast, stock drops"]
_NEU = ["{n} to hold annual shareholder meeting", "{n} announces new product event date", "{n} names new board member",
        "What to watch for {n} this week", "{n} files quarterly report"]


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    holidays = [d for d in days if (d.month, d.day) in {(1, 1), (7, 4), (12, 25)}]
    return days.difference(pd.DatetimeIndex(holidays))


def make_synthetic(tickers: list[str] | None = None, start: str = "2022-01-01", end: str = "2024-12-31",
                   signal: float = 0.006, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tickers = tickers or list(DEFAULT_UNIVERSE)
    days = trading_days(start, end)
    market, news = [], []
    for t in tickers:
        name = _NAMES.get(t, t)
        tone = rng.normal(0, 1, len(days))  # latent news tone revealed by session t's close
        ret = rng.normal(0.0003, 0.018, len(days))
        ret[1:] += signal * tone[:-1]
        close = 100 * np.exp(np.cumsum(ret))
        open_ = close * np.exp(rng.normal(0, 0.005, len(days)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, len(days))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, len(days))))
        vol = rng.lognormal(16, 0.3, len(days)) * (1 + 0.3 * np.abs(tone))
        market.append(pd.DataFrame({"ticker": t, "date": days, "open": open_, "high": high, "low": low, "close": close, "volume": vol.round()}))

        prev_close = pd.Timestamp(days[0]) - pd.Timedelta(days=1)
        for d, s in zip(days, tone):
            # Headlines for session d fall between the previous close and d's close (16:00 ET).
            window_start = (prev_close + pd.Timedelta(hours=16)).tz_localize("America/New_York")
            window_end = (d + pd.Timedelta(hours=16)).tz_localize("America/New_York")
            prev_close = d
            n = rng.poisson(3 * (1 + abs(s)))
            secs = rng.uniform(0, (window_end - window_start).total_seconds() - 60, n)
            for sec in secs:
                p_pos = 1 / (1 + np.exp(-1.5 * s))
                u = rng.random()
                bank = _NEU if u < 0.35 else (_POS if rng.random() < p_pos else _NEG)
                news.append({"ticker": t, "published_utc": (window_start + pd.Timedelta(seconds=float(sec))).tz_convert("UTC"),
                             "headline": rng.choice(bank).format(n=name), "url": None, "source": "synthetic"})
    news_df = pd.DataFrame(news)
    # Add exact duplicates (syndication) so cleaning is exercised.
    news_df = pd.concat([news_df, news_df.sample(frac=0.05, random_state=seed)], ignore_index=True)
    return pd.concat(market, ignore_index=True), news_df
