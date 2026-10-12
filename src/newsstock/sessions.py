"""Matching each headline to the trading day it can affect.

This is the step that joins our two datasets, so we have to be careful about
time. We make our prediction at the market close (4:00 pm New York time) of
day t, and we predict whether the stock closes higher on day t+1. So a
headline can only be used for day t if it came out before day t's close:

  - published before 4:00 pm on a trading day  -> that same day
  - published after 4:00 pm                     -> the next trading day
  - published on a weekend or holiday           -> the next trading day

We get the list of trading days from the price data itself, so weekends and
market holidays are handled automatically (they just aren't in the list).
"""

import numpy as np
import pandas as pd

from .config import MARKET_CLOSE_HOUR, MARKET_TIMEZONE


def map_to_sessions(headlines, prices):
    """Add a `session` column with the trading day each headline belongs to."""
    matched = []

    for ticker, ticker_news in headlines.groupby("ticker", sort=False):
        trading_days = np.sort(prices.loc[prices["ticker"] == ticker, "date"].unique())
        if len(trading_days) == 0:
            continue

        # Convert from UTC to New York time.
        ny_time = pd.to_datetime(ticker_news["published_utc"], utc=True).dt.tz_convert(MARKET_TIMEZONE)
        calendar_day = ny_time.dt.tz_localize(None).dt.normalize()

        # After the close, the earliest day the news can matter is tomorrow.
        after_close = ny_time.dt.hour >= MARKET_CLOSE_HOUR
        earliest_day = calendar_day.where(~after_close, calendar_day + pd.Timedelta(days=1))

        # Find the first trading day on or after that date.
        # (searchsorted returns len(trading_days) if there isn't one.)
        positions = np.searchsorted(trading_days, earliest_day.to_numpy(dtype="datetime64[ns]"))
        has_session = positions < len(trading_days)

        ticker_news = ticker_news[has_session].copy()
        ticker_news["session"] = pd.to_datetime(trading_days[positions[has_session]])
        matched.append(ticker_news)

    if not matched:
        return headlines.iloc[0:0].assign(session=pd.Series(dtype="datetime64[ns]"))
    return pd.concat(matched, ignore_index=True)
