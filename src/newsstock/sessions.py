"""Align headline timestamps with trading sessions.

A forecast for session ``t`` is made at the close of ``t`` and predicts whether
the close of ``t+1`` is higher. A headline therefore belongs to the first
session whose close comes after its publication time:

* published before 16:00 ET on a trading day  -> that day's session
* published at/after 16:00 ET, or on a weekend/holiday -> the next session

Trading days are taken from the market data itself, so weekends and exchange
holidays are handled without a separate holiday calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MARKET_CLOSE_HOUR, MARKET_TZ


def map_to_sessions(news: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Add a ``session`` column (trading date) to each headline; drop unmapped ones."""
    out = []
    for ticker, grp in news.groupby("ticker", sort=False):
        days = np.sort(market.loc[market["ticker"] == ticker, "date"].unique())
        if len(days) == 0:
            continue
        local = pd.to_datetime(grp["published_utc"], utc=True).dt.tz_convert(MARKET_TZ)
        candidate = local.dt.tz_localize(None).dt.normalize()
        candidate = candidate.where(local.dt.hour < MARKET_CLOSE_HOUR, candidate + pd.Timedelta(days=1))
        idx = np.searchsorted(days, candidate.to_numpy(dtype="datetime64[ns]"), side="left")
        valid = idx < len(days)
        g = grp.loc[valid].copy()
        g["session"] = pd.to_datetime(days[idx[valid]])
        out.append(g)
    if not out:
        return news.assign(session=pd.Series(dtype="datetime64[ns]")).iloc[0:0]
    return pd.concat(out, ignore_index=True)
