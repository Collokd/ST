"""Feature engineering and label construction.

Every feature for row (ticker, t) uses only information available at the close
of session t. The label is ``1`` if the close of session t+1 is above the close
of session t, else ``0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5",
    "ret_5d_cum", "vol_5", "vol_20",
    "volume_chg", "volume_z20", "hl_range", "close_sma10_ratio",
]

SENTIMENT_FEATURES = [
    "has_news", "news_count_log", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_net_mean", "sent_net_std", "sent_pos_share", "sent_neg_share",
    "sent_net_mean_lag1", "sent_net_3d", "news_count_3d_log",
]

SENT_AGG_COLS = [
    "news_count", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_net_mean", "sent_net_std", "sent_pos_share", "sent_neg_share",
]


def _market_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    r = np.log(g["close"]).diff()
    for k in range(1, 6):
        g[f"ret_{k}"] = r.shift(k - 1)  # ret_1 = return over session t itself
    g["ret_5d_cum"] = r.rolling(5).sum()
    g["vol_5"] = r.rolling(5).std()
    g["vol_20"] = r.rolling(20).std()
    logv = np.log1p(g["volume"])
    g["volume_chg"] = logv.diff()
    g["volume_z20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    g["hl_range"] = (g["high"] - g["low"]) / g["close"]
    g["close_sma10_ratio"] = g["close"] / g["close"].rolling(10).mean() - 1
    g["next_close"] = g["close"].shift(-1)
    g["target"] = (g["next_close"] > g["close"]).astype(float)
    g.loc[g["next_close"].isna(), "target"] = np.nan
    return g


def _sentiment_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["has_news"] = (g["news_count"] > 0).astype(float)
    g["news_count_log"] = np.log1p(g["news_count"])
    g["sent_net_mean_lag1"] = g["sent_net_mean"].shift(1).fillna(0.0)
    g["sent_net_3d"] = g["sent_net_mean"].rolling(3, min_periods=1).mean()
    g["news_count_3d_log"] = np.log1p(g["news_count"].rolling(3, min_periods=1).sum())
    return g


def _per_ticker(df: pd.DataFrame, fn) -> pd.DataFrame:
    return pd.concat([fn(g) for _, g in df.groupby("ticker", sort=False)], ignore_index=True)


def build_dataset(market: pd.DataFrame, sentiment: pd.DataFrame | None) -> pd.DataFrame:
    """Merge market data with session-level sentiment and build all features."""
    df = _per_ticker(market, _market_features)
    if sentiment is None or sentiment.empty:
        sentiment = pd.DataFrame(columns=["ticker", "date", *SENT_AGG_COLS])
    df = df.merge(sentiment[["ticker", "date", *SENT_AGG_COLS]], on=["ticker", "date"], how="left")
    # Sessions without headlines: zero counts and zero sentiment, flagged by has_news.
    df[SENT_AGG_COLS] = df[SENT_AGG_COLS].astype(float).fillna(0.0)
    df = _per_ticker(df, _sentiment_features)
    df = df.dropna(subset=[*MARKET_FEATURES, "target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)
