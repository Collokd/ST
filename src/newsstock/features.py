"""Building the features and the label for each (stock, trading day).

Rule we follow everywhere: a feature for day t may only use information we
would have at day t's close. The label is 1 if the stock closes higher on
day t+1 than on day t, and 0 otherwise.
"""

import numpy as np
import pandas as pd

# Features built from prices and volume only.
MARKET_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5",  # today's return and the 4 before it
    "ret_5d_cum",                                  # total return over the last 5 days
    "vol_5", "vol_20",                             # volatility (std of returns) over 5 and 20 days
    "volume_chg", "volume_z20",                    # volume change, and how unusual today's volume is
    "hl_range",                                    # (high - low) / close
    "close_sma10_ratio",                           # how far the close is from its 10-day average
]

# Features built from the news.
SENTIMENT_FEATURES = [
    "has_news", "news_count_log",
    "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_net_mean", "sent_net_std", "sent_pos_share", "sent_neg_share",
    "sent_net_mean_lag1",   # yesterday's net sentiment
    "sent_net_3d",          # average net sentiment over the last 3 days
    "news_count_3d_log",    # number of headlines over the last 3 days (log)
]

DAILY_SENTIMENT_COLUMNS = [
    "news_count", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_net_mean", "sent_net_std", "sent_pos_share", "sent_neg_share",
]


def add_market_features(stock):
    """Price/volume features for one stock. Rolling windows only look backwards."""
    stock = stock.sort_values("date").copy()
    log_return = np.log(stock["close"]).diff()

    for k in range(1, 6):
        stock[f"ret_{k}"] = log_return.shift(k - 1)
    stock["ret_5d_cum"] = log_return.rolling(5).sum()
    stock["vol_5"] = log_return.rolling(5).std()
    stock["vol_20"] = log_return.rolling(20).std()

    log_volume = np.log1p(stock["volume"])
    stock["volume_chg"] = log_volume.diff()
    stock["volume_z20"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()

    stock["hl_range"] = (stock["high"] - stock["low"]) / stock["close"]
    stock["close_sma10_ratio"] = stock["close"] / stock["close"].rolling(10).mean() - 1

    # The label looks one day ahead. The last day has no "tomorrow" yet, so its label is empty.
    stock["next_close"] = stock["close"].shift(-1)
    stock["target"] = np.where(stock["next_close"] > stock["close"], 1.0, 0.0)
    stock.loc[stock["next_close"].isna(), "target"] = np.nan
    return stock


def add_sentiment_features(stock):
    stock = stock.sort_values("date").copy()
    stock["has_news"] = (stock["news_count"] > 0).astype(float)
    stock["news_count_log"] = np.log1p(stock["news_count"])
    stock["sent_net_mean_lag1"] = stock["sent_net_mean"].shift(1).fillna(0.0)
    stock["sent_net_3d"] = stock["sent_net_mean"].rolling(3, min_periods=1).mean()
    stock["news_count_3d_log"] = np.log1p(stock["news_count"].rolling(3, min_periods=1).sum())
    return stock


def for_each_stock(table, function):
    pieces = [function(stock) for _, stock in table.groupby("ticker", sort=False)]
    return pd.concat(pieces, ignore_index=True)


def build_dataset(prices, daily_sentiment):
    """Join prices with daily sentiment (on ticker + date) and build all features."""
    data = for_each_stock(prices, add_market_features)

    if daily_sentiment is None or daily_sentiment.empty:
        daily_sentiment = pd.DataFrame(columns=["ticker", "date"] + DAILY_SENTIMENT_COLUMNS)

    # This is the join between the two datasets. We use a left join so that
    # trading days without any news are kept.
    data = data.merge(daily_sentiment[["ticker", "date"] + DAILY_SENTIMENT_COLUMNS],
                      on=["ticker", "date"], how="left")

    # Days with no news: 0 headlines and 0 sentiment. has_news tells the model which days these are.
    data[DAILY_SENTIMENT_COLUMNS] = data[DAILY_SENTIMENT_COLUMNS].astype(float).fillna(0.0)
    data = for_each_stock(data, add_sentiment_features)

    # The first ~20 days of each stock don't have enough history for the
    # rolling features, and the last day has no label. Drop those rows.
    data = data.dropna(subset=MARKET_FEATURES + ["target"]).reset_index(drop=True)
    data["target"] = data["target"].astype(int)
    return data.sort_values(["date", "ticker"]).reset_index(drop=True)
