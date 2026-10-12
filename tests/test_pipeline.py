import numpy as np
import pandas as pd
import pytest

from newsstock.config import Config
from newsstock.features import MARKET_FEATURES, SENTIMENT_FEATURES, build_dataset
from newsstock.modeling import chronological_split
from newsstock.news_data import clean_headlines
from newsstock.pipeline import run
from newsstock.sentiment import WordListScorer, aggregate_sentiment, score_headlines
from newsstock.sessions import map_to_sessions


def _market(dates, ticker="AAPL"):
    n = len(dates)
    close = np.linspace(100, 100 + n - 1, n)
    return pd.DataFrame({"ticker": ticker, "date": pd.to_datetime(dates), "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1e6 + 1e4 * (np.arange(n) % 7)})


def test_session_mapping_handles_close_weekend_and_holiday():
    # Thu 2024-07-03, (Thu 07-04 holiday absent), Fri 07-05, Mon 07-08
    market = _market(["2024-07-03", "2024-07-05", "2024-07-08"])
    news = pd.DataFrame({
        "ticker": "AAPL",
        "published_utc": pd.to_datetime([
            "2024-07-03 19:59",  # 15:59 ET Wed -> Wed
            "2024-07-03 20:00",  # 16:00 ET Wed (at close) -> next session Fri
            "2024-07-04 15:00",  # holiday -> Fri
            "2024-07-06 15:00",  # Saturday -> Mon
            "2024-07-08 21:00",  # after last close -> unmapped
        ], utc=True),
        "headline": list("abcde"),
    })
    out = map_to_sessions(news, market).set_index("headline")["session"].dt.strftime("%Y-%m-%d")
    assert out.to_dict() == {"a": "2024-07-03", "b": "2024-07-05", "c": "2024-07-05", "d": "2024-07-08"}


def test_clean_news_drops_duplicates_missing_and_unrelated():
    df = pd.DataFrame({
        "published_utc": ["2024-01-02 10:00Z", "2024-01-02 11:00Z", "2024-01-02 12:00Z", None, "2024-01-02 13:00Z"],
        "headline": ["Apple beats estimates", "APPLE beats estimates!", "Pie recipes for fall", "Apple news", "Apple shares fall"],
        "url": ["u1", "u2", "u3", "u4", "u5"],
    })
    out = clean_headlines(df, "AAPL")
    assert out["headline"].tolist() == ["Apple beats estimates", "Apple shares fall"]
    assert (out["ticker"] == "AAPL").all()


def test_features_use_only_past_data_and_label_is_next_close():
    dates = pd.bdate_range("2024-01-01", periods=40)
    market = _market(dates)
    base = build_dataset(market, None)
    # Changing a future close must not change features at earlier dates.
    shocked = market.copy()
    shocked.loc[shocked.index[-5], "close"] *= 2
    after = build_dataset(shocked, None)
    cutoff = dates[-6]
    a = base[base["date"] < cutoff][MARKET_FEATURES].to_numpy()
    b = after[after["date"] < cutoff][MARKET_FEATURES].to_numpy()
    assert np.allclose(a, b)
    assert (base["target"] == 1).all()  # strictly rising closes
    assert base["date"].max() < dates[-1]  # last session has no label


def test_sentiment_aggregation_and_no_news_fill():
    market = _market(pd.bdate_range("2024-01-01", periods=30))
    news = pd.DataFrame({"ticker": "AAPL", "session": [market["date"].iloc[25]] * 2,
                         "headline": ["Apple shares surge on record profit", "Apple faces lawsuit, shares fall"]})
    agg = aggregate_sentiment(score_headlines(news, WordListScorer()))
    assert agg["news_count"].tolist() == [2]
    ds = build_dataset(market, agg)
    assert set(SENTIMENT_FEATURES) <= set(ds.columns)
    assert ds["has_news"].sum() == 1
    assert ds.loc[ds["has_news"] == 0, "sent_net_mean"].eq(0).all()


def test_chronological_split_is_ordered_with_gap():
    ds = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=100), "target": 0})
    s = chronological_split(ds, 0.6, 0.2, gap=1)
    assert s.train["date"].max() < s.val["date"].min() < s.val["date"].max() < s.test["date"].min()
    assert len(s.train) == 59 and len(s.val) == 19 and len(s.test) == 20


@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    cfg = Config(tickers=["AAPL", "MSFT"], start="2023-01-01", end="2023-12-31",
                 data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    summary = run(cfg, "synthetic", use_finbert=False)
    m = summary["result"].metrics
    assert set(m["feature_set"]) == {"baseline", "market_only", "market_plus_sentiment"}
    assert (cfg.output_dir / "test_metrics.csv").exists()
    assert (cfg.output_dir / "confusion_matrices.png").exists()
