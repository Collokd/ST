"""Runs the whole project from the command line, one KDD step after another.

Examples:
    python -m newsstock.pipeline --source synthetic --no-finbert     # quick offline test
    python -m newsstock.pipeline --source online --tickers AAPL MSFT --start 2023-01-01 --end 2024-06-30
    python -m newsstock.pipeline --source csv     # uses data/raw/market_<T>.csv and data/raw/news_<T>.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from . import market_data, news_data, reporting
from .clustering import kmeans_profile
from .config import Config
from .features import MARKET_FEATURES, SENTIMENT_FEATURES, build_dataset
from .modeling import chronological_split, run_experiment
from .sentiment import aggregate_sentiment, get_scorer, score_headlines
from .sessions import map_to_sessions
from .synthetic import make_synthetic

CLUSTER_COLUMNS = ["ret_1", "vol_5", "volume_chg", "sent_net_mean", "news_count_log"]


def load_data(config, source, stooq_api_key=None):
    """Returns (prices, headlines), both already cleaned."""
    if source == "synthetic":
        prices, headlines = make_synthetic(config.tickers, config.start, config.end, seed=config.random_state)
        cleaned = [news_data.clean_headlines(group, ticker) for ticker, group in headlines.groupby("ticker")]
        return prices, pd.concat(cleaned, ignore_index=True)

    if source == "csv":
        prices = pd.concat([market_data.load_market_csv(config.raw_dir / f"market_{t}.csv", t)
                            for t in config.tickers], ignore_index=True)
        headlines = pd.concat([news_data.clean_headlines(pd.read_csv(config.raw_dir / f"news_{t}.csv"), t)
                               for t in config.tickers], ignore_index=True)
        prices = prices[(prices["date"] >= config.start) & (prices["date"] <= config.end)]
        return prices, headlines

    prices = market_data.get_market_data(config.tickers, config.start, config.end, config.raw_dir, stooq_api_key)
    headlines = news_data.get_news_data(config.tickers, config.start, config.end, config.raw_dir)
    return prices, headlines


def run(config, source="synthetic", use_finbert=True, cluster=False, stooq_api_key=None):
    config.make_folders()
    out = config.output_dir

    # Step 1: get and clean both datasets
    prices, headlines = load_data(config, source, stooq_api_key)
    print(f"Loaded {len(prices):,} days of prices and {len(headlines):,} headlines "
          f"for {prices['ticker'].nunique()} stocks.")

    # Step 2: match each headline to a trading day
    headlines = map_to_sessions(headlines, prices)

    # Step 3: sentiment
    scorer = get_scorer(use_finbert, config.finbert_model)
    scored = score_headlines(headlines, scorer)
    scored.to_csv(config.processed_dir / "headlines_scored.csv", index=False)
    print(f"Scored {len(scored):,} headlines with {scorer.name}.")

    # Step 4: join the datasets and build features
    data = build_dataset(prices, aggregate_sentiment(scored))
    data.to_csv(config.processed_dir / "model_dataset.csv", index=False)
    print(f"Final table: {len(data):,} rows (stock-days). "
          f"{data['target'].mean():.1%} are up days, {data['has_news'].mean():.1%} have news.")

    # Step 5: split by time
    split = chronological_split(data, config.train_frac, config.val_frac)
    for name, part in [("train", split.train), ("validation", split.val), ("test", split.test)]:
        print(f"  {name:10s} {part['date'].min():%Y-%m-%d} to {part['date'].max():%Y-%m-%d}  ({len(part):,} rows)")

    # Step 6: train and evaluate the models
    feature_sets = {
        "market_only": MARKET_FEATURES,
        "market_plus_sentiment": MARKET_FEATURES + SENTIMENT_FEATURES,
    }
    result = run_experiment(split, feature_sets, config.random_state)

    # Step 7: save tables and charts
    result.metrics.to_csv(out / "test_metrics.csv", index=False)
    comparison = reporting.comparison_table(result)
    comparison.to_csv(out / "market_vs_sentiment.csv")
    reporting.plot_eda(data, out / "eda.png")
    reporting.plot_confusions(result, out / "confusion_matrices.png")
    reporting.plot_metric_comparison(result, out / "accuracy_comparison.png", "accuracy")
    reporting.plot_metric_comparison(result, out / "f1_comparison.png", "f1")

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\nTest set results:")
        print(result.metrics.drop(columns=["best_params"]).to_string(index=False))
        print("\nMarket only vs market + sentiment:")
        print(comparison.to_string())

    summary = {"dataset": data, "split": split, "result": result, "comparison": comparison, "scorer": scorer.name}

    # Optional step: k-means clustering on the training period
    if cluster:
        silhouette, profile, crosstab = kmeans_profile(split.train, CLUSTER_COLUMNS, random_state=config.random_state)
        profile.to_csv(out / "kmeans_profile.csv")
        summary["kmeans"] = (silhouette, profile, crosstab)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print("\nk-means clusters (training period):")
            print(profile.to_string())
            print(crosstab.to_string())

    print(f"\nAll outputs saved in {out.resolve()}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "online", "csv"], default="synthetic")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-finbert", action="store_true", help="use the simple word-list scorer instead of FinBERT")
    parser.add_argument("--cluster", action="store_true", help="also run the k-means clustering")
    parser.add_argument("--stooq-api-key")
    args = parser.parse_args(argv)

    config = Config()
    for setting in ["tickers", "start", "end", "data_dir", "output_dir"]:
        value = getattr(args, setting)
        if value:
            setattr(config, setting, value)

    run(config, args.source, use_finbert=not args.no_finbert, cluster=args.cluster, stooq_api_key=args.stooq_api_key)


if __name__ == "__main__":
    main()
