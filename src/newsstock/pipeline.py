"""End-to-end pipeline: acquire -> clean -> align -> score -> features -> models -> report.

Examples::

    python -m newsstock.pipeline --source synthetic            # offline demo
    python -m newsstock.pipeline --source online --tickers AAPL MSFT --start 2023-01-01 --end 2024-06-30
    python -m newsstock.pipeline --source csv --data-dir data   # pre-downloaded data/raw/{market,news}_<T>.csv
"""

from __future__ import annotations

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


def load_data(cfg: Config, source: str, stooq_api_key: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "synthetic":
        market, news = make_synthetic(cfg.tickers, cfg.start, cfg.end, seed=cfg.random_state)
        news = pd.concat([news_data.clean_news(g, t) for t, g in news.groupby("ticker")], ignore_index=True)
        return market, news
    if source == "csv":
        market = pd.concat([market_data.load_market_csv(cfg.raw_dir / f"market_{t}.csv", t) for t in cfg.tickers], ignore_index=True)
        news = pd.concat([news_data.clean_news(pd.read_csv(cfg.raw_dir / f"news_{t}.csv"), t) for t in cfg.tickers], ignore_index=True)
        market = market[(market["date"] >= cfg.start) & (market["date"] <= cfg.end)]
        return market, news
    market = market_data.get_market_data(cfg.tickers, cfg.start, cfg.end, cfg.raw_dir, api_key=stooq_api_key)
    news = news_data.get_news_data(cfg.tickers, cfg.start, cfg.end, cfg.raw_dir)
    return market, news


def run(cfg: Config, source: str = "synthetic", use_finbert: bool = True, cluster: bool = False,
        stooq_api_key: str | None = None) -> dict:
    cfg.ensure_dirs()
    market, news = load_data(cfg, source, stooq_api_key)
    print(f"[data] {len(market):,} market rows, {len(news):,} cleaned headlines for {market['ticker'].nunique()} tickers")

    news = map_to_sessions(news, market)
    scorer = get_scorer(use_finbert, cfg.finbert_model)
    scored = score_headlines(news, scorer)
    scored.to_csv(cfg.processed_dir / "headlines_scored.csv", index=False)
    print(f"[sentiment] scored {len(scored):,} headlines with {scorer.name}")

    dataset = build_dataset(market, aggregate_sentiment(scored))
    dataset.to_csv(cfg.processed_dir / "model_dataset.csv", index=False)
    print(f"[features] {len(dataset):,} ticker-sessions; up-day share {dataset['target'].mean():.3f}; "
          f"sessions with news {dataset['has_news'].mean():.1%}")

    split = chronological_split(dataset, cfg.train_frac, cfg.val_frac)
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        print(f"[split] {name:5s} {part['date'].min():%Y-%m-%d} .. {part['date'].max():%Y-%m-%d}  n={len(part):,}")

    feature_sets = {"market_only": MARKET_FEATURES, "market_plus_sentiment": MARKET_FEATURES + SENTIMENT_FEATURES}
    result = run_experiment(split, feature_sets, cfg.random_state)

    out = cfg.output_dir
    metrics = result.metrics
    metrics.to_csv(out / "test_metrics.csv", index=False)
    comparison = reporting.comparison_table(result)
    comparison.to_csv(out / "market_vs_sentiment.csv")
    reporting.plot_eda(dataset, out / "eda.png")
    reporting.plot_confusions(result, out / "confusion_matrices.png")
    reporting.plot_metric_comparison(result, out / "accuracy_comparison.png", "accuracy")
    reporting.plot_metric_comparison(result, out / "f1_comparison.png", "f1")

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n[results] held-out test metrics")
        print(metrics.drop(columns=["best_params"]).to_string(index=False))
        print("\n[results] market-only vs market+sentiment")
        print(comparison.to_string())

    summary = {"dataset": dataset, "split": split, "result": result, "comparison": comparison, "scorer": scorer.name}
    if cluster:
        scores, profile = kmeans_profile(split.train, ["ret_1", "vol_5", "volume_chg", "sent_net_mean", "news_count_log"], random_state=cfg.random_state)
        profile.to_csv(out / "kmeans_profile.csv")
        summary["kmeans"] = (scores, profile)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print("\n[k-means] cluster profile (train period)")
            print(profile.to_string())
    print(f"\n[done] outputs written to {out.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "online", "csv"], default="synthetic")
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--no-finbert", action="store_true", help="use the lexicon fallback instead of FinBERT")
    ap.add_argument("--cluster", action="store_true", help="also run exploratory k-means")
    ap.add_argument("--stooq-api-key")
    args = ap.parse_args(argv)

    cfg = Config()
    for attr in ("tickers", "start", "end", "data_dir", "output_dir"):
        if getattr(args, attr):
            setattr(cfg, attr, getattr(args, attr))
    run(cfg, args.source, use_finbert=not args.no_finbert, cluster=args.cluster, stooq_api_key=args.stooq_api_key)


if __name__ == "__main__":
    main()
