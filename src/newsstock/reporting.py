"""Plots and tables for the report: EDA, confusion matrices, model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .modeling import ExperimentResult  # noqa: E402


def comparison_table(result: ExperimentResult) -> pd.DataFrame:
    """Side-by-side test metrics: market-only vs market+sentiment per model."""
    m = result.metrics
    m = m[m["feature_set"] != "baseline"]
    wide = m.pivot(index="model", columns="feature_set", values=["accuracy", "f1", "roc_auc"])
    if {"market_only", "market_plus_sentiment"} <= set(m["feature_set"]):
        for metric in ("accuracy", "f1", "roc_auc"):
            wide[(metric, "delta")] = wide[(metric, "market_plus_sentiment")] - wide[(metric, "market_only")]
    return wide.sort_index(axis=1)


def plot_confusions(result: ExperimentResult, path: Path) -> None:
    keys = list(result.confusions)
    ncols = 3
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        sns.heatmap(result.confusions[key], annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["down", "up"], yticklabels=["down", "up"])
        ax.set_title(f"{key[1]}\n({key[0]})", fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_comparison(result: ExperimentResult, path: Path, metric: str = "accuracy") -> None:
    m = result.metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=m[m["feature_set"] != "baseline"], x="model", y=metric, hue="feature_set", ax=ax)
    base = m.loc[m["feature_set"] == "baseline", metric]
    if len(base):
        ax.axhline(base.iloc[0], ls="--", c="grey", label="majority baseline")
    ax.set_title(f"Test {metric}: market-only vs market + sentiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_eda(dataset: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    dataset.groupby("ticker")["news_count"].mean().plot.bar(ax=axes[0], title="Mean headlines per session")
    dataset.groupby("ticker")["target"].mean().plot.bar(ax=axes[1], title="Share of up days (class balance)")
    axes[1].axhline(0.5, ls="--", c="grey")
    news = dataset[dataset["has_news"] == 1]
    if len(news):
        news = news.assign(next_ret=np.log(news["next_close"] / news["close"]))
        bins = pd.qcut(news["sent_net_mean"], 5, duplicates="drop")
        news.groupby(bins, observed=True)["next_ret"].mean().plot.bar(ax=axes[2], title="Next-day log return by sentiment quintile")
        axes[2].tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
