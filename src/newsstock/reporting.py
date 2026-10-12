"""Tables and charts for the report and presentation."""

import matplotlib

matplotlib.use("Agg")  # save charts to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def comparison_table(result):
    """Test-set scores side by side: market only vs market + sentiment, and the difference."""
    metrics = result.metrics[result.metrics["feature_set"] != "baseline"]
    table = metrics.pivot(index="model", columns="feature_set", values=["accuracy", "f1", "roc_auc"])

    if {"market_only", "market_plus_sentiment"} <= set(metrics["feature_set"]):
        for metric in ["accuracy", "f1", "roc_auc"]:
            table[(metric, "difference")] = (table[(metric, "market_plus_sentiment")]
                                             - table[(metric, "market_only")])
    return table.sort_index(axis=1)


def plot_confusions(result, path):
    names = list(result.confusions)
    n_cols = 3
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows), squeeze=False)

    for ax, (feature_set, model) in zip(axes.flat, names):
        sns.heatmap(result.confusions[(feature_set, model)], annot=True, fmt="d", cbar=False, cmap="Blues",
                    ax=ax, xticklabels=["down", "up"], yticklabels=["down", "up"])
        ax.set_title(f"{model}\n({feature_set})", fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")

    # Hide any empty panels in the last row.
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_comparison(result, path, metric="accuracy"):
    metrics = result.metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metrics[metrics["feature_set"] != "baseline"], x="model", y=metric, hue="feature_set", ax=ax)

    baseline = metrics.loc[metrics["feature_set"] == "baseline", metric]
    if len(baseline):
        ax.axhline(baseline.iloc[0], linestyle="--", color="grey", label="majority-class baseline")

    ax.set_title(f"Test {metric}: market only vs market + sentiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_eda(data, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    data.groupby("ticker")["news_count"].mean().plot.bar(ax=axes[0], title="Average headlines per trading day")

    data.groupby("ticker")["target"].mean().plot.bar(ax=axes[1], title="Share of up days (class balance)")
    axes[1].axhline(0.5, linestyle="--", color="grey")

    # Do days with more positive news tend to be followed by better returns?
    news_days = data[data["has_news"] == 1].copy()
    if len(news_days):
        news_days["next_return"] = np.log(news_days["next_close"] / news_days["close"])
        quintile = pd.qcut(news_days["sent_net_mean"], 5, duplicates="drop")
        news_days.groupby(quintile, observed=True)["next_return"].mean().plot.bar(
            ax=axes[2], title="Next-day return by sentiment quintile")
        axes[2].tick_params(axis="x", labelrotation=30, labelsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
