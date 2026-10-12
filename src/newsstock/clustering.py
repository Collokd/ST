"""Exploratory k-means clustering of (stock, day) rows.

Question: if we group trading days by their market and news behaviour,
without telling the algorithm about the label, do some groups end up with
noticeably more "up" next days than others?

We choose k with the silhouette score and then compare clusters to the
up/down label with a cross-tabulation.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def kmeans_profile(data, columns, k_values=range(2, 7), random_state=42):
    """Returns (silhouette score per k, cluster profile, cluster vs label cross-tab)."""
    X = StandardScaler().fit_transform(data[columns])

    # Silhouette is slow on big data, so we score a sample of up to 3000 rows.
    sample_size = min(len(X), 3000)
    silhouette_rows = []
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
        score = silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
        silhouette_rows.append({"k": k, "silhouette": score})
    silhouette = pd.DataFrame(silhouette_rows)

    best_k = int(silhouette.loc[silhouette["silhouette"].idxmax(), "k"])
    clusters = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(X)
    with_clusters = data.assign(cluster=clusters)

    # Average feature values per cluster, plus the share of days followed by an up day.
    profile = with_clusters.groupby("cluster")[columns + ["target"]].mean()
    profile = profile.rename(columns={"target": "up_next_day_share"})
    profile.insert(0, "size", with_clusters["cluster"].value_counts().sort_index())

    crosstab = pd.crosstab(with_clusters["cluster"], with_clusters["target"].map({0: "down", 1: "up"}))
    return silhouette, profile, crosstab
