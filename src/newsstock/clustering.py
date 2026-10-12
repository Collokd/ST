"""Optional: exploratory k-means clustering of ticker-sessions."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def kmeans_profile(df: pd.DataFrame, cols: list[str], k_values=range(2, 7), random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick k by silhouette score and profile clusters (incl. next-day up rate)."""
    X = StandardScaler().fit_transform(df[cols])
    sample = min(len(X), 3000)
    scores = []
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
        scores.append({"k": k, "silhouette": silhouette_score(X, labels, sample_size=sample, random_state=random_state)})
    scores = pd.DataFrame(scores)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(X)
    profile = df.assign(cluster=labels).groupby("cluster")[[*cols, "target"]].mean()
    profile.insert(0, "size", pd.Series(labels).value_counts().sort_index().to_numpy())
    return scores, profile
