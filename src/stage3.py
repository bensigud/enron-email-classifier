"""
Stage 3 — Unsupervised Clustering

Takes the pair feature vectors from Stage 2 and discovers natural
relationship groupings using:
  - K-Means (trying K=3..7, picking best by silhouette score)
  - DBSCAN (finds number of clusters automatically)
  - Compares both methods
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.stage2 import FEATURE_COLS


def cluster_relationships(pair_features_df: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Run unsupervised clustering on pair feature vectors.
    """
    print("  Clustering relationships...")

    X = pair_features_df[FEATURE_COLS].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means ---
    kmeans_scores = {}
    kmeans_models = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        kmeans_scores[k] = score
        kmeans_models[k] = (km, labels)
        print(f"    K-Means K={k}: silhouette={score:.3f}")

    best_k = max(kmeans_scores, key=kmeans_scores.get)
    best_km_score = kmeans_scores[best_k]
    print(f"    Best K-Means: K={best_k} (silhouette={best_km_score:.3f})")

    # --- DBSCAN ---
    dbscan_scores = {}
    dbscan_models = {}
    for eps in [0.5, 0.75, 1.0, 1.5, 2.0]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > 0 and len(set(labels[mask])) >= 2:
                score = silhouette_score(X_scaled[mask], labels[mask])
                n_noise = (labels == -1).sum()
                dbscan_scores[eps] = score
                dbscan_models[eps] = (db, labels)
                print(f"    DBSCAN eps={eps}: {n_clusters} clusters, "
                      f"{n_noise} outliers, silhouette={score:.3f}")

    # --- Pick best ---
    best_method = "kmeans"
    best_labels = kmeans_models[best_k][1]
    best_score = best_km_score

    if dbscan_scores:
        best_eps = max(dbscan_scores, key=dbscan_scores.get)
        if dbscan_scores[best_eps] > best_km_score:
            best_method = "dbscan"
            best_labels = dbscan_models[best_eps][1]
            best_score = dbscan_scores[best_eps]
            print(f"\n    Best method: DBSCAN eps={best_eps} "
                  f"(silhouette={best_score:.3f})")
        else:
            print(f"\n    Best method: K-Means K={best_k} "
                  f"(silhouette={best_score:.3f})")
    else:
        print(f"\n    Best method: K-Means K={best_k} "
              f"(silhouette={best_score:.3f})")
        print("    (DBSCAN did not find valid clusters)")

    pair_features_df = pair_features_df.copy()
    pair_features_df["cluster"] = best_labels

    # Save plots and metadata
    _plot_silhouette_scores(kmeans_scores, output_dir)
    _plot_cluster_distribution(pair_features_df, output_dir)

    # Cluster profiles
    valid = pair_features_df[pair_features_df["cluster"] != -1]
    profiles = valid.groupby("cluster")[FEATURE_COLS].mean().round(4)
    profiles["pair_count"] = valid.groupby("cluster").size()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out / "cluster_profiles.csv")
    _plot_cluster_heatmap(profiles[FEATURE_COLS], output_dir)

    print("\n    Cluster profiles:")
    summary_cols = ["avg_intimacy", "avg_warmth", "avg_sentiment",
                    "email_count", "time_span_days", "same_community", "pair_count"]
    available = [c for c in summary_cols if c in profiles.columns]
    print(profiles[available].to_string())

    # Save metadata
    meta = {
        "best_method": best_method,
        "best_score": best_score,
        "kmeans_scores": {str(k): v for k, v in kmeans_scores.items()},
        "dbscan_scores": {str(k): v for k, v in dbscan_scores.items()},
    }
    if best_method == "kmeans":
        meta["best_k"] = best_k
    with open(out / "clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return pair_features_df


def _plot_silhouette_scores(kmeans_scores, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(kmeans_scores.keys())
    scores = list(kmeans_scores.values())
    ax.plot(ks, scores, "o-", label="K-Means", color="#3498db")
    best_k = max(kmeans_scores, key=kmeans_scores.get)
    ax.axvline(x=best_k, color="#3498db", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs Number of Clusters", fontsize=13)
    ax.legend()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "silhouette_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cluster_distribution(pair_df, output_dir):
    counts = pair_df["cluster"].value_counts().sort_index()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#e91e8c", "#1abc9c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [colors[i % len(colors)] for i in range(len(counts))]
    ax.bar(counts.index.astype(str), counts.values, color=bar_colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Pairs")
    ax.set_title("Relationship Cluster Distribution", fontsize=13)
    for i, (idx, val) in enumerate(counts.items()):
        ax.text(i, val + 0.5, str(val), ha="center", fontsize=10)

    out = Path(output_dir)
    plt.savefig(out / "cluster_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cluster_heatmap(profiles, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    data = profiles.T
    im = ax.imshow(data.values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"Cluster {c}" for c in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title("Cluster Feature Profiles", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    out = Path(output_dir)
    plt.savefig(out / "cluster_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_stage3(pair_features_df: pd.DataFrame,
               output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 3: cluster the pair feature vectors.
    """
    print("\n=== STAGE 3: Unsupervised Clustering ===")

    pair_features_df = cluster_relationships(pair_features_df, output_dir)

    # Save updated pairs with cluster column
    out = Path(output_dir)
    pair_features_df.to_csv(out / "relationship_pairs.csv", index=False)

    print(f"\n  Stage 3 complete. {len(pair_features_df):,} pairs clustered "
          f"into {pair_features_df['cluster'].nunique()} groups")

    return pair_features_df
