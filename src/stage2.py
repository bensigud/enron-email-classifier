"""
Stage 2 — Pair-Level Relationship Clustering

For every pair of people who exchanged enough emails, we:
  1. Build a feature vector (17 features from 3 sources)
  2. Standardise features so no single one dominates
  3. Cluster with K-Means (trying K=3..7) and DBSCAN
  4. Compare methods using silhouette score
  5. Interpret clusters by examining their feature averages

The clusters ARE the relationship types — the algorithm discovers them
from the data, we interpret and name them.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# Minimum emails between two people to include them as a pair
MIN_EMAILS_PER_PAIR = 5


def build_pair_features(df: pd.DataFrame,
                        person_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 17-feature vector for every qualifying pair.

    Features come from 3 sources:
      - Stage 1 scores (personal_score, sentiment_score) aggregated per pair
      - Email patterns (count, direction ratio)
      - Network features (degree, pagerank, community)

    We compute directional features (A→B and B→A separately) because
    relationships are not always symmetric — mentorship or one-sided
    friendships show up as imbalanced scores.
    """
    print("  Building pair feature vectors...")

    # Explode recipients so each row = one sender → one recipient
    exploded = df[["sender", "recipients", "personal_score", "sentiment_score"]].copy()
    rows = []
    for _, row in exploded.iterrows():
        for recipient in row["recipients"]:
            if recipient != row["sender"]:
                rows.append({
                    "sender": row["sender"],
                    "recipient": recipient,
                    "personal_score": row["personal_score"],
                    "sentiment_score": row["sentiment_score"],
                })

    edge_df = pd.DataFrame(rows)

    # --- Directional aggregation (A→B separately from B→A) ---
    directional = (edge_df
                   .groupby(["sender", "recipient"])
                   .agg(
                       avg_personal=("personal_score", "mean"),
                       avg_sentiment=("sentiment_score", "mean"),
                       email_count=("personal_score", "count"),
                   )
                   .reset_index())

    # Only keep directions with enough emails
    directional = directional[directional["email_count"] >= 2]

    # --- Merge both directions to get pairs ---
    # Join A→B with B→A
    merged = directional.merge(
        directional.rename(columns={
            "sender": "recipient",
            "recipient": "sender",
            "avg_personal": "avg_personal_reverse",
            "avg_sentiment": "avg_sentiment_reverse",
            "email_count": "email_count_reverse",
        }),
        on=["sender", "recipient"],
        how="inner",
    )

    # De-duplicate: keep each pair once (alphabetical order)
    merged["person_a"] = merged.apply(
        lambda r: min(r["sender"], r["recipient"]), axis=1
    )
    merged["person_b"] = merged.apply(
        lambda r: max(r["sender"], r["recipient"]), axis=1
    )
    merged = merged.drop_duplicates(subset=["person_a", "person_b"])

    # Total emails for the pair
    merged["email_count"] = merged["email_count"] + merged["email_count_reverse"]

    # Filter by minimum emails
    merged = merged[merged["email_count"] >= MIN_EMAILS_PER_PAIR]

    # --- Compute per-pair email-level stats ---
    # We need std for personal and sentiment scores per pair
    pair_stats = _compute_pair_stats(edge_df, merged)
    merged = merged.merge(pair_stats, on=["person_a", "person_b"], how="left")

    # --- Build the 17 features ---
    person_feat = person_features_df.set_index("person")
    features = pd.DataFrame()
    features["person_a"] = merged["person_a"]
    features["person_b"] = merged["person_b"]

    # Stage 1 features (aggregated)
    features["avg_personal_score"] = (
        merged["avg_personal"] + merged["avg_personal_reverse"]
    ) / 2
    features["avg_sentiment"] = (
        merged["avg_sentiment"] + merged["avg_sentiment_reverse"]
    ) / 2
    features["personal_score_std"] = merged["personal_std"].fillna(0)
    features["sentiment_std"] = merged["sentiment_std"].fillna(0)

    # Directional features
    features["personal_a_to_b"] = merged["avg_personal"]
    features["personal_b_to_a"] = merged["avg_personal_reverse"]
    features["personal_imbalance"] = abs(
        merged["avg_personal"] - merged["avg_personal_reverse"]
    )
    features["sentiment_a_to_b"] = merged["avg_sentiment"]
    features["sentiment_b_to_a"] = merged["avg_sentiment_reverse"]
    features["sentiment_imbalance"] = abs(
        merged["avg_sentiment"] - merged["avg_sentiment_reverse"]
    )

    # Email pattern features
    features["email_count"] = merged["email_count"]
    a_count = merged["email_count"] - merged["email_count_reverse"]
    total = merged["email_count"]
    features["direction_ratio"] = (a_count / total).clip(0, 1)

    # Network features
    features["sender_degree"] = features["person_a"].map(
        person_feat["total_degree"]
    ).fillna(0)
    features["recipient_degree"] = features["person_b"].map(
        person_feat["total_degree"]
    ).fillna(0)
    features["degree_difference"] = abs(
        features["sender_degree"] - features["recipient_degree"]
    )
    features["same_community"] = features.apply(
        lambda r: (
            1 if (r["person_a"] in person_feat.index and
                  r["person_b"] in person_feat.index and
                  person_feat.loc[r["person_a"], "community"] ==
                  person_feat.loc[r["person_b"], "community"])
            else 0
        ), axis=1
    )

    pr_a = features["person_a"].map(person_feat["pagerank"]).fillna(0)
    pr_b = features["person_b"].map(person_feat["pagerank"]).fillna(0)
    features["pagerank_ratio"] = pr_a / pr_b.clip(lower=1e-10)

    print(f"    Built features for {len(features):,} pairs")
    return features


def _compute_pair_stats(edge_df: pd.DataFrame,
                        merged: pd.DataFrame) -> pd.DataFrame:
    """Compute std of personal_score and sentiment_score per pair."""
    edge_df = edge_df.copy()
    edge_df["person_a"] = edge_df.apply(
        lambda r: min(r["sender"], r["recipient"]), axis=1
    )
    edge_df["person_b"] = edge_df.apply(
        lambda r: max(r["sender"], r["recipient"]), axis=1
    )

    pair_keys = set(zip(merged["person_a"], merged["person_b"]))
    mask = edge_df.apply(
        lambda r: (r["person_a"], r["person_b"]) in pair_keys, axis=1
    )
    filtered = edge_df[mask]

    stats = (filtered
             .groupby(["person_a", "person_b"])
             .agg(
                 personal_std=("personal_score", "std"),
                 sentiment_std=("sentiment_score", "std"),
             )
             .reset_index())

    return stats


def cluster_relationships(pair_features_df: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Run unsupervised clustering on pair feature vectors.

    1. Standardise features (mean=0, std=1) so no single feature dominates
    2. Try K-Means with K = 3, 4, 5, 6, 7 — pick best by silhouette score
    3. Try DBSCAN — it finds the number of clusters automatically
    4. Compare and pick the best method
    5. Assign cluster labels to each pair

    Returns the pair DataFrame with a 'cluster' column added.
    """
    print("  Clustering relationships...")

    feature_cols = [
        "avg_personal_score", "avg_sentiment", "personal_score_std",
        "sentiment_std", "personal_a_to_b", "personal_b_to_a",
        "personal_imbalance", "sentiment_a_to_b", "sentiment_b_to_a",
        "sentiment_imbalance", "email_count", "direction_ratio",
        "sender_degree", "recipient_degree", "degree_difference",
        "same_community", "pagerank_ratio",
    ]

    X = pair_features_df[feature_cols].copy()
    X = X.fillna(0)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means: try different K values ---
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
    # eps controls how close points must be to form a cluster
    # We try a few values and pick the best
    dbscan_scores = {}
    dbscan_models = {}
    for eps in [0.5, 0.75, 1.0, 1.5, 2.0]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            # Silhouette score needs at least 2 clusters
            # Exclude noise points (-1) from score
            mask = labels != -1
            if mask.sum() > 0 and len(set(labels[mask])) >= 2:
                score = silhouette_score(X_scaled[mask], labels[mask])
                n_noise = (labels == -1).sum()
                dbscan_scores[eps] = score
                dbscan_models[eps] = (db, labels)
                print(f"    DBSCAN eps={eps}: {n_clusters} clusters, "
                      f"{n_noise} outliers, silhouette={score:.3f}")

    # --- Pick the best overall method ---
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

    # Save plots
    _plot_silhouette_scores(kmeans_scores, dbscan_scores, output_dir)
    _plot_cluster_distribution(pair_features_df, output_dir)

    # Save clustering metadata
    out = Path(output_dir)
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


def interpret_clusters(pair_features_df: pd.DataFrame,
                       output_dir: str) -> pd.DataFrame:
    """
    For each cluster, compute the average of every feature.

    This tells us what each cluster "looks like" — e.g., a cluster
    with high personal_score + high sentiment + bidirectional is probably
    Friendly or Romantic.

    We save this as a CSV so the team can examine it and assign
    human-readable names to each cluster in the report.
    """
    print("  Interpreting clusters...")

    feature_cols = [
        "avg_personal_score", "avg_sentiment", "personal_score_std",
        "sentiment_std", "personal_a_to_b", "personal_b_to_a",
        "personal_imbalance", "sentiment_a_to_b", "sentiment_b_to_a",
        "sentiment_imbalance", "email_count", "direction_ratio",
        "sender_degree", "recipient_degree", "degree_difference",
        "same_community", "pagerank_ratio",
    ]

    # Only interpret non-outlier clusters (DBSCAN marks outliers as -1)
    valid = pair_features_df[pair_features_df["cluster"] != -1]

    profiles = valid.groupby("cluster")[feature_cols].mean().round(4)
    profiles["pair_count"] = valid.groupby("cluster").size()

    out = Path(output_dir)
    profiles.to_csv(out / "cluster_profiles.csv")
    print(f"    Cluster profiles saved to {output_dir}/cluster_profiles.csv")

    # Print a summary
    print("\n    Cluster profiles (average feature values):")
    summary_cols = [
        "avg_personal_score", "avg_sentiment", "personal_imbalance",
        "sentiment_imbalance", "email_count", "same_community", "pair_count",
    ]
    print(profiles[summary_cols].to_string())

    # Feature heatmap
    _plot_cluster_heatmap(profiles[feature_cols], output_dir)

    return profiles


def _plot_silhouette_scores(kmeans_scores: dict, dbscan_scores: dict,
                            output_dir: str):
    """Plot silhouette scores for K-Means and DBSCAN."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if kmeans_scores:
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
    print(f"  Silhouette plot saved to {output_dir}/silhouette_scores.png")


def _plot_cluster_distribution(pair_df: pd.DataFrame, output_dir: str):
    """Bar chart of how many pairs are in each cluster."""
    counts = pair_df["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#e91e8c", "#1abc9c", "#95a5a6"]
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
    print(f"  Cluster distribution saved to {output_dir}/cluster_distribution.png")


def _plot_cluster_heatmap(profiles: pd.DataFrame, output_dir: str):
    """Heatmap showing average feature values per cluster."""
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
    print(f"  Cluster heatmap saved to {output_dir}/cluster_heatmap.png")


def save_results(pair_features_df: pd.DataFrame, output_dir: str):
    """Save all pair-level results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_features_df.to_csv(out / "relationship_pairs.csv", index=False)
    print(f"  Relationship pairs saved to {output_dir}/relationship_pairs.csv")


def run_stage2(df: pd.DataFrame, person_features_df: pd.DataFrame,
               output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 2 pipeline:
      1. Build pair feature vectors (17 features)
      2. Cluster with K-Means + DBSCAN, compare
      3. Interpret clusters
      4. Save results

    Returns the pair features DataFrame with cluster assignments.
    """
    print("\n=== STAGE 2: Pair-Level Relationship Clustering ===")

    # Step 1 — Build features
    pair_features = build_pair_features(df, person_features_df)

    # Step 2 — Cluster
    pair_features = cluster_relationships(pair_features, output_dir)

    # Step 3 — Interpret
    interpret_clusters(pair_features, output_dir)

    # Step 4 — Save
    save_results(pair_features, output_dir)

    print(f"\n  Stage 2 complete. {len(pair_features):,} pairs clustered "
          f"into {pair_features['cluster'].nunique()} groups")

    return pair_features
