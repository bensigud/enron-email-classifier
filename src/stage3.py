"""
Stage 3 — Unsupervised Clustering

Takes the pair feature vectors from Stage 2 and discovers natural
relationship groupings using:
  - GMM (soft clustering — each pair gets a probability per cluster)
  - K-Means (hard clustering — for comparison)
  - DBSCAN (density-based — finds outliers)
  - Compares all methods by silhouette score, prefers GMM for soft labels
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.stage2 import FEATURE_COLS


def cluster_relationships(pair_features_df: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Run unsupervised clustering on pair feature vectors.
    Uses GMM (soft), K-Means (hard), and DBSCAN (density).
    GMM is preferred because it gives probabilities per cluster,
    allowing relationships to have mixed types.
    """
    print("  Clustering relationships...")

    X = pair_features_df[FEATURE_COLS].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- GMM (soft clustering) ---
    gmm_scores = {}
    gmm_models = {}
    for k in range(3, 8):
        gmm = GaussianMixture(
            n_components=k, random_state=42, n_init=5, covariance_type="full"
        )
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        gmm_scores[k] = score
        gmm_models[k] = gmm
        print(f"    GMM K={k}: silhouette={score:.3f}, BIC={gmm.bic(X_scaled):.0f}")

    # Pick K using BIC (lower is better) — silhouette alone always picks K=3
    # BIC balances fit quality with model complexity
    gmm_bics = {k: gmm_models[k].bic(X_scaled) for k in gmm_models}
    best_gmm_k = min(gmm_bics, key=gmm_bics.get)
    best_gmm_score = gmm_scores[best_gmm_k]
    print(f"    Best GMM by BIC: K={best_gmm_k} "
          f"(BIC={gmm_bics[best_gmm_k]:.0f}, silhouette={best_gmm_score:.3f})")

    # --- K-Means (hard clustering, for comparison) ---
    kmeans_scores = {}
    kmeans_models = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        kmeans_scores[k] = score
        kmeans_models[k] = (km, labels)
        print(f"    K-Means K={k}: silhouette={score:.3f}")

    best_km_k = max(kmeans_scores, key=kmeans_scores.get)
    best_km_score = kmeans_scores[best_km_k]
    print(f"    Best K-Means: K={best_km_k} (silhouette={best_km_score:.3f})")

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

    # --- Pick best method ---
    # Prefer GMM if its silhouette is competitive (within 0.05 of best),
    # because soft labels are more informative
    all_scores = {"gmm": best_gmm_score, "kmeans": best_km_score}
    if dbscan_scores:
        best_eps = max(dbscan_scores, key=dbscan_scores.get)
        db_labels = dbscan_models[best_eps][1]
        outlier_ratio = (db_labels == -1).sum() / len(db_labels)
        if outlier_ratio <= 0.5:
            all_scores["dbscan"] = dbscan_scores[best_eps]

    overall_best = max(all_scores, key=all_scores.get)
    best_hard_score = all_scores[overall_best]

    # Use GMM if it's within 0.05 of the best hard method
    if best_gmm_score >= best_hard_score - 0.05:
        best_method = "gmm"
        best_gmm = gmm_models[best_gmm_k]
        best_labels = best_gmm.predict(X_scaled)
        best_score = best_gmm_score
        print(f"\n    Selected: GMM K={best_gmm_k} "
              f"(silhouette={best_score:.3f}, soft labels)")
    elif overall_best == "dbscan":
        best_method = "dbscan"
        best_labels = dbscan_models[best_eps][1]
        best_score = dbscan_scores[best_eps]
        print(f"\n    Selected: DBSCAN eps={best_eps} "
              f"(silhouette={best_score:.3f})")
    else:
        best_method = "kmeans"
        best_labels = kmeans_models[best_km_k][1]
        best_score = best_km_score
        print(f"\n    Selected: K-Means K={best_km_k} "
              f"(silhouette={best_score:.3f})")

    # --- Additional clustering metrics ---
    valid_mask = best_labels != -1
    if valid_mask.sum() > 0 and len(set(best_labels[valid_mask])) >= 2:
        db_score = davies_bouldin_score(X_scaled[valid_mask], best_labels[valid_mask])
        ch_score = calinski_harabasz_score(X_scaled[valid_mask], best_labels[valid_mask])
        print(f"\n    Clustering quality metrics:")
        print(f"      Silhouette:        {best_score:.3f}  (higher is better, range -1 to 1)")
        print(f"      Davies-Bouldin:    {db_score:.3f}  (lower is better)")
        print(f"      Calinski-Harabasz: {ch_score:.1f}  (higher is better)")
    else:
        db_score = None
        ch_score = None

    pair_features_df = pair_features_df.copy()
    pair_features_df["cluster"] = best_labels

    # --- Save soft probabilities if GMM was used ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if best_method == "gmm":
        probas = best_gmm.predict_proba(X_scaled)
        proba_df = pd.DataFrame(
            probas,
            columns=[f"cluster_{i}_prob" for i in range(best_gmm_k)],
        )
        # Add pair identifiers
        proba_df["person_a"] = pair_features_df["person_a"].values
        proba_df["person_b"] = pair_features_df["person_b"].values
        proba_df["primary_cluster"] = best_labels

        # Secondary cluster: second-highest probability
        proba_cols = [f"cluster_{i}_prob" for i in range(best_gmm_k)]
        proba_vals = probas.copy()
        # Zero out the primary to find secondary
        for idx in range(len(proba_vals)):
            proba_vals[idx, best_labels[idx]] = 0
        pair_features_df["secondary_cluster"] = proba_vals.argmax(axis=1)
        pair_features_df["primary_prob"] = probas[
            np.arange(len(probas)), best_labels
        ].round(3)
        pair_features_df["secondary_prob"] = proba_vals.max(axis=1).round(3)

        proba_df.to_csv(out / "cluster_probabilities.csv", index=False)

        # Report mixed-type pairs
        mixed_mask = pair_features_df["secondary_prob"] >= 0.25
        n_mixed = mixed_mask.sum()
        print(f"\n    Mixed-type pairs (secondary >= 25%): "
              f"{n_mixed} ({n_mixed/len(pair_features_df)*100:.1f}%)")

    # Save plots and metadata
    _plot_silhouette_scores(kmeans_scores, gmm_scores, output_dir)
    _plot_cluster_distribution(pair_features_df, output_dir)

    # Cluster profiles
    valid = pair_features_df[pair_features_df["cluster"] != -1]
    available_feat_cols = [c for c in FEATURE_COLS if c in valid.columns]
    profiles = valid.groupby("cluster")[available_feat_cols].mean().round(4)
    profiles["pair_count"] = valid.groupby("cluster").size()

    profiles.to_csv(out / "cluster_profiles.csv")
    _plot_cluster_heatmap(profiles[available_feat_cols], output_dir)

    print("\n    Cluster profiles:")
    summary_cols = ["avg_intimacy", "avg_warmth", "avg_sentiment",
                    "email_count", "time_span_days", "same_community", "pair_count"]
    available = [c for c in summary_cols if c in profiles.columns]
    print(profiles[available].to_string())

    # Save metadata
    meta = {
        "best_method": best_method,
        "best_score": best_score,
        "davies_bouldin": round(db_score, 3) if db_score is not None else None,
        "calinski_harabasz": round(ch_score, 1) if ch_score is not None else None,
        "kmeans_scores": {str(k): v for k, v in kmeans_scores.items()},
        "gmm_scores": {str(k): v for k, v in gmm_scores.items()},
        "dbscan_scores": {str(k): v for k, v in dbscan_scores.items()},
    }
    if best_method == "gmm":
        meta["best_k"] = best_gmm_k
        meta["n_mixed_pairs"] = int(n_mixed)
    elif best_method == "kmeans":
        meta["best_k"] = best_km_k
    with open(out / "clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return pair_features_df


def _plot_silhouette_scores(kmeans_scores, gmm_scores, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(kmeans_scores.keys())

    ax.plot(ks, [kmeans_scores[k] for k in ks], "o-",
            label="K-Means", color="#3498db")
    ax.plot(ks, [gmm_scores[k] for k in ks], "s-",
            label="GMM (soft)", color="#e74c3c")

    best_gmm_k = max(gmm_scores, key=gmm_scores.get)
    ax.axvline(x=best_gmm_k, color="#e74c3c", linestyle="--", alpha=0.4)

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
