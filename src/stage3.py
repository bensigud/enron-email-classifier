"""
Stage 3 — Cluster Interpretation & Naming

Takes the clusters from Stage 2 and assigns human-readable names:
  1. Compute z-scores for each cluster centroid vs global mean
     (which features make each cluster distinctive?)
  2. Pull sample emails from each cluster
  3. Send cluster profiles + sample emails to Claude
  4. Claude names each cluster (e.g. "Professional", "Friendly", etc.)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.claude_client import client, MODEL


def compute_cluster_zscores(pair_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, compute how much each feature deviates from the
    global mean, measured in standard deviations (z-scores).

    A z-score of +2.0 on 'personal_imbalance' means that cluster has
    a personal_imbalance 2 standard deviations above average — that's
    a strong signal.

    Returns a DataFrame: rows = clusters, columns = features, values = z-scores.
    """
    feature_cols = [
        "avg_personal_score", "avg_sentiment", "personal_score_std",
        "sentiment_std", "personal_a_to_b", "personal_b_to_a",
        "personal_imbalance", "sentiment_a_to_b", "sentiment_b_to_a",
        "sentiment_imbalance", "email_count", "direction_ratio",
        "sender_degree", "recipient_degree", "degree_difference",
        "same_community", "pagerank_ratio",
    ]

    valid = pair_features_df[pair_features_df["cluster"] != -1]

    global_mean = valid[feature_cols].mean()
    global_std = valid[feature_cols].std().replace(0, 1)  # avoid division by zero

    cluster_means = valid.groupby("cluster")[feature_cols].mean()
    zscores = (cluster_means - global_mean) / global_std

    return zscores.round(3)


def get_top_features_per_cluster(zscores_df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    For each cluster, find the top features by absolute z-score.
    These are the features that most define this cluster.

    Returns dict: {cluster_id: [(feature, zscore), ...]}
    """
    result = {}
    for cluster_id in zscores_df.index:
        row = zscores_df.loc[cluster_id]
        top = row.abs().nlargest(top_n)
        result[cluster_id] = [
            (feat, row[feat]) for feat in top.index
        ]
    return result


def get_sample_emails_per_cluster(pair_features_df: pd.DataFrame,
                                  emails_df: pd.DataFrame,
                                  n_samples: int = 3) -> dict:
    """
    For each cluster, pick a representative pair and pull a few
    of their actual emails. This gives Claude real text to read
    when interpreting the cluster.

    Returns dict: {cluster_id: [email_body_1, email_body_2, ...]}
    """
    result = {}
    valid = pair_features_df[pair_features_df["cluster"] != -1]

    for cluster_id in sorted(valid["cluster"].unique()):
        cluster_pairs = valid[valid["cluster"] == cluster_id]

        # Pick the pair closest to the cluster centroid (most representative)
        # Simple proxy: pick the pair with the most emails
        best_pair = cluster_pairs.nlargest(1, "email_count").iloc[0]
        person_a = best_pair["person_a"]
        person_b = best_pair["person_b"]

        # Find their emails
        mask = (
            (emails_df["sender"] == person_a) &
            (emails_df["recipients"].apply(lambda r: person_b in r if isinstance(r, list) else False))
        ) | (
            (emails_df["sender"] == person_b) &
            (emails_df["recipients"].apply(lambda r: person_a in r if isinstance(r, list) else False))
        )
        matched = emails_df[mask]

        samples = []
        for _, row in matched.head(n_samples).iterrows():
            body = row["body"] if isinstance(row["body"], str) else ""
            # Truncate to 400 chars to keep costs down
            samples.append(body[:400].strip())

        result[cluster_id] = samples

    return result


def name_clusters_with_claude(zscores_df: pd.DataFrame,
                              top_features: dict,
                              sample_emails: dict,
                              cluster_profiles: pd.DataFrame) -> dict:
    """
    Send each cluster's profile to Claude and ask it to name the
    relationship type.

    Claude receives:
      - The cluster's average feature values
      - Which features are most distinctive (z-scores)
      - Sample emails from a representative pair

    Returns dict: {cluster_id: {"name": str, "description": str}}
    """
    print("  Asking Claude to name each cluster...")

    cluster_names = {}

    for cluster_id in sorted(zscores_df.index):
        # Build the profile summary
        profile = cluster_profiles.loc[cluster_id]
        top_feats = top_features.get(cluster_id, [])
        emails = sample_emails.get(cluster_id, [])

        feature_summary = "\n".join(
            f"    {feat}: z-score = {z:+.2f} ({'above' if z > 0 else 'below'} average)"
            for feat, z in top_feats
        )

        email_text = "\n---\n".join(emails) if emails else "(no sample emails available)"

        prompt = f"""You are analyzing relationship clusters from the Enron email dataset.

We clustered pairs of people based on 17 features (email tone, frequency, network position).
One cluster emerged with these characteristics:

CLUSTER {cluster_id} ({int(profile.get('pair_count', 0))} pairs):

Average feature values:
    avg_personal_score: {profile.get('avg_personal_score', 0):.3f}  (0=professional, 1=personal)
    avg_sentiment: {profile.get('avg_sentiment', 0):.3f}  (-1=negative, +1=positive)
    personal_imbalance: {profile.get('personal_imbalance', 0):.3f}  (0=balanced, high=one-sided)
    sentiment_imbalance: {profile.get('sentiment_imbalance', 0):.3f}  (0=balanced, high=one-sided)
    email_count: {profile.get('email_count', 0):.1f}  (average emails between pairs)
    direction_ratio: {profile.get('direction_ratio', 0):.3f}  (0.5=balanced, 1=one-way)
    same_community: {profile.get('same_community', 0):.2f}  (1=same group, 0=different groups)
    degree_difference: {profile.get('degree_difference', 0):.1f}  (difference in connectedness)

Most distinctive features (z-scores vs overall average):
{feature_summary}

Sample emails from a representative pair in this cluster:
{email_text}

Based on this data, give this cluster a relationship type name.
Choose from: Professional, Friendly, Hostile, Mentorship, Romantic, or suggest a better name if none fit.

Reply in exactly this format:
Name: [one or two word name]
Description: [one sentence explaining why]"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Parse the response
        name = "Unknown"
        description = ""
        for line in text.split("\n"):
            if line.lower().startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip()

        cluster_names[int(cluster_id)] = {
            "name": name,
            "description": description,
        }
        print(f"    Cluster {cluster_id} -> {name}")

    return cluster_names


def run_stage3(pair_features_df: pd.DataFrame,
               emails_df: pd.DataFrame,
               output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 3 pipeline:
      1. Compute z-scores per cluster
      2. Get sample emails per cluster
      3. Ask Claude to name each cluster
      4. Add cluster names to the pair DataFrame
      5. Save results

    Returns the pair DataFrame with a 'relationship_type' column added.
    """
    print("\n=== STAGE 3: Cluster Interpretation ===")

    out = Path(output_dir)

    # Load cluster profiles
    profiles_path = out / "cluster_profiles.csv"
    cluster_profiles = pd.read_csv(profiles_path, index_col="cluster")

    # Step 1 — Z-scores
    print("  Computing feature z-scores per cluster...")
    zscores = compute_cluster_zscores(pair_features_df)
    zscores.to_csv(out / "cluster_zscores.csv")
    print(f"    Z-scores saved to {output_dir}/cluster_zscores.csv")

    # Step 2 — Top features
    top_features = get_top_features_per_cluster(zscores)

    # Step 3 — Sample emails
    sample_emails = get_sample_emails_per_cluster(pair_features_df, emails_df)

    # Step 4 — Claude names the clusters
    cluster_names = name_clusters_with_claude(
        zscores, top_features, sample_emails, cluster_profiles
    )

    # Save cluster names
    with open(out / "cluster_names.json", "w") as f:
        json.dump(cluster_names, f, indent=2)
    print(f"    Cluster names saved to {output_dir}/cluster_names.json")

    # Step 5 — Add names to pair DataFrame
    name_map = {cid: info["name"] for cid, info in cluster_names.items()}
    pair_features_df = pair_features_df.copy()
    pair_features_df["relationship_type"] = pair_features_df["cluster"].map(name_map)

    # Update the saved results
    pair_features_df.to_csv(out / "relationship_pairs.csv", index=False)

    # Print summary
    print("\n    Cluster naming results:")
    for cid, info in sorted(cluster_names.items()):
        count = (pair_features_df["cluster"] == cid).sum()
        print(f"      Cluster {cid} -> {info['name']} ({count} pairs)")
        print(f"        {info['description']}")

    print(f"\n  Stage 3 complete.")
    return pair_features_df
