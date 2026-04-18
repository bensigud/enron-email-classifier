"""
Stage 4 — Cluster Interpretation & Naming

Takes the clusters from Stage 3 and assigns human-readable names:
  1. Compute z-scores for each cluster centroid vs global mean
  2. Pull sample emails from each cluster
  3. Send cluster profiles + sample emails to Claude
  4. Claude names each cluster as a relationship type
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.claude_client import client, MODEL
from src.stage2 import FEATURE_COLS


def compute_cluster_zscores(pair_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, compute how much each feature deviates from the
    global mean in standard deviations (z-scores).
    """
    valid = pair_features_df[pair_features_df["cluster"] != -1]
    global_mean = valid[FEATURE_COLS].mean()
    global_std = valid[FEATURE_COLS].std().replace(0, 1)
    cluster_means = valid.groupby("cluster")[FEATURE_COLS].mean()
    zscores = (cluster_means - global_mean) / global_std
    return zscores.round(3)


def get_top_features_per_cluster(zscores_df: pd.DataFrame, top_n: int = 5) -> dict:
    """For each cluster, find the top features by absolute z-score."""
    result = {}
    for cluster_id in zscores_df.index:
        row = zscores_df.loc[cluster_id]
        top = row.abs().nlargest(top_n)
        result[cluster_id] = [(feat, row[feat]) for feat in top.index]
    return result


def get_sample_emails_per_cluster(pair_features_df: pd.DataFrame,
                                  emails_df: pd.DataFrame,
                                  n_samples: int = 3) -> dict:
    """Pull sample emails from a representative pair in each cluster."""
    result = {}
    valid = pair_features_df[pair_features_df["cluster"] != -1]

    for cluster_id in sorted(valid["cluster"].unique()):
        cluster_pairs = valid[valid["cluster"] == cluster_id]
        best_pair = cluster_pairs.nlargest(1, "email_count").iloc[0]
        person_a = best_pair["person_a"]
        person_b = best_pair["person_b"]

        mask = (
            (emails_df["sender"] == person_a) &
            (emails_df["recipients"].apply(
                lambda r: person_b in r if isinstance(r, list) else False))
        ) | (
            (emails_df["sender"] == person_b) &
            (emails_df["recipients"].apply(
                lambda r: person_a in r if isinstance(r, list) else False))
        )
        matched = emails_df[mask]

        samples = []
        for _, row in matched.head(n_samples).iterrows():
            body = row["body"] if isinstance(row["body"], str) else ""
            samples.append(body[:400].strip())

        result[cluster_id] = samples

    return result


def name_clusters_with_claude(zscores_df: pd.DataFrame,
                              top_features: dict,
                              sample_emails: dict,
                              cluster_profiles: pd.DataFrame) -> dict:
    """
    Send each cluster's profile to Claude and ask it to name the
    relationship type based on z-scores, feature averages, and sample emails.
    """
    print("  Asking Claude to name each cluster...")

    cluster_names = {}

    for cluster_id in sorted(zscores_df.index):
        profile = cluster_profiles.loc[cluster_id]
        top_feats = top_features.get(cluster_id, [])
        emails = sample_emails.get(cluster_id, [])

        feature_summary = "\n".join(
            f"    {feat}: z-score = {z:+.2f} ({'above' if z > 0 else 'below'} average)"
            for feat, z in top_feats
        )

        email_text = "\n---\n".join(emails) if emails else "(no sample emails available)"

        prompt = f"""You are analyzing relationship clusters from the Enron email dataset.

We used Gilbert & Karahalios's (2009) tie strength framework to build features
measuring Intimacy, Emotional Support, Intensity, Duration, Structural position,
and Social Distance for each pair of people. Then we clustered them.

One cluster emerged with these characteristics:

CLUSTER {cluster_id} ({int(profile.get('pair_count', 0))} pairs):

Average feature values:
    avg_intimacy: {profile.get('avg_intimacy', 0):.3f}  (0=formal business, 1=deeply personal)
    avg_warmth: {profile.get('avg_warmth', 0):.3f}  (0=hostile/cold, 1=loving/supportive)
    avg_sentiment: {profile.get('avg_sentiment', 0):.3f}  (-1=negative, +1=positive)
    intimacy_imbalance: {profile.get('intimacy_imbalance', 0):.3f}  (0=balanced, high=one-sided)
    warmth_imbalance: {profile.get('warmth_imbalance', 0):.3f}  (0=balanced, high=one-sided)
    email_count: {profile.get('email_count', 0):.1f}  (average emails between pairs)
    time_span_days: {profile.get('time_span_days', 0):.0f}  (days between first and last email)
    direction_ratio: {profile.get('direction_ratio', 0):.3f}  (0.5=balanced, 1=one-way)
    same_community: {profile.get('same_community', 0):.2f}  (1=same group, 0=different groups)
    degree_difference: {profile.get('degree_difference', 0):.1f}  (difference in connectedness)

Most distinctive features (z-scores vs overall average):
{feature_summary}

Sample emails from a representative pair in this cluster:
{email_text}

Based on this data, give this cluster a relationship type name.
Choose from: Professional, Friendly, Hostile, Mentorship, Romantic, Close Personal, Distant, or suggest a better name if none fit.

Reply in exactly this format:
Name: [one or two word name]
Description: [one sentence explaining why]"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

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


def run_stage4(pair_features_df: pd.DataFrame,
               emails_df: pd.DataFrame,
               output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 4 pipeline:
      1. Compute z-scores per cluster
      2. Get sample emails per cluster
      3. Ask Claude to name each cluster
      4. Add cluster names to the pair DataFrame
      5. Save results
    """
    print("\n=== STAGE 4: Cluster Interpretation ===")

    out = Path(output_dir)

    # Load cluster profiles
    cluster_profiles = pd.read_csv(out / "cluster_profiles.csv", index_col="cluster")

    # Step 1 — Z-scores
    print("  Computing feature z-scores per cluster...")
    zscores = compute_cluster_zscores(pair_features_df)
    zscores.to_csv(out / "cluster_zscores.csv")

    # Step 2 — Top features
    top_features = get_top_features_per_cluster(zscores)

    # Step 3 — Sample emails
    sample_emails = get_sample_emails_per_cluster(pair_features_df, emails_df)

    # Step 4 — Claude names
    cluster_names = name_clusters_with_claude(
        zscores, top_features, sample_emails, cluster_profiles)

    with open(out / "cluster_names.json", "w") as f:
        json.dump(cluster_names, f, indent=2)

    # Step 5 — Add names to pairs
    name_map = {cid: info["name"] for cid, info in cluster_names.items()}
    pair_features_df = pair_features_df.copy()
    pair_features_df["relationship_type"] = pair_features_df["cluster"].map(name_map)
    pair_features_df.to_csv(out / "relationship_pairs.csv", index=False)

    # Summary
    print("\n    Cluster naming results:")
    for cid, info in sorted(cluster_names.items()):
        count = (pair_features_df["cluster"] == cid).sum()
        print(f"      Cluster {cid} -> {info['name']} ({count} pairs)")
        print(f"        {info['description']}")

    print(f"\n  Stage 4 complete.")
    return pair_features_df
