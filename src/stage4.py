"""
Stage 4 — Cluster Interpretation & Naming

Takes the clusters from Stage 3 and maps each to one of 9 relationship
types grounded in Kram & Isabella's (1985) workplace peer typology,
extended with categories from Sias & Cahill (1998).

Process:
  1. Compute z-scores for each cluster centroid vs global mean
  2. Pull sample emails from each cluster
  3. Send cluster profiles + sample emails to Claude
  4. Claude maps each cluster to the best-fitting relationship type
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
    Send ALL cluster profiles to Claude in a single prompt so it can
    see the full picture and assign unique, contrasting names.
    """
    print("  Asking Claude to name all clusters...")

    # Build one section per cluster
    cluster_sections = []
    for cluster_id in sorted(zscores_df.index):
        profile = cluster_profiles.loc[cluster_id]
        top_feats = top_features.get(cluster_id, [])
        emails = sample_emails.get(cluster_id, [])

        feature_summary = "\n".join(
            f"    {feat}: z-score = {z:+.2f} ({'above' if z > 0 else 'below'} average)"
            for feat, z in top_feats
        )

        email_text = "\n---\n".join(emails) if emails else "(no sample emails available)"

        section = f"""CLUSTER {cluster_id} ({int(profile.get('pair_count', 0))} pairs):

Average feature values:
    avg_intimacy: {profile.get('avg_intimacy', 0):.3f}  (0=no self-disclosure, 1=deeply personal — Reis & Shaver 1988)
    avg_warmth: {profile.get('avg_warmth', 0):.3f}  (0=hostile/cold, 1=deeply responsive/caring — Reis & Shaver 1988)
    avg_sentiment: {profile.get('avg_sentiment', 0):.3f}  (-1=very negative, +1=very positive, VADER)
    intimacy_imbalance: {profile.get('intimacy_imbalance', 0):.3f}  (0=mutual disclosure, high=one-sided)
    warmth_imbalance: {profile.get('warmth_imbalance', 0):.3f}  (0=mutual responsiveness, high=one-sided)
    intimacy_std: {profile.get('intimacy_std', 0):.3f}  (volatility of self-disclosure over time)
    warmth_std: {profile.get('warmth_std', 0):.3f}  (volatility of responsiveness over time)
    email_count: {profile.get('email_count', 0):.1f}  (average emails between pairs)
    time_span_days: {profile.get('time_span_days', 0):.0f}  (days between first and last email)
    direction_ratio: {profile.get('direction_ratio', 0):.3f}  (0.5=balanced, 1=one-way)
    burstiness: {profile.get('burstiness', 0):.3f}  (-1=perfectly regular, +1=very bursty)
    inter_event_regularity: {profile.get('inter_event_regularity', 0):.3f}  (0=irregular, 1=clockwork)
    temporal_stability: {profile.get('temporal_stability', 0):.3f}  (fraction of months with contact)
    same_community: {profile.get('same_community', 0):.2f}  (1=same group, 0=different groups)
    shared_neighbors: {profile.get('shared_neighbors', 0):.1f}  (people both of them email)
    dispersion: {profile.get('dispersion', 0):.3f}  (0=clustered mutual friends, 1=dispersed across circles)
    degree_difference: {profile.get('degree_difference', 0):.1f}  (difference in connectedness)

Most distinctive features (z-scores vs overall average):
{feature_summary}

Sample emails from a representative pair:
{email_text}"""
        cluster_sections.append(section)

    all_clusters = "\n\n" + ("=" * 50 + "\n\n").join(cluster_sections)

    prompt = f"""You are analyzing relationship clusters from the Enron email dataset.

We built pair-level features drawing on Reis & Shaver's (1988) interpersonal
process model (self-disclosure, responsiveness, emotional tone), temporal
communication patterns (Ureña-Carrion et al. 2020), and structural dispersion
(Backstrom & Kleinberg 2014) for each pair of people. Then we clustered them.

Here are ALL {len(cluster_sections)} clusters that emerged:
{all_clusters}

Map each cluster to the BEST matching relationship type from this list.
These types are adapted from Kram & Isabella's (1985) workplace peer typology
and Sias & Cahill (1998):

1. Transactional — surface-level info exchange, low disclosure, low warmth, neutral
2. Friendly Colleagues — moderate responsiveness, mostly work content, balanced, regular contact
3. Close — high disclosure + responsiveness + stability, real trust, personal sharing
4. Boss-Employee — high degree_difference, skewed direction, low disclosure, top-down
5. Mentor — high degree_difference + high warmth_imbalance (senior is more responsive), supportive
6. Romance — highest disclosure + responsiveness + after_hours_ratio, personal/emotional language
7. Tense / Conflict — low responsiveness, negative sentiment, volatile, friction
8. Fading — low temporal_stability, high burstiness, was active then went silent

Rules:
- You MUST pick from the 8 types above. Do not invent new names.
- Each cluster gets exactly one type.
- If two clusters fit the same type, pick the closer match and choose the next
  best type for the other.
- Use the feature values AND the sample emails together to decide.

Reply in exactly this format (one block per cluster, no extra text):
Cluster 0:
Name: [exact name from the list above]
Description: [one sentence explaining which features and email evidence led to this choice]

Cluster 1:
Name: [exact name from the list above]
Description: [one sentence explaining which features and email evidence led to this choice]

(and so on for each cluster)"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()

    # Parse response
    cluster_names = {}
    current_cluster = None
    current_name = None
    current_desc = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("cluster "):
            # Save previous cluster if any
            if current_cluster is not None and current_name:
                cluster_names[current_cluster] = {
                    "name": current_name,
                    "description": current_desc,
                }
            # Parse cluster ID
            try:
                cid = int(line.split(":")[0].replace("Cluster", "").replace("cluster", "").strip())
                current_cluster = cid
                current_name = None
                current_desc = ""
            except ValueError:
                pass
        elif line.lower().startswith("name:"):
            current_name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description:"):
            current_desc = line.split(":", 1)[1].strip()

    # Save the last cluster
    if current_cluster is not None and current_name:
        cluster_names[current_cluster] = {
            "name": current_name,
            "description": current_desc,
        }

    # Dedup: if Claude assigned the same name to multiple clusters,
    # keep the first and rename duplicates to next best unused type
    VALID_TYPES = [
        "Transactional", "Friendly Colleagues", "Close",
        "Boss-Employee", "Mentor", "Romance", "Tense / Conflict", "Fading",
    ]
    used_names = set()
    for cid in sorted(cluster_names):
        name = cluster_names[cid]["name"]
        if name in used_names:
            # Find an unused type
            for fallback in VALID_TYPES:
                if fallback not in used_names:
                    print(f"    WARNING: Duplicate '{name}' for cluster {cid} "
                          f"-> reassigned to '{fallback}'")
                    cluster_names[cid]["name"] = fallback
                    cluster_names[cid]["description"] += f" (reassigned from duplicate '{name}')"
                    name = fallback
                    break
        used_names.add(name)

    for cid in sorted(cluster_names):
        print(f"    Cluster {cid} -> {cluster_names[cid]['name']}")

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
