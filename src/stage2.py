"""
Stage 2 — Pair-Level Feature Engineering

For every pair of executives who exchanged enough emails, builds a
20-feature vector mapping to 6 of Gilbert & Karahalios's 7 tie strength
dimensions:
  - Intimacy (4 features)
  - Emotional support (5 features)
  - Intensity (2 features)
  - Duration (1 feature)
  - Structural (1 feature)
  - Social distance (4 features)
  + Variance features (3 features)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Minimum emails between two people to include them as a pair
MIN_EMAILS_PER_PAIR = 5


def build_pair_features(df: pd.DataFrame,
                        person_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 20-feature vector for every qualifying pair.
    """
    print("  Building pair feature vectors...")

    # Explode recipients so each row = one sender → one recipient
    rows = []
    for _, row in df.iterrows():
        for recipient in row["recipients"]:
            if recipient != row["sender"]:
                rows.append({
                    "sender": row["sender"],
                    "recipient": recipient,
                    "intimacy_score": row["intimacy_score"],
                    "warmth_score": row["warmth_score"],
                    "sentiment_score": row["sentiment_score"],
                    "date": row["date"],
                })

    edge_df = pd.DataFrame(rows)

    # --- Directional aggregation (A→B separately from B→A) ---
    directional = (edge_df
                   .groupby(["sender", "recipient"])
                   .agg(
                       avg_intimacy=("intimacy_score", "mean"),
                       avg_warmth=("warmth_score", "mean"),
                       avg_sentiment=("sentiment_score", "mean"),
                       email_count=("intimacy_score", "count"),
                       first_date=("date", "min"),
                       last_date=("date", "max"),
                   )
                   .reset_index())

    directional = directional[directional["email_count"] >= 2]

    # --- Merge both directions ---
    merged = directional.merge(
        directional.rename(columns={
            "sender": "recipient",
            "recipient": "sender",
            "avg_intimacy": "avg_intimacy_reverse",
            "avg_warmth": "avg_warmth_reverse",
            "avg_sentiment": "avg_sentiment_reverse",
            "email_count": "email_count_reverse",
            "first_date": "first_date_reverse",
            "last_date": "last_date_reverse",
        }),
        on=["sender", "recipient"],
        how="inner",
    )

    # De-duplicate
    merged["person_a"] = merged.apply(
        lambda r: min(r["sender"], r["recipient"]), axis=1)
    merged["person_b"] = merged.apply(
        lambda r: max(r["sender"], r["recipient"]), axis=1)
    merged = merged.drop_duplicates(subset=["person_a", "person_b"])

    merged["email_count"] = merged["email_count"] + merged["email_count_reverse"]
    merged = merged[merged["email_count"] >= MIN_EMAILS_PER_PAIR]

    # Compute per-pair std
    pair_stats = _compute_pair_stats(edge_df, merged)
    merged = merged.merge(pair_stats, on=["person_a", "person_b"], how="left")

    # --- Build features ---
    person_feat = person_features_df.set_index("person")
    features = pd.DataFrame()
    features["person_a"] = merged["person_a"]
    features["person_b"] = merged["person_b"]

    # === INTIMACY dimension (Gilbert #1) ===
    features["avg_intimacy"] = (
        merged["avg_intimacy"] + merged["avg_intimacy_reverse"]) / 2
    features["intimacy_a_to_b"] = merged["avg_intimacy"]
    features["intimacy_b_to_a"] = merged["avg_intimacy_reverse"]
    features["intimacy_imbalance"] = abs(
        merged["avg_intimacy"] - merged["avg_intimacy_reverse"])

    # === EMOTIONAL SUPPORT dimension (Gilbert #5) ===
    features["avg_warmth"] = (
        merged["avg_warmth"] + merged["avg_warmth_reverse"]) / 2
    features["warmth_a_to_b"] = merged["avg_warmth"]
    features["warmth_b_to_a"] = merged["avg_warmth_reverse"]
    features["warmth_imbalance"] = abs(
        merged["avg_warmth"] - merged["avg_warmth_reverse"])
    features["avg_sentiment"] = (
        merged["avg_sentiment"] + merged["avg_sentiment_reverse"]) / 2

    # === INTENSITY dimension (Gilbert #2) ===
    features["email_count"] = merged["email_count"]
    a_count = merged["email_count"] - merged["email_count_reverse"]
    features["direction_ratio"] = (a_count / merged["email_count"]).clip(0, 1)

    # === DURATION dimension (Gilbert #3) ===
    earliest = pd.concat([
        merged["first_date"], merged["first_date_reverse"]
    ], axis=1).min(axis=1)
    latest = pd.concat([
        merged["last_date"], merged["last_date_reverse"]
    ], axis=1).max(axis=1)
    features["time_span_days"] = (
        pd.to_datetime(latest) - pd.to_datetime(earliest)
    ).dt.days.fillna(0).clip(lower=0)

    # === STRUCTURAL dimension (Gilbert #4) ===
    features["same_community"] = features.apply(
        lambda r: (
            1 if (r["person_a"] in person_feat.index and
                  r["person_b"] in person_feat.index and
                  person_feat.loc[r["person_a"], "community"] ==
                  person_feat.loc[r["person_b"], "community"])
            else 0
        ), axis=1)

    # === SOCIAL DISTANCE dimension (Gilbert #6) ===
    features["sender_degree"] = features["person_a"].map(
        person_feat["total_degree"]).fillna(0)
    features["recipient_degree"] = features["person_b"].map(
        person_feat["total_degree"]).fillna(0)
    features["degree_difference"] = abs(
        features["sender_degree"] - features["recipient_degree"])
    pr_a = features["person_a"].map(person_feat["pagerank"]).fillna(0)
    pr_b = features["person_b"].map(person_feat["pagerank"]).fillna(0)
    features["pagerank_ratio"] = pr_a / pr_b.clip(lower=1e-10)

    # === VARIANCE features ===
    features["intimacy_std"] = merged["intimacy_std"].fillna(0)
    features["warmth_std"] = merged["warmth_std"].fillna(0)
    features["sentiment_std"] = merged["sentiment_std"].fillna(0)

    print(f"    Built features for {len(features):,} pairs")
    return features


def _compute_pair_stats(edge_df: pd.DataFrame,
                        merged: pd.DataFrame) -> pd.DataFrame:
    """Compute std of scores per pair."""
    edge_df = edge_df.copy()
    edge_df["person_a"] = edge_df.apply(
        lambda r: min(r["sender"], r["recipient"]), axis=1)
    edge_df["person_b"] = edge_df.apply(
        lambda r: max(r["sender"], r["recipient"]), axis=1)

    pair_keys = set(zip(merged["person_a"], merged["person_b"]))
    mask = edge_df.apply(
        lambda r: (r["person_a"], r["person_b"]) in pair_keys, axis=1)
    filtered = edge_df[mask]

    stats = (filtered
             .groupby(["person_a", "person_b"])
             .agg(
                 intimacy_std=("intimacy_score", "std"),
                 warmth_std=("warmth_score", "std"),
                 sentiment_std=("sentiment_score", "std"),
             )
             .reset_index())

    return stats


FEATURE_COLS = [
    "avg_intimacy", "intimacy_a_to_b", "intimacy_b_to_a", "intimacy_imbalance",
    "avg_warmth", "warmth_a_to_b", "warmth_b_to_a", "warmth_imbalance",
    "avg_sentiment",
    "email_count", "direction_ratio",
    "time_span_days",
    "same_community",
    "sender_degree", "recipient_degree", "degree_difference", "pagerank_ratio",
    "intimacy_std", "warmth_std", "sentiment_std",
]


def save_results(pair_features_df: pd.DataFrame, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_features_df.to_csv(out / "relationship_pairs.csv", index=False)
    print(f"  Pair features saved to {output_dir}/relationship_pairs.csv")


def run_stage2(df: pd.DataFrame, person_features_df: pd.DataFrame,
               output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 2 pipeline: build pair feature vectors.
    """
    print("\n=== STAGE 2: Pair-Level Feature Engineering ===")

    pair_features = build_pair_features(df, person_features_df)
    save_results(pair_features, output_dir)

    print(f"\n  Stage 2 complete. {len(pair_features):,} pairs with "
          f"{len(FEATURE_COLS)} features each.")

    return pair_features
