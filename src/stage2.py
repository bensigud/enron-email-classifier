"""
Stage 2 — Pair-Level Feature Engineering

For every pair of executives who exchanged enough emails, builds a
feature vector drawing on three complementary frameworks:

  Ureña-Carrion, Saramäki & Kivelä (2020) — temporal communication
  features that go beyond raw contact counts:
    - Burstiness, inter-event regularity, temporal stability

  Backstrom & Kleinberg (2014) — structural dispersion from Facebook
  research, measuring how spread out mutual contacts are:
    - Dispersion (complement to shared-neighbour embeddedness)

  Ireland, Slatcher, Eastwick, Scissors, Finkel & Pennebaker (2011) —
  language style matching as a predictor of relationship closeness:
    - Style similarity, formality difference, pronoun-rate difference

Feature dimensions (24 features total):
  - Self-disclosure (3): mean, imbalance, variance — Reis & Shaver 1988
  - Responsiveness (3): mean, imbalance, variance — Reis & Shaver 1988
  - Sentiment (2): mean, variance — independent VADER signal
  - Intensity (4): count, direction, after-hours, direct vs CC
  - Temporal patterns (3): burstiness, regularity, stability
  - Duration (1)
  - Structural (3): community, shared neighbours, dispersion
  - Social distance (2): degree difference, PageRank ratio
  - Language style matching (3) — Ireland et al. 2011
"""

import re
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path


# Minimum emails between two people to include them as a pair
MIN_EMAILS_PER_PAIR = 5


def _compute_text_style(texts: list) -> dict:
    """
    Compute writing style metrics from a list of email texts.
    Returns averages across all texts for this person's direction.
    """
    if not texts:
        return {
            "avg_word_len": 0, "avg_sent_len": 0, "pronoun_rate": 0,
            "question_rate": 0, "exclamation_rate": 0, "greeting_formality": 0,
        }

    PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "yours",
                "we", "us", "our", "ours", "he", "she", "him", "her", "they", "them"}
    CASUAL_GREETINGS = {"hey", "hi", "hiya", "yo", "sup"}
    FORMAL_GREETINGS = {"dear", "greetings", "good morning", "good afternoon"}

    word_lens = []
    sent_lens = []
    pronoun_rates = []
    question_rates = []
    exclamation_rates = []
    formality_scores = []

    for text in texts:
        if not isinstance(text, str) or len(text.strip()) < 5:
            continue

        words = text.lower().split()
        if not words:
            continue

        # Average word length
        word_lens.append(np.mean([len(w) for w in words]))

        # Average sentence length (words per sentence)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            sent_lens.append(np.mean([len(s.split()) for s in sentences]))

        # Pronoun rate
        n_pronouns = sum(1 for w in words if w in PRONOUNS)
        pronoun_rates.append(n_pronouns / len(words))

        # Question rate (questions per sentence)
        n_questions = text.count("?")
        n_sents = max(len(sentences), 1)
        question_rates.append(n_questions / n_sents)

        # Exclamation rate
        n_excl = text.count("!")
        exclamation_rates.append(n_excl / n_sents)

        # Greeting formality: -1=casual, 0=none, +1=formal
        first_line = text.strip().split("\n")[0].lower().strip()
        if any(g in first_line for g in CASUAL_GREETINGS):
            formality_scores.append(-1)
        elif any(g in first_line for g in FORMAL_GREETINGS):
            formality_scores.append(1)
        else:
            formality_scores.append(0)

    def _safe_mean(lst):
        return np.mean(lst) if lst else 0.0

    return {
        "avg_word_len": _safe_mean(word_lens),
        "avg_sent_len": _safe_mean(sent_lens),
        "pronoun_rate": _safe_mean(pronoun_rates),
        "question_rate": _safe_mean(question_rates),
        "exclamation_rate": _safe_mean(exclamation_rates),
        "greeting_formality": _safe_mean(formality_scores),
    }


def _compute_style_matching(df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair, compute how similarly A and B write to each other.
    Lower difference = higher style matching = closer relationship.

    Based on Ireland et al. (2011) linguistic style matching.
    """
    print("  Computing language style matching features...")

    # Group all email texts by (sender, recipient)
    texts_by_direction = {}
    for _, row in df.iterrows():
        body = row["body"] if isinstance(row["body"], str) else ""
        sender = row["sender"]
        for recipient in row["recipients"]:
            if recipient != sender:
                key = (sender, recipient)
                if key not in texts_by_direction:
                    texts_by_direction[key] = []
                texts_by_direction[key].append(body)

    style_features = []

    for _, pair_row in pairs_df.iterrows():
        a, b = pair_row["person_a"], pair_row["person_b"]

        # Get texts A→B and B→A
        texts_a_to_b = texts_by_direction.get((a, b), [])
        texts_b_to_a = texts_by_direction.get((b, a), [])

        style_a = _compute_text_style(texts_a_to_b)
        style_b = _compute_text_style(texts_b_to_a)

        # Style matching = 1 - normalized absolute difference per metric
        # Lower difference = higher matching
        metrics = ["avg_word_len", "avg_sent_len", "pronoun_rate",
                   "question_rate", "exclamation_rate"]

        diffs = []
        for m in metrics:
            max_val = max(abs(style_a[m]), abs(style_b[m]), 0.001)
            diff = abs(style_a[m] - style_b[m]) / max_val
            diffs.append(min(diff, 1.0))  # cap at 1

        # Overall style similarity: 1 = identical style, 0 = completely different
        style_similarity = 1 - np.mean(diffs)

        # Greeting formality difference (do they greet each other the same way?)
        formality_diff = abs(style_a["greeting_formality"] - style_b["greeting_formality"])

        style_features.append({
            "person_a": a,
            "person_b": b,
            "style_similarity": round(style_similarity, 4),
            "formality_diff": round(formality_diff, 4),
            "pronoun_rate_diff": round(abs(style_a["pronoun_rate"] - style_b["pronoun_rate"]), 4),
        })

    result = pd.DataFrame(style_features)
    print(f"    Style matching: mean similarity = {result['style_similarity'].mean():.3f}")
    return result


def _compute_temporal_features(edge_df: pd.DataFrame,
                               pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal communication features per pair.
    Based on Ureña-Carrion, Saramäki & Kivelä (2020): temporal patterns
    outperform simple contact counts for estimating tie strength.

    Returns burstiness, inter-event regularity, and temporal stability.
    """
    print("  Computing temporal features (Ureña-Carrion et al. 2020)...")

    # Collect all timestamps per pair
    pair_dates = {}
    for _, row in edge_df.iterrows():
        if pd.isna(row["date"]):
            continue
        key = tuple(sorted([row["sender"], row["recipient"]]))
        if key not in pair_dates:
            pair_dates[key] = []
        pair_dates[key].append(row["date"])

    temporal_rows = []
    for _, pair_row in pairs_df.iterrows():
        a, b = pair_row["person_a"], pair_row["person_b"]
        key = tuple(sorted([a, b]))
        dates = sorted(pair_dates.get(key, []))

        if len(dates) < 3:
            temporal_rows.append({
                "person_a": a, "person_b": b,
                "burstiness": 0.0, "inter_event_regularity": 0.0,
                "temporal_stability": 0.0,
            })
            continue

        # Inter-event times in hours
        iets = [(dates[i+1] - dates[i]).total_seconds() / 3600
                for i in range(len(dates) - 1)]
        iets = [t for t in iets if t > 0]

        if not iets or np.mean(iets) == 0:
            temporal_rows.append({
                "person_a": a, "person_b": b,
                "burstiness": 0.0, "inter_event_regularity": 0.0,
                "temporal_stability": 0.0,
            })
            continue

        mean_iet = np.mean(iets)
        std_iet = np.std(iets)

        # Burstiness: (std - mean) / (std + mean), range [-1, 1]
        # +1 = very bursty, 0 = Poisson, -1 = perfectly regular
        burstiness = (std_iet - mean_iet) / (std_iet + mean_iet) if (std_iet + mean_iet) > 0 else 0

        # Inter-event regularity: coefficient of variation (inverted)
        # Higher = more regular communication pattern
        cv = std_iet / mean_iet if mean_iet > 0 else 0
        regularity = 1 / (1 + cv)  # maps to (0, 1]

        # Temporal stability: fraction of monthly windows with at least 1 email
        # Higher = relationship persists across time
        if dates:
            first_month = dates[0].to_period("M")
            last_month = dates[-1].to_period("M")
            total_months = max((last_month - first_month).n + 1, 1)
            active_months = len(set(d.to_period("M") for d in dates))
            stability = active_months / total_months
        else:
            stability = 0.0

        temporal_rows.append({
            "person_a": a, "person_b": b,
            "burstiness": round(burstiness, 4),
            "inter_event_regularity": round(regularity, 4),
            "temporal_stability": round(stability, 4),
        })

    result = pd.DataFrame(temporal_rows)
    print(f"    Temporal: mean burstiness={result['burstiness'].mean():.3f}, "
          f"stability={result['temporal_stability'].mean():.3f}")
    return result


def _compute_dispersion(graph: nx.DiGraph,
                        pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dispersion for each pair — Backstrom & Kleinberg (2014).
    Dispersion measures how spread out two people's mutual friends are.
    High dispersion = mutual friends come from different social circles
    = deeper, more personal relationship.
    """
    print("  Computing dispersion (Backstrom & Kleinberg 2014)...")

    undirected = graph.to_undirected()
    dispersion_rows = []

    for _, pair_row in pairs_df.iterrows():
        a, b = pair_row["person_a"], pair_row["person_b"]

        if not undirected.has_node(a) or not undirected.has_node(b):
            dispersion_rows.append({"person_a": a, "person_b": b, "dispersion": 0.0})
            continue

        common = set(undirected.neighbors(a)) & set(undirected.neighbors(b))
        common -= {a, b}

        if len(common) < 2:
            dispersion_rows.append({"person_a": a, "person_b": b, "dispersion": 0.0})
            continue

        # Count pairs of mutual friends that are NOT connected to each other
        common_list = list(common)
        disconnected_pairs = 0
        total_pairs = 0
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                total_pairs += 1
                if not undirected.has_edge(common_list[i], common_list[j]):
                    disconnected_pairs += 1

        disp = disconnected_pairs / total_pairs if total_pairs > 0 else 0.0
        dispersion_rows.append({"person_a": a, "person_b": b, "dispersion": round(disp, 4)})

    result = pd.DataFrame(dispersion_rows)
    print(f"    Dispersion: mean={result['dispersion'].mean():.3f}")
    return result


def build_pair_features(df: pd.DataFrame,
                        person_features_df: pd.DataFrame,
                        graph: nx.DiGraph = None) -> pd.DataFrame:
    """
    Build the feature vector for every qualifying pair.
    """
    print("  Building pair feature vectors...")

    # Explode recipients so each row = one sender → one recipient
    has_cc = "cc" in df.columns
    rows = []
    for _, row in df.iterrows():
        # Determine after-hours: weekends or before 7am / after 7pm
        is_after_hours = 0
        if pd.notna(row["date"]):
            hour = row["date"].hour
            weekday = row["date"].weekday()  # 0=Mon, 6=Sun
            if weekday >= 5 or hour < 7 or hour >= 19:
                is_after_hours = 1

        # Track who was CC'd vs directly addressed
        cc_set = set(row["cc"]) if has_cc and isinstance(row.get("cc"), list) else set()

        for recipient in row["recipients"]:
            if recipient != row["sender"]:
                rows.append({
                    "sender": row["sender"],
                    "recipient": recipient,
                    "intimacy_score": row["intimacy_score"],
                    "warmth_score": row["warmth_score"],
                    "sentiment_score": row["sentiment_score"],
                    "date": row["date"],
                    "after_hours": is_after_hours,
                    "is_direct": 1 if recipient not in cc_set else 0,
                })

    edge_df = pd.DataFrame(rows)

    # --- Directional aggregation (A→B separately from B→A) ---
    directional = (edge_df
                   .groupby(["sender", "recipient"])
                   .agg(
                       avg_intimacy=("intimacy_score", "mean"),
                       avg_warmth=("warmth_score", "mean"),
                       avg_sentiment=("sentiment_score", "mean"),
                       std_intimacy=("intimacy_score", "std"),
                       std_warmth=("warmth_score", "std"),
                       std_sentiment=("sentiment_score", "std"),
                       email_count=("intimacy_score", "count"),
                       first_date=("date", "min"),
                       last_date=("date", "max"),
                       after_hours_sum=("after_hours", "sum"),
                       direct_sum=("is_direct", "sum"),
                   )
                   .reset_index())

    directional = directional[directional["email_count"] >= 2]

    # --- Merge both directions ---
    rename_map = {
        "sender": "recipient",
        "recipient": "sender",
        "avg_intimacy": "avg_intimacy_reverse",
        "avg_warmth": "avg_warmth_reverse",
        "avg_sentiment": "avg_sentiment_reverse",
        "std_intimacy": "std_intimacy_reverse",
        "std_warmth": "std_warmth_reverse",
        "std_sentiment": "std_sentiment_reverse",
        "email_count": "email_count_reverse",
        "first_date": "first_date_reverse",
        "last_date": "last_date_reverse",
        "after_hours_sum": "after_hours_sum_reverse",
        "direct_sum": "direct_sum_reverse",
    }
    merged = directional.merge(
        directional.rename(columns=rename_map),
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

    # --- Build features ---
    person_feat = person_features_df.set_index("person")
    features = pd.DataFrame()
    features["person_a"] = merged["person_a"]
    features["person_b"] = merged["person_b"]

    # === SELF-DISCLOSURE dimension — Reis & Shaver (1988) ===
    features["avg_intimacy"] = (
        merged["avg_intimacy"] + merged["avg_intimacy_reverse"]) / 2
    features["intimacy_imbalance"] = abs(
        merged["avg_intimacy"] - merged["avg_intimacy_reverse"])
    features["intimacy_std"] = (
        merged["std_intimacy"].fillna(0) + merged["std_intimacy_reverse"].fillna(0)) / 2

    # === RESPONSIVENESS dimension — Reis & Shaver (1988) ===
    features["avg_warmth"] = (
        merged["avg_warmth"] + merged["avg_warmth_reverse"]) / 2
    features["warmth_imbalance"] = abs(
        merged["avg_warmth"] - merged["avg_warmth_reverse"])
    features["warmth_std"] = (
        merged["std_warmth"].fillna(0) + merged["std_warmth_reverse"].fillna(0)) / 2

    # === SENTIMENT (independent VADER signal) ===
    features["avg_sentiment"] = (
        merged["avg_sentiment"] + merged["avg_sentiment_reverse"]) / 2
    features["sentiment_std"] = (
        merged["std_sentiment"].fillna(0) + merged["std_sentiment_reverse"].fillna(0)) / 2

    # === INTENSITY dimension ===
    # Log-transform to prevent high-volume pairs from dominating clustering
    features["email_count"] = np.log1p(merged["email_count"])
    a_count = merged["email_count"] - merged["email_count_reverse"]
    features["direction_ratio"] = (a_count / merged["email_count"]).clip(0, 1)

    # === AFTER-HOURS RATIO (personal vs professional signal) ===
    total_after = merged["after_hours_sum"] + merged["after_hours_sum_reverse"]
    features["after_hours_ratio"] = (total_after / merged["email_count"]).fillna(0)

    # === CC vs DIRECT RATIO (intentional vs organizational contact) ===
    total_direct = merged["direct_sum"] + merged["direct_sum_reverse"]
    features["direct_ratio"] = (total_direct / merged["email_count"]).fillna(1)

    # === DURATION dimension ===
    earliest = pd.concat([
        merged["first_date"], merged["first_date_reverse"]
    ], axis=1).min(axis=1)
    latest = pd.concat([
        merged["last_date"], merged["last_date_reverse"]
    ], axis=1).max(axis=1)
    raw_days = (
        pd.to_datetime(latest) - pd.to_datetime(earliest)
    ).dt.days.fillna(0).clip(lower=0)
    # Log-transform to prevent dominating clustering (range 0-7000+ → 0-9)
    features["time_span_days"] = np.log1p(raw_days)

    # === TEMPORAL PATTERNS — Ureña-Carrion et al. (2020) ===
    temporal_df = _compute_temporal_features(edge_df, features)
    features = features.merge(temporal_df, on=["person_a", "person_b"], how="left")
    features["burstiness"] = features["burstiness"].fillna(0)
    features["inter_event_regularity"] = features["inter_event_regularity"].fillna(0)
    features["temporal_stability"] = features["temporal_stability"].fillna(0)

    # === STRUCTURAL dimension ===
    features["same_community"] = features.apply(
        lambda r: (
            1 if (r["person_a"] in person_feat.index and
                  r["person_b"] in person_feat.index and
                  person_feat.loc[r["person_a"], "community"] ==
                  person_feat.loc[r["person_b"], "community"])
            else 0
        ), axis=1)

    # Shared neighbors — how many people do both A and B email?
    if graph is not None:
        undirected = graph.to_undirected()
        features["shared_neighbors"] = features.apply(
            lambda r: len(set(undirected.neighbors(r["person_a"]))
                          & set(undirected.neighbors(r["person_b"])))
            if (undirected.has_node(r["person_a"]) and
                undirected.has_node(r["person_b"]))
            else 0,
            axis=1)
    else:
        features["shared_neighbors"] = 0

    # Dispersion — Backstrom & Kleinberg (2014)
    if graph is not None:
        disp_df = _compute_dispersion(graph, features)
        features = features.merge(disp_df, on=["person_a", "person_b"], how="left")
        features["dispersion"] = features["dispersion"].fillna(0)
    else:
        features["dispersion"] = 0

    # === SOCIAL DISTANCE dimension ===
    features["degree_difference"] = abs(
        features["person_a"].map(person_feat["total_degree"]).fillna(0) -
        features["person_b"].map(person_feat["total_degree"]).fillna(0))
    pr_a = features["person_a"].map(person_feat["pagerank"]).fillna(0)
    pr_b = features["person_b"].map(person_feat["pagerank"]).fillna(0)
    features["pagerank_ratio"] = pr_a / pr_b.clip(lower=1e-10)

    # === LANGUAGE STYLE MATCHING — Ireland et al. (2011) ===
    style_df = _compute_style_matching(df, features)
    features = features.merge(
        style_df, on=["person_a", "person_b"], how="left")
    features["style_similarity"] = features["style_similarity"].fillna(0.5)
    features["formality_diff"] = features["formality_diff"].fillna(0)
    features["pronoun_rate_diff"] = features["pronoun_rate_diff"].fillna(0)

    print(f"    Built features for {len(features):,} pairs")
    return features


FEATURE_COLS = [
    # Self-disclosure — Reis & Shaver 1988 (3)
    "avg_intimacy", "intimacy_imbalance", "intimacy_std",
    # Responsiveness — Reis & Shaver 1988 (3)
    "avg_warmth", "warmth_imbalance", "warmth_std",
    # Sentiment — independent VADER signal (2)
    "avg_sentiment", "sentiment_std",
    # Intensity (4)
    "email_count", "direction_ratio", "after_hours_ratio", "direct_ratio",
    # Temporal patterns — Ureña-Carrion et al. 2020 (3)
    "burstiness", "inter_event_regularity", "temporal_stability",
    # Duration (1)
    "time_span_days",
    # Structural — incl. dispersion, Backstrom & Kleinberg 2014 (3)
    "same_community", "shared_neighbors", "dispersion",
    # Social distance (2)
    "degree_difference", "pagerank_ratio",
    # Language style matching — Ireland et al. 2011 (3)
    "style_similarity", "formality_diff", "pronoun_rate_diff",
]


def flag_outlier_relationships(pair_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rare relationship types that clustering would miss due to
    low frequency. Uses score thresholds on pair features.

    Flags are additive — a pair can have multiple flags.
    Flagged pairs are still clustered, but their flags are preserved
    as an independent signal.

    Returns the DataFrame with a new 'flags' column (comma-separated
    string of flags, empty string if none).
    """
    print("  Flagging outlier relationships...")

    df = pair_features_df.copy()
    flags = [[] for _ in range(len(df))]

    # --- Romantic / deeply personal ---
    # Must be 3+ standard deviations above mean on BOTH scales
    if "avg_intimacy" in df.columns and "avg_warmth" in df.columns:
        int_threshold = df["avg_intimacy"].mean() + 3 * df["avg_intimacy"].std()
        warm_threshold = df["avg_warmth"].mean() + 3 * df["avg_warmth"].std()
        romantic_mask = (df["avg_intimacy"] > int_threshold) & (df["avg_warmth"] > warm_threshold)
        for i in df.index[romantic_mask]:
            flags[df.index.get_loc(i)].append("Romantic")
        n = romantic_mask.sum()
        if n > 0:
            print(f"    Romantic: {n} pairs (high disclosure + high responsiveness)")

    # --- Hostile / adversarial ---
    # Very low responsiveness AND negative sentiment
    if "avg_warmth" in df.columns and "avg_sentiment" in df.columns:
        warm_low = df["avg_warmth"].mean() - 3 * df["avg_warmth"].std()
        sent_low = df["avg_sentiment"].mean() - 3 * df["avg_sentiment"].std()
        hostile_mask = (df["avg_warmth"] < warm_low) & (df["avg_sentiment"] < sent_low)
        for i in df.index[hostile_mask]:
            flags[df.index.get_loc(i)].append("Hostile")
        n = hostile_mask.sum()
        if n > 0:
            print(f"    Hostile: {n} pairs (low responsiveness + negative sentiment)")

    # --- After-hours / personal ---
    # High after-hours ratio suggests personal relationship
    if "after_hours_ratio" in df.columns:
        personal_mask = df["after_hours_ratio"] > 0.4
        for i in df.index[personal_mask]:
            flags[df.index.get_loc(i)].append("After-hours")
        n = personal_mask.sum()
        if n > 0:
            print(f"    After-hours: {n} pairs (>40% outside work hours)")

    # --- One-sided / power dynamic ---
    # Very skewed direction ratio AND high social distance
    if "direction_ratio" in df.columns and "degree_difference" in df.columns:
        onesided_mask = (
            ((df["direction_ratio"] > 0.85) | (df["direction_ratio"] < 0.15)) &
            (df["degree_difference"] > df["degree_difference"].quantile(0.75))
        )
        for i in df.index[onesided_mask]:
            flags[df.index.get_loc(i)].append("Hierarchical")
        n = onesided_mask.sum()
        if n > 0:
            print(f"    Hierarchical: {n} pairs (one-sided + high degree difference)")

    # Build flags column
    df["flags"] = [", ".join(f) if f else "" for f in flags]

    n_flagged = (df["flags"] != "").sum()
    print(f"    Total: {n_flagged} pairs flagged "
          f"({n_flagged/len(df)*100:.1f}%)")

    return df


def save_results(pair_features_df: pd.DataFrame, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_features_df.to_csv(out / "relationship_pairs.csv", index=False)
    print(f"  Pair features saved to {output_dir}/relationship_pairs.csv")


def run_stage2(df: pd.DataFrame, person_features_df: pd.DataFrame,
               output_dir: str = "data/results",
               graph: nx.DiGraph = None) -> pd.DataFrame:
    """
    Full Stage 2 pipeline: build pair feature vectors.
    """
    print("\n=== STAGE 2: Pair-Level Feature Engineering ===")

    pair_features = build_pair_features(df, person_features_df, graph)
    save_results(pair_features, output_dir)

    print(f"\n  Stage 2 complete. {len(pair_features):,} pairs with "
          f"{len(FEATURE_COLS)} features each.")

    return pair_features
