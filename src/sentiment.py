import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Thresholds for relationship classification
FRIENDLY_THRESHOLD = 0.15    # avg sentiment above this = Friendly
HOSTILE_THRESHOLD = -0.10    # avg sentiment below this = Hostile
MENTORSHIP_IMBALANCE = 0.20  # one side much more positive than the other


def score_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """
    Score a single text using NLTK VADER.
    Returns a compound score from -1.0 (very negative) to +1.0 (very positive).

    VADER is great for short, informal text like emails — it understands
    things like capitalisation, exclamation marks and common phrases.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def score_all_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a sentiment score column to the emails DataFrame.
    """
    print("Scoring sentiment on all emails...")
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df["sentiment"] = df["body"].apply(lambda t: score_sentiment(t, analyzer))
    return df


def score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every pair of people (A, B) who exchanged at least 3 emails,
    compute:
      - avg_sentiment_ab: how positively A writes to B
      - avg_sentiment_ba: how positively B writes to A
      - relationship_class: Professional / Friendly / Hostile / Mentorship

    Returns a DataFrame with one row per pair.
    """
    print("Computing sentiment between pairs...")

    # Explode recipients so each row = one sender → one recipient
    rows = []
    for _, row in df.iterrows():
        for recipient in row["recipients"]:
            rows.append({
                "sender": row["sender"],
                "recipient": recipient,
                "sentiment": row["sentiment"],
            })

    pair_df = pd.DataFrame(rows)

    # Group by sender → recipient and average the sentiment
    grouped = (pair_df
               .groupby(["sender", "recipient"])
               .agg(avg_sentiment=("sentiment", "mean"),
                    email_count=("sentiment", "count"))
               .reset_index())

    # Only keep pairs with enough emails to be meaningful
    grouped = grouped[grouped["email_count"] >= 3]

    # Join both directions: A→B and B→A
    merged = grouped.merge(
        grouped.rename(columns={
            "sender": "recipient",
            "recipient": "sender",
            "avg_sentiment": "reverse_sentiment",
            "email_count": "reverse_count",
        }),
        on=["sender", "recipient"],
        how="inner",
    )

    # Only keep each pair once (A,B not both A→B and B→A)
    merged["pair_key"] = merged.apply(
        lambda r: tuple(sorted([r["sender"], r["recipient"]])), axis=1
    )
    merged = merged.drop_duplicates(subset="pair_key")

    # Classify the relationship
    merged["relationship_class"] = merged.apply(
        lambda r: classify_relationship(r["avg_sentiment"], r["reverse_sentiment"]),
        axis=1,
    )

    print(f"Scored {len(merged)} unique pairs")
    return merged


def classify_relationship(score_ab: float, score_ba: float) -> str:
    """
    Classify the relationship between two people based on their
    mutual sentiment scores.

    Rules:
    - Friendly: both sides positive
    - Hostile: at least one side clearly negative
    - Mentorship: one side clearly more positive (advice-giving direction)
    - Professional: everything else (neutral)
    """
    avg = (score_ab + score_ba) / 2
    diff = abs(score_ab - score_ba)

    if score_ab < HOSTILE_THRESHOLD or score_ba < HOSTILE_THRESHOLD:
        return "Hostile"
    elif diff >= MENTORSHIP_IMBALANCE:
        return "Mentorship"
    elif avg >= FRIENDLY_THRESHOLD:
        return "Friendly"
    else:
        return "Professional"


def plot_relationship_distribution(pairs_df: pd.DataFrame, output_path: str):
    """
    Bar chart showing how many pairs fall into each relationship class.
    """
    COLORS = {
        "Professional": "#3498db",
        "Friendly": "#2ecc71",
        "Hostile": "#e74c3c",
        "Mentorship": "#f39c12",
        "Romantic": "#e91e8c",
    }

    counts = pairs_df["relationship_class"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index,
                  counts.values,
                  color=[COLORS.get(c, "#aaa") for c in counts.index])

    ax.set_title("Relationship Classes Across Enron", fontsize=14)
    ax.set_xlabel("Relationship Type")
    ax.set_ylabel("Number of Pairs")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Relationship chart saved to {output_path}")


def save_results(pairs_df: pd.DataFrame, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(out / "sentiment_pairs.csv", index=False)
    print(f"Sentiment results saved to {output_dir}")


def run_sentiment_analysis(df: pd.DataFrame, output_dir: str = "data/results"):
    """
    Full pipeline: emails → sentiment scores → pair scores → classify → save.
    Call this from main.py.
    """
    print("\n=== Analysis 3: Friends & Enemies ===")
    df = score_all_emails(df)
    pairs_df = score_pairs(df)

    plot_relationship_distribution(pairs_df,
                                   f"{output_dir}/relationship_distribution.png")
    save_results(pairs_df, output_dir)

    print("\nRelationship class distribution:")
    print(pairs_df["relationship_class"].value_counts().to_string())
    print("\nMost hostile pairs:")
    hostile = (pairs_df[pairs_df["relationship_class"] == "Hostile"]
               .nsmallest(5, "avg_sentiment")[
                   ["sender", "recipient", "avg_sentiment", "reverse_sentiment"]
               ])
    print(hostile.to_string(index=False))

    print("\nMost friendly pairs:")
    friendly = (pairs_df[pairs_df["relationship_class"] == "Friendly"]
                .nlargest(5, "avg_sentiment")[
                    ["sender", "recipient", "avg_sentiment", "reverse_sentiment"]
                ])
    print(friendly.to_string(index=False))

    return df, pairs_df
