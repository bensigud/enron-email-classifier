import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.claude_client import (
    label_email_batch,
    save_labeled_sample,
    load_labeled_sample,
)


# Minimum number of emails between two people to consider their relationship
MIN_EMAILS_FOR_PAIR = 5

# Score above this = flag as romantic pair
ROMANTIC_PAIR_THRESHOLD = 0.25


def train_romance_classifier(labeled_df: pd.DataFrame):
    """
    Train a text classifier to detect romantic/personal emails.

    How it works:
    1. We take the emails that Claude already labeled (500 emails)
    2. We convert each email body into a bag-of-words using TF-IDF
       (TF-IDF = Term Frequency-Inverse Document Frequency — it turns
        text into numbers that represent which words are important)
    3. We train a Logistic Regression model on those numbers
    4. The model learns: "emails with words like 'miss you', 'dinner',
       'love' → romantic. Emails with 'contract', 'meeting', 'report'
       → professional"

    Returns the trained pipeline (TF-IDF + Logistic Regression together).
    """
    print("Training romance classifier...")

    # We simplify to binary: romantic/personal vs professional
    # This makes the problem easier and cleaner
    labeled_df = labeled_df.copy()
    labeled_df["label"] = labeled_df["claude_label"].apply(
        lambda x: "personal_romantic" if x in ("romantic", "personal") else "professional"
    )

    X = labeled_df["body"].fillna("")
    y = labeled_df["label"]

    # Split into training (80%) and testing (20%)
    # We test on 20% to check how accurate our model is
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # A Pipeline chains steps together:
    # Step 1: TfidfVectorizer — converts text to numbers
    # Step 2: LogisticRegression — trains the classifier
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,      # use the 5000 most common words
            ngram_range=(1, 2),     # also look at pairs of words (e.g. "miss you")
            stop_words="english",   # ignore common words like "the", "and", "is"
            min_df=2,               # ignore words that appear in fewer than 2 emails
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handles imbalanced classes (most emails are professional)
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on the test set so we know how well it works
    y_pred = pipeline.predict(X_test)
    print("\nClassifier performance on test set:")
    print(classification_report(y_test, y_pred))

    return pipeline


def classify_all_emails(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Apply the trained classifier to ALL emails (not just the 500 sample).

    For each email we get:
    - predicted label: "professional" or "personal_romantic"
    - romance_score: probability (0.0 to 1.0) of being personal/romantic
      Higher score = more likely to be romantic
    """
    print(f"Classifying all {len(df)} emails...")

    df = df.copy()
    texts = df["body"].fillna("")

    df["romance_label"] = pipeline.predict(texts)

    # predict_proba gives us probabilities for each class
    # We take the probability of the "personal_romantic" class
    classes = list(pipeline.classes_)
    romantic_index = classes.index("personal_romantic")
    probabilities = pipeline.predict_proba(texts)
    df["romance_score"] = probabilities[:, romantic_index]

    personal_count = (df["romance_label"] == "personal_romantic").sum()
    print(f"Found {personal_count} personal/romantic emails "
          f"({personal_count/len(df)*100:.1f}% of all emails)")

    return df


def find_romantic_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find pairs of people with an unusually high romantic/personal
    email score between them.

    For each pair (A, B) we:
    1. Collect all emails A sent to B and B sent to A
    2. Average their romance scores
    3. Flag the pair if the average score is above the threshold

    Returns a DataFrame of romantic pairs sorted by score.
    """
    print("Finding romantic pairs...")

    rows = []
    for _, row in df.iterrows():
        for recipient in row["recipients"]:
            rows.append({
                "sender": row["sender"],
                "recipient": recipient,
                "romance_score": row["romance_score"],
            })

    pair_df = pd.DataFrame(rows)

    # Normalise pair order so (A,B) and (B,A) are treated as the same pair
    pair_df["person_a"] = pair_df.apply(
        lambda r: min(r["sender"], r["recipient"]), axis=1
    )
    pair_df["person_b"] = pair_df.apply(
        lambda r: max(r["sender"], r["recipient"]), axis=1
    )

    # Average romance score per pair + count emails
    grouped = (pair_df
               .groupby(["person_a", "person_b"])
               .agg(avg_romance_score=("romance_score", "mean"),
                    email_count=("romance_score", "count"))
               .reset_index())

    # Only keep pairs with enough emails
    grouped = grouped[grouped["email_count"] >= MIN_EMAILS_FOR_PAIR]

    # Flag the romantic ones
    romantic_pairs = (grouped[grouped["avg_romance_score"] >= ROMANTIC_PAIR_THRESHOLD]
                      .sort_values("avg_romance_score", ascending=False))

    print(f"Found {len(romantic_pairs)} potentially romantic/personal pairs")
    return romantic_pairs


def plot_top_romantic_pairs(pairs_df: pd.DataFrame, output_path: str, top_n: int = 15):
    """
    Horizontal bar chart of the top romantic pairs by score.
    """
    top = pairs_df.head(top_n).copy()
    top["pair"] = top["person_a"].str.split("@").str[0] + " ↔ " + \
                  top["person_b"].str.split("@").str[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["pair"], top["avg_romance_score"], color="#e91e8c", alpha=0.8)
    ax.set_xlabel("Romance / Personal Score (0 = professional, 1 = very personal)")
    ax.set_title("Top Personal & Romantic Pairs at Enron", fontsize=13)
    ax.axvline(x=ROMANTIC_PAIR_THRESHOLD, color="grey",
               linestyle="--", label="Threshold")
    ax.legend()
    ax.invert_yaxis()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Romance chart saved to {output_path}")


def save_results(df: pd.DataFrame, pairs_df: pd.DataFrame, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["message_id", "sender", "romance_label", "romance_score"]].to_csv(
        out / "romance_scores.csv", index=False
    )
    pairs_df.to_csv(out / "romance_pairs.csv", index=False)
    print(f"Romance results saved to {output_dir}")


def run_romance_analysis(df: pd.DataFrame, output_dir: str = "data/results"):
    """
    Full pipeline: emails → Claude labels → train classifier →
    classify all → find pairs → plot → save.

    If Claude labels already exist on disk we skip the API calls
    (saves time and money on repeated runs).
    """
    print("\n=== Analysis 4: Office Romance ===")

    # Step 1 — get labels (from disk if available, else call Claude)
    labeled_df = load_labeled_sample(output_dir)
    if labeled_df is None:
        labeled_df = label_email_batch(df)
        save_labeled_sample(labeled_df, output_dir)

    # Step 2 — train classifier on Claude's labels
    pipeline = train_romance_classifier(labeled_df)

    # Step 3 — classify all 500k emails
    df = classify_all_emails(df, pipeline)

    # Step 4 — find romantic pairs
    pairs_df = find_romantic_pairs(df)

    # Step 5 — plot and save
    plot_top_romantic_pairs(pairs_df, f"{output_dir}/romance_pairs.png")
    save_results(df, pairs_df, output_dir)

    print("\nTop 10 most personal/romantic pairs:")
    top = pairs_df.head(10).copy()
    top["pair"] = (top["person_a"].str.split("@").str[0] + " ↔ " +
                   top["person_b"].str.split("@").str[0])
    print(top[["pair", "avg_romance_score", "email_count"]].to_string(index=False))

    return df, pairs_df, pipeline
