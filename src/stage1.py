"""
Stage 1 — Email-Level Scoring

Scores every email on three dimensions (mapping to Gilbert & Karahalios 2009):
  1. intimacy_score (0.0 to 1.0)  — ML classifier → Gilbert's "Intimacy" dimension
  2. warmth_score (0.0 to 1.0)    — ML classifier → Gilbert's "Emotional support" dimension
  3. sentiment_score (-1.0 to +1.0) — VADER       → backup emotional signal

Claude labels 500 emails on two scales (intimacy 1-5, warmth 1-5).
We train two separate ML classifiers, each comparing LogReg vs SVM vs RF.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nltk
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)

from src.claude_client import (
    label_email_batch,
    save_labeled_sample,
    load_labeled_sample,
)


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a sentiment_score column using NLTK VADER.
    Maps to Gilbert's "Emotional support" dimension as a backup signal.
    """
    print("  Scoring sentiment with VADER...")
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()

    def _score(text):
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    df["sentiment_score"] = df["body"].apply(_score)
    return df


def train_classifier(labeled_df: pd.DataFrame, scale_name: str,
                     label_col: str, output_dir: str) -> dict:
    """
    Train a binary classifier for one scale (intimacy or warmth).

    We convert the 1-5 labels to binary:
      - Low: 1-2
      - High: 4-5
      - Labels of 3 are excluded (ambiguous middle)

    This gives cleaner decision boundaries and works better with small data.

    Trains LogReg, SVM, RF with 5-fold CV and picks the best by F1.
    Returns dict with the best pipeline and comparison metrics.
    """
    print(f"  Training {scale_name} classifier...")

    df = labeled_df.copy()

    # Convert to binary (exclude 3s)
    df = df[df[label_col] != 3].copy()
    df["binary_label"] = df[label_col].apply(lambda x: "high" if x >= 4 else "low")

    X = df["body"].fillna("")
    y = df["binary_label"]

    print(f"    Samples: {len(df)} ({(y=='high').sum()} high, {(y=='low').sum()} low)")

    if len(df) < 20 or y.nunique() < 2:
        print(f"    WARNING: Not enough samples for {scale_name}. Skipping classifier.")
        return None

    # Models to compare
    def _make_models():
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced"),
            "SVM": SVC(
                kernel="linear", class_weight="balanced", probability=True),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42),
        }

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label="high", zero_division=0),
        "recall":    make_scorer(recall_score, pos_label="high", zero_division=0),
        "f1":        make_scorer(f1_score, pos_label="high", zero_division=0),
    }

    results = {}
    models = _make_models()

    for name, model in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2),
                stop_words="english", min_df=1)),
            ("clf", model),
        ])

        cv_scores = cross_validate(
            pipeline, X, y, cv=5, scoring=scoring, return_train_score=False)

        results[name] = {
            "accuracy":  cv_scores["test_accuracy"].mean(),
            "precision": cv_scores["test_precision"].mean(),
            "recall":    cv_scores["test_recall"].mean(),
            "f1":        cv_scores["test_f1"].mean(),
        }
        print(f"    {name}: F1={results[name]['f1']:.3f}  "
              f"Acc={results[name]['accuracy']:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"    Best: {best_name} (F1={results[best_name]['f1']:.3f})")

    # Confusion matrix on 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models2 = _make_models()
    eval_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", min_df=1)),
        ("clf", models2[best_name]),
    ])
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)

    print(f"\n    {scale_name} classification report ({best_name}):")
    print(classification_report(y_test, y_pred, zero_division=0))

    _plot_confusion_matrix(y_test, y_pred, f"{scale_name} — {best_name}", output_dir, scale_name)

    # Retrain on ALL data for production
    models3 = _make_models()
    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", min_df=1)),
        ("clf", models3[best_name]),
    ])
    best_pipeline.fit(X, y)

    # Save comparison
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"model_comparison_{scale_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return {
        "pipeline": best_pipeline,
        "best_model": best_name,
        "results": results,
    }


def classify_all_emails(df: pd.DataFrame, pipeline, score_col: str) -> pd.DataFrame:
    """
    Apply a trained classifier to ALL emails.
    Uses predict_proba to get a continuous score (0.0-1.0).
    """
    df = df.copy()
    texts = df["body"].fillna("")

    classes = list(pipeline.classes_)
    high_index = classes.index("high")
    probabilities = pipeline.predict_proba(texts)
    df[score_col] = probabilities[:, high_index]

    mean_score = df[score_col].mean()
    high_count = (df[score_col] > 0.5).sum()
    print(f"    {score_col}: mean={mean_score:.3f}, "
          f">{0.5}: {high_count:,} ({high_count/len(df)*100:.1f}%)")

    return df


def _plot_confusion_matrix(y_true, y_pred, title, output_dir, scale_name):
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=["low", "high"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "High"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title}", fontsize=12)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f"confusion_matrix_{scale_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_model_comparison(all_results: dict, output_dir: str):
    """Bar chart comparing models across both scales."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (scale_name, results) in enumerate(all_results.items()):
        if results is None:
            continue
        ax = axes[idx]
        metrics = ["accuracy", "precision", "recall", "f1"]
        model_names = list(results.keys())
        x = np.arange(len(metrics))
        width = 0.25

        for i, name in enumerate(model_names):
            values = [results[name][m] for m in metrics]
            bars = ax.bar(x + i * width, values, width, label=name)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel("Score")
        ax.set_title(f"{scale_name.capitalize()} Classifier", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.15)

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved to {output_dir}/model_comparison.png")


def run_stage1(df: pd.DataFrame, output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 1 pipeline:
      1. Get Claude labels (intimacy + warmth) from disk or API
      2. Train intimacy classifier (compare 3 models)
      3. Train warmth classifier (compare 3 models)
      4. Apply both to all emails
      5. Score sentiment with VADER
      6. Save results
    """
    print("\n=== STAGE 1: Email-Level Scoring ===")

    # Step 1 — Claude labels
    labeled_df = load_labeled_sample(output_dir)
    if labeled_df is None:
        labeled_df = label_email_batch(df)
        save_labeled_sample(labeled_df, output_dir)

    # Step 2 — Train intimacy classifier
    intimacy_result = train_classifier(
        labeled_df, "intimacy", "intimacy_label", output_dir)

    # Step 3 — Train warmth classifier
    warmth_result = train_classifier(
        labeled_df, "warmth", "warmth_label", output_dir)

    # Step 4 — Classify all emails
    print(f"\n  Classifying all {len(df):,} emails...")
    if intimacy_result:
        df = classify_all_emails(df, intimacy_result["pipeline"], "intimacy_score")
    else:
        df["intimacy_score"] = 0.5  # fallback

    if warmth_result:
        df = classify_all_emails(df, warmth_result["pipeline"], "warmth_score")
    else:
        df["warmth_score"] = 0.5  # fallback

    # Step 5 — VADER sentiment
    df = score_sentiment(df)

    # Step 6 — Plot comparison
    _plot_model_comparison({
        "intimacy": intimacy_result["results"] if intimacy_result else None,
        "warmth": warmth_result["results"] if warmth_result else None,
    }, output_dir)

    # Save scored emails
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["message_id", "sender", "intimacy_score", "warmth_score", "sentiment_score"]].to_csv(
        out / "email_scores.csv", index=False)

    print(f"\n  Stage 1 complete.")
    if intimacy_result:
        print(f"    Intimacy: {intimacy_result['best_model']}")
    if warmth_result:
        print(f"    Warmth: {warmth_result['best_model']}")

    return df
