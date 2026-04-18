"""
Stage 1 — Email-Level Scoring

Scores every email on two dimensions:
  1. personal_score (0.0 to 1.0) — trained ML classifier
  2. sentiment_score (-1.0 to +1.0) — VADER sentiment

The personal classifier is the core AI component:
  - Claude labels 500 emails as professional/personal (binary)
  - We train 3 models (LogReg, SVM, Random Forest) and compare them
  - Best model is selected via 5-fold cross-validation
  - The model's predict_proba gives a continuous score, not a hard label
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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
    accuracy_score,
)

from src.claude_client import (
    label_email_batch,
    save_labeled_sample,
    load_labeled_sample,
)


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a sentiment_score column using NLTK VADER.

    VADER returns a compound score from -1.0 (very negative) to +1.0
    (very positive). It works well on short, informal text like emails
    because it understands capitalisation, exclamation marks, and
    common phrases.

    This score measures TONE (positive/negative), which is independent
    from the personal_score that measures CONTENT (professional/personal).
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


def train_personal_classifier(labeled_df: pd.DataFrame, output_dir: str) -> dict:
    """
    Train three classifiers on the Claude-labeled data and compare them.

    Models compared:
      - Logistic Regression: linear model, fast, interpretable
      - SVM (Support Vector Machine): finds optimal boundary between classes
      - Random Forest: ensemble of decision trees, handles non-linear patterns

    All three use TF-IDF to convert email text into numerical features.
    TF-IDF (Term Frequency-Inverse Document Frequency) weights words that
    are distinctive to a document — e.g. "miss you" appears in personal
    emails but rarely in professional ones, so it gets a high weight.

    We use 5-fold cross-validation: the data is split into 5 parts,
    the model trains on 4 and tests on 1, rotating through all 5 splits.
    This gives a more reliable estimate of performance than a single split.

    Returns a dict with the best pipeline, comparison metrics, and all results.
    """
    print("  Training and comparing classifiers...")

    X = labeled_df["body"].fillna("")
    y = labeled_df["claude_label"]

    # The TF-IDF settings are shared across all models for fair comparison
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),     # unigrams + bigrams (e.g. "miss you")
        stop_words="english",
        min_df=1,
    )

    # Define the three models we want to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,       # needed for predict_proba
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
    }

    # 5-fold cross-validation for each model
    # We must specify pos_label since our labels are strings, not 0/1
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label="personal", zero_division=0),
        "recall":    make_scorer(recall_score, pos_label="personal", zero_division=0),
        "f1":        make_scorer(f1_score, pos_label="personal", zero_division=0),
    }
    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", model),
        ])

        cv_scores = cross_validate(
            pipeline, X, y,
            cv=5,
            scoring=scoring,
            return_train_score=False,
        )

        results[name] = {
            "accuracy":  cv_scores["test_accuracy"].mean(),
            "precision": cv_scores["test_precision"].mean(),
            "recall":    cv_scores["test_recall"].mean(),
            "f1":        cv_scores["test_f1"].mean(),
        }

        print(f"    {name}: F1={results[name]['f1']:.3f}  "
              f"Acc={results[name]['accuracy']:.3f}")

    # Pick the best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  Best model: {best_name} (F1={results[best_name]['f1']:.3f})")

    # Helper to create a fresh model instance (models may have been mutated by CV)
    def _make_model(name):
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        elif name == "SVM":
            return SVC(kernel="linear", class_weight="balanced", probability=True)
        else:
            return RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                          random_state=42)

    # Do an 80/20 split to produce a confusion matrix for the report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    eval_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", min_df=2,
        )),
        ("clf", _make_model(best_name)),
    ])
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)

    print(f"\n  Classification report ({best_name} on test set):")
    print(classification_report(y_test, y_pred))

    # Now retrain the best model on ALL labeled data for production use
    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", min_df=2,
        )),
        ("clf", _make_model(best_name)),
    ])
    best_pipeline.fit(X, y)

    # Save confusion matrix plot
    _plot_confusion_matrix(y_test, y_pred, best_name, output_dir)
    _plot_model_comparison(results, output_dir)

    # Save comparison metrics as JSON for the report
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return {
        "pipeline": best_pipeline,
        "best_model": best_name,
        "results": results,
    }


def classify_all_emails(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Apply the trained classifier to ALL emails.

    For each email we get:
    - personal_score: probability (0.0 to 1.0) of being personal
      This is a continuous score, not a hard label.
      0.0 = clearly professional, 1.0 = clearly personal
    """
    print(f"  Classifying all {len(df):,} emails...")

    df = df.copy()
    texts = df["body"].fillna("")

    classes = list(pipeline.classes_)
    personal_index = classes.index("personal")
    probabilities = pipeline.predict_proba(texts)
    df["personal_score"] = probabilities[:, personal_index]

    # Summary stats
    mean_score = df["personal_score"].mean()
    high_personal = (df["personal_score"] > 0.5).sum()
    print(f"    Mean personal score: {mean_score:.3f}")
    print(f"    Emails scoring > 0.5: {high_personal:,} "
          f"({high_personal/len(df)*100:.1f}%)")

    return df


def _plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=["professional", "personal"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Professional", "Personal"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {output_dir}/confusion_matrix.png")


def _plot_model_comparison(results: dict, output_dir: str):
    """Bar chart comparing the three models across all metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved to {output_dir}/model_comparison.png")


def run_stage1(df: pd.DataFrame, output_dir: str = "data/results") -> pd.DataFrame:
    """
    Full Stage 1 pipeline:
      1. Get Claude labels (from disk if available, else call API)
      2. Train + compare 3 classifiers, pick best
      3. Apply best model to all emails -> personal_score
      4. Score sentiment with VADER -> sentiment_score
      5. Save results

    Returns the DataFrame with personal_score and sentiment_score columns added.
    """
    print("\n=== STAGE 1: Email-Level Scoring ===")

    # Step 1 — Claude labels
    labeled_df = load_labeled_sample(output_dir)
    if labeled_df is None:
        labeled_df = label_email_batch(df)
        save_labeled_sample(labeled_df, output_dir)

    # Step 2 — Train + compare classifiers
    classifier_result = train_personal_classifier(labeled_df, output_dir)

    # Step 3 — Classify all emails
    df = classify_all_emails(df, classifier_result["pipeline"])

    # Step 4 — VADER sentiment
    df = score_sentiment(df)

    # Step 5 — Save scored emails
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["message_id", "sender", "personal_score", "sentiment_score"]].to_csv(
        out / "email_scores.csv", index=False
    )

    print(f"\n  Stage 1 complete. Best model: {classifier_result['best_model']}")
    return df
