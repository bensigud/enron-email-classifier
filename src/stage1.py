"""
Stage 1 — Email-Level Scoring

Scores every email on three dimensions grounded in Reis & Shaver's
Interpersonal Process Model of Intimacy (1988; validated by Laurenceau,
Barrett & Pietromonaco 1998):

  1. intimacy_score (0.0 to 1.0) — Self-disclosure: does the email share
     personal information, feelings, or experiences? (binary classifier)
  2. warmth_score (0.0 to 1.0)   — Responsiveness: does the email show
     understanding, validation, or caring toward the recipient? (regressor)
  3. sentiment_score (-1.0 to 1.0) — Emotional tone: independent affective
     quality of the exchange (VADER rule-based)

Hybrid approach:
  - Self-disclosure uses CLASSIFICATION (binary: low vs high) because the
    data splits cleanly into formal (1-2) vs personal (3-5) with enough samples.
  - Responsiveness uses REGRESSION (predict 1-5 directly) because most
    corporate emails are neutral (3), making binary splits either imbalanced
    or lossy.
  - VADER provides an independent rule-based emotional tone signal.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    r2_score,
    mean_absolute_error,
    mean_squared_error,
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

    This measures TONE (positive/negative), which is independent from
    the warmth_score that measures supportiveness, and from the
    intimacy_score that measures personal content.
    """
    print("  Scoring sentiment with VADER...")
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()

    def _score(text):
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    df["sentiment_score"] = df["body"].apply(_score)

    mean_s = df["sentiment_score"].mean()
    std_s = df["sentiment_score"].std()
    pos = (df["sentiment_score"] > 0.05).sum()
    neg = (df["sentiment_score"] < -0.05).sum()
    print(f"    sentiment_score: mean={mean_s:.3f}, std={std_s:.3f}, "
          f"positive={pos:,}, negative={neg:,}")

    return df


def train_classifier(labeled_df: pd.DataFrame, scale_name: str,
                     label_col: str, low_range: list, high_range: list,
                     output_dir: str) -> dict:
    """
    Train a binary classifier for one scale (intimacy or warmth).

    Converts 1-5 labels to binary using the provided ranges, then
    trains LogReg, SVM, RF with 5-fold CV. Uses predict_proba to
    get continuous scores (0.0 to 1.0).
    """
    print(f"  Training {scale_name} classifier...")

    df = labeled_df.copy()

    # Convert to binary using provided ranges
    df = df[df[label_col].isin(low_range + high_range)].copy()
    df["binary_label"] = df[label_col].apply(
        lambda x: "high" if x in high_range else "low"
    )

    X = df["body"].fillna("")
    y = df["binary_label"]

    n_high = (y == "high").sum()
    n_low = (y == "low").sum()
    print(f"    Samples: {len(df)} ({n_high} high, {n_low} low)")

    if len(df) < 20 or y.nunique() < 2:
        print(f"    WARNING: Not enough samples for {scale_name}. Skipping.")
        return None

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
    cv_f1_scores = {}  # Store raw fold scores for significance testing
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

        cv_f1_scores[name] = cv_scores["test_f1"]

        results[name] = {
            "accuracy":  cv_scores["test_accuracy"].mean(),
            "precision": cv_scores["test_precision"].mean(),
            "recall":    cv_scores["test_recall"].mean(),
            "f1":        cv_scores["test_f1"].mean(),
            "f1_std":    float(cv_scores["test_f1"].std()),
            "accuracy_std": float(cv_scores["test_accuracy"].std()),
        }
        print(f"    {name}: F1={results[name]['f1']:.3f} ± {results[name]['f1_std']:.3f}  "
              f"Acc={results[name]['accuracy']:.3f} ± {results[name]['accuracy_std']:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"    Best: {best_name} (F1={results[best_name]['f1']:.3f} ± {results[best_name]['f1_std']:.3f})")

    # Statistical significance: paired t-test between best and second-best
    from scipy.stats import ttest_rel
    sorted_models = sorted(results, key=lambda k: results[k]["f1"], reverse=True)
    if len(sorted_models) >= 2:
        second_name = sorted_models[1]
        t_stat, p_value = ttest_rel(cv_f1_scores[best_name], cv_f1_scores[second_name])
        sig = "significant" if p_value < 0.05 else "NOT significant"
        print(f"    Significance ({best_name} vs {second_name}): "
              f"t={t_stat:.3f}, p={p_value:.3f} ({sig})")
        results["_significance"] = {
            "best": best_name,
            "second": second_name,
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 3),
            "significant": bool(p_value < 0.05),
        }

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

    _plot_confusion_matrix(y_test, y_pred,
                           f"{scale_name} — {best_name}", output_dir, scale_name)

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
    std_score = df[score_col].std()
    high_count = (df[score_col] > 0.5).sum()
    print(f"    {score_col}: mean={mean_score:.3f}, std={std_score:.3f}, "
          f">{0.5}: {high_count:,} ({high_count/len(df)*100:.1f}%)")

    return df


def train_regressor(labeled_df: pd.DataFrame, scale_name: str,
                    label_col: str, output_dir: str) -> dict:
    """
    Train a regressor for one scale (warmth).

    Uses the raw 1-5 labels directly — no binarization needed.
    Trains Ridge, SVR, RF Regressor with 5-fold CV.
    Output is normalized to 0.0-1.0.
    """
    print(f"  Training {scale_name} regressor...")

    df = labeled_df.copy()
    df = df[df[label_col].notna()].copy()

    X = df["body"].fillna("")
    y = df[label_col].astype(float)

    print(f"    Samples: {len(df)}")
    print(f"    Label distribution:\n{y.value_counts().sort_index().to_string()}")

    if len(df) < 20:
        print(f"    WARNING: Not enough samples for {scale_name}. Skipping.")
        return None

    def _make_models():
        return {
            "Ridge Regression": Ridge(alpha=1.0),
            "SVR": SVR(kernel="linear", C=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42),
        }

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }

    results = {}
    cv_r2_scores = {}  # Store raw fold scores for significance testing
    models = _make_models()

    for name, model in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2),
                stop_words="english", min_df=1)),
            ("reg", model),
        ])

        cv_scores = cross_validate(
            pipeline, X, y, cv=5, scoring=scoring, return_train_score=False)

        cv_r2_scores[name] = cv_scores["test_r2"]

        results[name] = {
            "r2":       cv_scores["test_r2"].mean(),
            "r2_std":   float(cv_scores["test_r2"].std()),
            "mae":      -cv_scores["test_neg_mae"].mean(),
            "mae_std":  float(cv_scores["test_neg_mae"].std()),
            "rmse":     -cv_scores["test_neg_rmse"].mean(),
        }
        print(f"    {name}: R²={results[name]['r2']:.3f} ± {results[name]['r2_std']:.3f}  "
              f"MAE={results[name]['mae']:.3f} ± {results[name]['mae_std']:.3f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"    Best: {best_name} (R²={results[best_name]['r2']:.3f} ± {results[best_name]['r2_std']:.3f})")

    # Statistical significance: paired t-test between best and second-best
    from scipy.stats import ttest_rel
    sorted_models = sorted(results, key=lambda k: results[k]["r2"], reverse=True)
    if len(sorted_models) >= 2:
        second_name = sorted_models[1]
        t_stat, p_value = ttest_rel(cv_r2_scores[best_name], cv_r2_scores[second_name])
        sig = "significant" if p_value < 0.05 else "NOT significant"
        print(f"    Significance ({best_name} vs {second_name}): "
              f"t={t_stat:.3f}, p={p_value:.3f} ({sig})")
        results["_significance"] = {
            "best": best_name,
            "second": second_name,
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 3),
            "significant": bool(p_value < 0.05),
        }

    # Scatter plot on 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models2 = _make_models()
    eval_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", min_df=1)),
        ("reg", models2[best_name]),
    ])
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n    {scale_name} test set ({best_name}): R²={r2:.3f}, MAE={mae:.3f}")

    _plot_regression_scatter(y_test, y_pred,
                             f"{scale_name} — {best_name}", output_dir, scale_name)

    # Retrain on ALL data for production
    models3 = _make_models()
    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", min_df=1)),
        ("reg", models3[best_name]),
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


def regress_all_emails(df: pd.DataFrame, pipeline, score_col: str) -> pd.DataFrame:
    """
    Apply a trained regressor to ALL emails.
    Predicts 1-5, then normalizes to 0.0-1.0.
    """
    df = df.copy()
    texts = df["body"].fillna("")

    raw_scores = pipeline.predict(texts)
    # Normalize from 1-5 scale to 0.0-1.0
    df[score_col] = ((raw_scores - 1) / 4).clip(0, 1)

    mean_score = df[score_col].mean()
    std_score = df[score_col].std()
    high_count = (df[score_col] > 0.5).sum()
    print(f"    {score_col}: mean={mean_score:.3f}, std={std_score:.3f}, "
          f">{0.5}: {high_count:,} ({high_count/len(df)*100:.1f}%)")

    return df


def _plot_regression_scatter(y_true, y_pred, title, output_dir, scale_name):
    """Predicted vs actual scatter plot for regression evaluation."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.5)
    ax.plot([1, 5], [1, 5], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual (Claude label)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {title}", fontsize=12)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.legend()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f"scatter_{scale_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, title, output_dir, scale_name):
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
    """
    Plot model comparison for both scales.
    Handles both classification metrics (intimacy) and regression metrics (warmth).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (scale_name, info) in enumerate(all_results.items()):
        results = info["results"]
        mode = info["mode"]
        if results is None:
            continue

        ax = axes[idx]
        model_names = [k for k in results.keys() if not k.startswith("_")]

        if mode == "classification":
            metrics = ["accuracy", "precision", "recall", "f1"]
            ylabel = "Score"
            title = f"{scale_name.capitalize()} Classifier"
            ylim = (0, 1.15)
        else:
            metrics = ["r2", "mae", "rmse"]
            ylabel = "Score"
            title = f"{scale_name.capitalize()} Regressor"
            ylim = None

        x = np.arange(len(metrics))
        width = 0.25

        for i, name in enumerate(model_names):
            values = [results[name][m] for m in metrics]
            bars = ax.bar(x + i * width, values, width, label=name)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() if m in ("r2", "mae", "rmse") else m.capitalize()
                            for m in metrics])
        ax.legend(fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved to {output_dir}/model_comparison.png")


def _save_models(intimacy_result, warmth_result, output_dir):
    """Save trained models to disk so Full Run can load them without retraining."""
    import pickle
    out = Path(output_dir)
    if intimacy_result:
        with open(out / "model_intimacy.pkl", "wb") as f:
            pickle.dump(intimacy_result["pipeline"], f)
        print(f"    Saved intimacy model: {intimacy_result['best_model']}")
    if warmth_result:
        with open(out / "model_warmth.pkl", "wb") as f:
            pickle.dump(warmth_result["pipeline"], f)
        print(f"    Saved warmth model: {warmth_result['best_model']}")


def _load_models(output_dir):
    """Load previously trained models from disk."""
    import pickle
    out = Path(output_dir)
    intimacy_pipeline = None
    warmth_pipeline = None

    intimacy_path = out / "model_intimacy.pkl"
    warmth_path = out / "model_warmth.pkl"

    if intimacy_path.exists():
        with open(intimacy_path, "rb") as f:
            intimacy_pipeline = pickle.load(f)
        print(f"    Loaded intimacy model from disk")
    if warmth_path.exists():
        with open(warmth_path, "rb") as f:
            warmth_pipeline = pickle.load(f)
        print(f"    Loaded warmth model from disk")

    return intimacy_pipeline, warmth_pipeline


def run_stage1(df: pd.DataFrame, output_dir: str = "data/results",
               mode: str = "train") -> pd.DataFrame:
    """
    Stage 1 pipeline with two modes:

    mode="train" (Sample runs):
      1. Label emails with Claude (or load cached labels)
      2. Merge human labels (override Claude where humans disagree)
      3. Train classifiers + regressors
      4. Save trained models to disk
      5. Score all emails in this batch
      6. Report metrics + comparisons

    mode="predict" (Full Run):
      1. Load saved models from disk (no training, no Claude calls)
      2. Score all emails
      3. VADER sentiment
    """
    print(f"\n=== STAGE 1: Email-Level Scoring ({mode.upper()} mode) ===")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if mode == "predict":
        # --- PREDICT MODE: load saved models, score everything ---
        print("  Loading trained models...")
        intimacy_pipeline, warmth_pipeline = _load_models(output_dir)

        if intimacy_pipeline is None or warmth_pipeline is None:
            print("  WARNING: No saved models found. Falling back to train mode.")
            return run_stage1(df, output_dir, mode="train")

        print(f"\n  Scoring all {len(df):,} emails with saved models...")
        if intimacy_pipeline:
            df = classify_all_emails(df, intimacy_pipeline, "intimacy_score")
        else:
            df["intimacy_score"] = 0.5

        if warmth_pipeline:
            df = regress_all_emails(df, warmth_pipeline, "warmth_score")
        else:
            df["warmth_score"] = 0.5

        df = score_sentiment(df)

        # Save scored emails
        df[["message_id", "sender", "intimacy_score", "warmth_score",
            "sentiment_score"]].to_csv(out / "email_scores.csv", index=False)

        print(f"\n  Stage 1 complete (predict mode). {len(df):,} emails scored.")
        return df

    # --- TRAIN MODE: label, train, save models, score ---

    # Step 1 — Claude labels
    labeled_df = load_labeled_sample(output_dir)
    if labeled_df is None:
        labeled_df = label_email_batch(df)
        save_labeled_sample(labeled_df, output_dir)

    # Step 1b — Merge human labels (override Claude where humans disagree)
    human_path = Path(output_dir) / "human_labels.csv"
    if human_path.exists():
        human_df = pd.read_csv(human_path)
        human_df = human_df[(human_df["intimacy_label"] > 0) & (human_df["warmth_label"] > 0)]
        if len(human_df) > 0:
            human_df = human_df.drop_duplicates(subset=["message_id"], keep="last")
            labeled_df = labeled_df.set_index("message_id")
            for _, row in human_df.iterrows():
                mid = row["message_id"]
                if mid in labeled_df.index:
                    labeled_df.loc[mid, "intimacy_label"] = row["intimacy_label"]
                    labeled_df.loc[mid, "warmth_label"] = row["warmth_label"]
            labeled_df = labeled_df.reset_index()
            n_overrides = len(human_df[human_df["message_id"].isin(labeled_df["message_id"])])
            print(f"  Human labels: {len(human_df)} total, "
                  f"{n_overrides} overriding Claude labels")

    # Step 2 — Train self-disclosure CLASSIFIER
    intimacy_result = train_classifier(
        labeled_df, "intimacy", "intimacy_label",
        low_range=[1, 2], high_range=[3, 4, 5],
        output_dir=output_dir)

    # Step 3 — Train responsiveness REGRESSOR
    warmth_result = train_regressor(
        labeled_df, "warmth", "warmth_label",
        output_dir=output_dir)

    # Step 3b — Save trained models to disk
    print("\n  Saving trained models...")
    _save_models(intimacy_result, warmth_result, output_dir)

    # Step 4 — Score all emails
    print(f"\n  Scoring all {len(df):,} emails...")
    if intimacy_result:
        df = classify_all_emails(df, intimacy_result["pipeline"], "intimacy_score")
    else:
        df["intimacy_score"] = 0.5

    if warmth_result:
        df = regress_all_emails(df, warmth_result["pipeline"], "warmth_score")
    else:
        df["warmth_score"] = 0.5

    # Step 5 — VADER sentiment
    df = score_sentiment(df)

    # Step 6 — Report correlations
    corr = df["intimacy_score"].corr(df["warmth_score"])
    corr_sent_int = df["intimacy_score"].corr(df["sentiment_score"])
    corr_sent_warm = df["warmth_score"].corr(df["sentiment_score"])
    print(f"\n  Score correlations:")
    print(f"    intimacy ↔ warmth:    r={corr:.3f}")
    print(f"    intimacy ↔ sentiment: r={corr_sent_int:.3f}")
    print(f"    warmth   ↔ sentiment: r={corr_sent_warm:.3f}")

    # Step 7 — Plot comparison
    _plot_model_comparison({
        "intimacy": {
            "results": intimacy_result["results"] if intimacy_result else None,
            "mode": "classification",
        },
        "warmth": {
            "results": warmth_result["results"] if warmth_result else None,
            "mode": "regression",
        },
    }, output_dir)

    # Save scored emails
    df[["message_id", "sender", "intimacy_score", "warmth_score",
        "sentiment_score"]].to_csv(out / "email_scores.csv", index=False)

    print(f"\n  Stage 1 complete (train mode).")
    if intimacy_result:
        print(f"    Intimacy: {intimacy_result['best_model']} (classifier)")
    if warmth_result:
        print(f"    Warmth: {warmth_result['best_model']} (regressor)")
    print(f"    Sentiment: VADER (rule-based)")

    return df
