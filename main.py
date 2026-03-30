"""
The Social World of Enron — Main Analysis Runner
RAF620M — University of Iceland

Run this file to execute all 4 analyses:
  python main.py

Results are saved to data/results/ and can then be
explored in the Streamlit UI:
  streamlit run app.py
"""

import sys
from pathlib import Path
from src.loader import load_emails, load_processed, save_processed
from src.network import run_network_analysis
from src.sentiment import run_sentiment_analysis
from src.romance import run_romance_analysis

# Paths
RAW_DATA_PATH = "data/raw/maildir"
PROCESSED_PATH = "data/processed/emails.csv"
RESULTS_PATH = "data/results"


def main():
    print("=" * 60)
    print("  The Social World of Enron — Behavioral Analytics")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1 — Load emails
    # ----------------------------------------------------------------
    # If we already parsed and saved the emails before, load the fast
    # CSV version. Otherwise parse the raw files (takes ~15 mins).
    # ----------------------------------------------------------------
    print("\n[1/4] Loading emails...")

    if Path(PROCESSED_PATH).exists():
        df = load_processed(PROCESSED_PATH)
    else:
        print("Processed file not found — parsing raw emails (this takes ~15 mins)...")
        df = load_emails(RAW_DATA_PATH)
        save_processed(df, PROCESSED_PATH)

    # Use a smaller sample for faster testing — remove this line for full run
    df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

    print(f"  {len(df):,} emails loaded")
    print(f"  {df['sender'].nunique():,} unique senders")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ----------------------------------------------------------------
    # STEP 2 — Social Network Analysis
    # Builds the graph, classifies each person (Hub / Gatekeeper etc.)
    # and detects communities.
    # ----------------------------------------------------------------
    results = Path(RESULTS_PATH)
    if (results / "person_classes.csv").exists() and (results / "communities.json").exists():
        print("\n[2/4] Social network analysis — SKIPPED (results already exist)")
    else:
        print("\n[2/4] Running social network analysis...")
        graph, person_df = run_network_analysis(df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # STEP 3 — Friends & Enemies (Sentiment Analysis)
    # Scores the sentiment of every email, then computes the
    # average sentiment between every pair of people.
    # ----------------------------------------------------------------
    if (results / "pair_sentiments.csv").exists():
        print("\n[3/4] Sentiment analysis — SKIPPED (results already exist)")
    else:
        print("\n[3/4] Running sentiment analysis (friends & enemies)...")
        df, pairs_df = run_sentiment_analysis(df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # STEP 4 — Office Romance Detection
    # Labels 500 emails with Claude, trains a classifier,
    # applies it to all emails, finds romantic pairs.
    #
    # NOTE: Requires ANTHROPIC_API_KEY environment variable to be set.
    # On first run this calls the Claude API (~500 calls).
    # On subsequent runs it loads saved labels from disk.
    # ----------------------------------------------------------------
    if (results / "romance_pairs.csv").exists():
        print("\n[4/4] Romance detection — SKIPPED (results already exist)")
    else:
        print("\n[4/4] Running romance detection...")
        df, romance_pairs_df, _ = run_romance_analysis(df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # DONE
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  All analyses complete!")
    print(f"  Results saved to: {RESULTS_PATH}/")
    print("  Launch the UI with:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
